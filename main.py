
import cv2
import time
import os
import csv
import math
import queue
import threading
import winsound
import numpy as np
import requests
import mediapipe as mp

from collections import deque
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= SETTINGS =================

CAM_INDEX = 0

WARN_AFTER = 2.0
BEEP_AFTER = 5.0
BEEP_INTERVAL = 1.0

SOS_HOLD = 2.0
SOS_POPUP_SECONDS = 10

TURN_THRESHOLD = 0.35
PHONE_CONF = 0.35
KP_CONF = 0.4

CALIB_SECONDS = 5
DOWN_FACTOR = 0.70

NOD_WINDOW = 20
NODS_FOR_DROWSY = 3
DROWSY_HOLD = 6

SPEAK_COOLDOWN = 8
VOICE_ENABLED = True

# None enables approximate IP location lookup
MANUAL_LOCATION = None

FONT = cv2.FONT_HERSHEY_SIMPLEX

os.makedirs("alerts", exist_ok=True)

# ================= YOLO MODELS =================

pose_model = YOLO("yolo11n-pose.pt")
det_model = YOLO("yolo11n.pt")

PHONE_ID = 67

# ================= HAND MODEL =================

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = vision.HandLandmarker.create_from_options(options)

# ================= LOCATION =================

location = {
    "text": "Locating...",
    "link": ""
}

def fetch_location():

    if MANUAL_LOCATION:
        location["text"] = MANUAL_LOCATION
        return

    try:
        response = requests.get(
            "https://ipinfo.io/json",
            timeout=5
        )

        response.raise_for_status()
        data = response.json()

        city = data.get("city", "")
        region = data.get("region", "")
        country = data.get("country", "")

        location["text"] = (
            f"{city}, {region}, {country} (approx.)"
        )

        if data.get("loc"):
            location["link"] = (
                "https://maps.google.com/?q="
                + data["loc"]
            )

    except Exception:
        location["text"] = "Location unavailable"

threading.Thread(
    target=fetch_location,
    daemon=True
).start()

# ================= BEEP SOUND =================

beep_lock = threading.Lock()

def beep(emergency=False):

    def play():

        if not beep_lock.acquire(blocking=False):
            return

        try:

            if emergency:

                # Emergency SOS sound
                winsound.Beep(1800, 250)
                winsound.Beep(1000, 250)
                winsound.Beep(1800, 250)

            else:

                # Normal distraction sound
                winsound.Beep(1200, 400)

        except RuntimeError:
            print("\a", end="", flush=True)

        finally:
            beep_lock.release()

    threading.Thread(
        target=play,
        daemon=True
    ).start()

# ================= VOICE WARNINGS =================

speech_q = queue.Queue(maxsize=3)
last_spoken = {}
SPEECH_OK = VOICE_ENABLED

def speech_worker():

    global SPEECH_OK

    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 170)

    except Exception:
        SPEECH_OK = False
        return

    while True:

        message = speech_q.get()

        try:
            engine.say(message)
            engine.runAndWait()

        except Exception:
            pass

if VOICE_ENABLED:

    threading.Thread(
        target=speech_worker,
        daemon=True
    ).start()

def speak(message, key=None):

    key = key or message
    now = time.monotonic()

    if (
        SPEECH_OK
        and now - last_spoken.get(key, -100)
        >= SPEAK_COOLDOWN
    ):

        try:
            speech_q.put_nowait(message)
            last_spoken[key] = now

        except queue.Full:
            pass

# ================= ALERT LOGGING =================

def log_event(kind, frame):

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    path = (
        f"alerts/{kind}_{timestamp}_"
        f"{time.time_ns()}.jpg"
    )

    saved = cv2.imwrite(path, frame)

    if not saved:
        path = "Snapshot unavailable"

    new_file = not os.path.exists("alerts/log.csv")

    with open(
        "alerts/log.csv",
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if new_file:

            writer.writerow([
                "Time",
                "Event",
                "Location",
                "Snapshot"
            ])

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            kind,
            location["text"],
            path
        ])

# ================= HEAD TURN DETECTION =================

def head_turned(kp, c):

    if c[0] < KP_CONF:
        return True

    left_ear = c[3] > KP_CONF
    right_ear = c[4] > KP_CONF

    if left_ear and right_ear:

        ear_distance = (
            abs(kp[3][0] - kp[4][0]) + 1e-6
        )

        middle = (
            kp[3][0] + kp[4][0]
        ) / 2

        return (
            abs(kp[0][0] - middle)
            / ear_distance
            > TURN_THRESHOLD
        )

    return left_ear != right_ear

# ================= PHONE DETECTION =================

def phone_in_use(phones, kp, c):

    for x1, y1, x2, y2 in phones:

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        size = max(x2 - x1, y2 - y1)

        for j in (9, 10, 0):

            distance = math.hypot(
                kp[j][0] - cx,
                kp[j][1] - cy
            )

            if (
                c[j] > KP_CONF
                and distance < 3 * size
            ):
                return True

    return False

# ================= LOOKING DOWN =================

def head_drop_ratio(kp, c):

    if (
        c[0] < KP_CONF
        or c[5] < KP_CONF
        or c[6] < KP_CONF
    ):
        return None

    shoulder_width = abs(
        kp[5][0] - kp[6][0]
    )

    if shoulder_width < 20:
        return None

    shoulder_y = (
        kp[5][1] + kp[6][1]
    ) / 2

    return (
        shoulder_y - kp[0][1]
    ) / shoulder_width

# ================= OPEN PALM SOS =================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

last_timestamp = -1

def open_palm_detected(frame, view):

    global last_timestamp

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = max(
        int(time.monotonic() * 1000),
        last_timestamp + 1
    )

    last_timestamp = timestamp

    result = hands.detect_for_video(
        mp_image,
        timestamp
    )

    if not result.hand_landmarks:
        return False

    landmarks = result.hand_landmarks[0]

    h, w = frame.shape[:2]

    p = np.array([
        (v.x * w, v.y * h, v.z * w)
        for v in landmarks
    ])

    # Draw hand skeleton

    for a, b in HAND_CONNECTIONS:

        cv2.line(
            view,
            tuple(p[a][:2].astype(int)),
            tuple(p[b][:2].astype(int)),
            (0, 255, 0),
            2
        )

    for point in p:

        cv2.circle(
            view,
            tuple(point[:2].astype(int)),
            4,
            (0, 255, 255),
            -1
        )

    # Use 3D landmarks when available

    if result.hand_world_landmarks:

        points = np.array([
            (v.x, v.y, v.z)
            for v in result.hand_world_landmarks[0]
        ])

    else:
        points = p

    # Calculate finger joint angle

    def angle(a, b, c):

        u = points[a] - points[b]
        v = points[c] - points[b]

        cosine = np.dot(u, v) / (
            np.linalg.norm(u)
            * np.linalg.norm(v)
            + 1e-9
        )

        return np.degrees(
            np.arccos(
                np.clip(cosine, -1, 1)
            )
        )

    # Check four extended fingers

    finger_angles = [
        angle(i, i + 1, i + 3)
        for i in (5, 9, 13, 17)
    ]

    fingers_open = all(
        a > 145 for a in finger_angles
    )

    # Check extended thumb

    palm_width = (
        np.linalg.norm(points[5] - points[17])
        + 1e-6
    )

    thumb_open = (
        np.linalg.norm(points[4] - points[5])
        / palm_width > 0.65
    )

    palm_open = fingers_open and thumb_open

    if palm_open:

        cv2.putText(
            view,
            "OPEN PALM DETECTED",
            (15, 80),
            FONT,
            0.65,
            (0, 255, 0),
            2
        )

    return palm_open

# ================= SESSION STATISTICS =================

session_start = time.monotonic()

present_time = 0.0
distracted_time = 0.0

distraction_events = 0
sos_events = 0

reason_counts = {}

def print_summary():

    total = time.monotonic() - session_start

    focus = (
        100 * (1 - distracted_time / present_time)
        if present_time > 0
        else 0
    )

    print("\n========== SESSION SUMMARY ==========")

    print(f"Session length     : {total / 60:.1f} min")
    print(f"Time in view       : {present_time / 60:.1f} min")
    print(f"Focus percentage   : {focus:.1f}%")
    print(f"Distraction events : {distraction_events}")
    print(f"SOS events         : {sos_events}")

    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}")

    print("=====================================")

    new_file = not os.path.exists(
        "alerts/sessions.csv"
    )

    with open(
        "alerts/sessions.csv",
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if new_file:

            writer.writerow([
                "Date",
                "Duration",
                "Time in view",
                "Focus percentage",
                "Distractions",
                "SOS events",
                "Reasons"
            ])

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M"),
            f"{total / 60:.1f}",
            f"{present_time / 60:.1f}",
            f"{focus:.1f}",
            distraction_events,
            sos_events,
            reason_counts
        ])

# ================= WEBCAM =================

cap = cv2.VideoCapture(CAM_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():

    hands.close()
    cap.release()

    raise RuntimeError("Could not open webcam.")

# ================= INITIAL STATE =================

distract_since = None
episode_counted = False
distract_logged = False

sos_since = None
sos_latched = False
sos_until = 0
sos_cooldown = 0

last_beep = -100
popup_open = False

calib_start = None
calib_vals = []
baseline = None

was_down = False
nods = deque()
drowsy_until = 0

prev_t = time.monotonic()

# ================= MAIN LOOP =================

try:

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)

        now = time.monotonic()

        dt = now - prev_t
        prev_t = now

        H, W = frame.shape[:2]

        # ================= YOLO PREDICTIONS =================

        pr = pose_model(
            frame,
            verbose=False,
            imgsz=480
        )[0]

        dr = det_model(
            frame,
            classes=[PHONE_ID],
            conf=PHONE_CONF,
            verbose=False,
            imgsz=480
        )[0]

        view = pr.plot()

        phones = [
            tuple(map(int, b.xyxy[0].tolist()))
            for b in dr.boxes
        ]

        # Draw phone detection boxes

        for x1, y1, x2, y2 in phones:

            cv2.rectangle(
                view,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2
            )

            cv2.putText(
                view,
                "PHONE",
                (x1, max(y1 - 8, 15)),
                FONT,
                0.6,
                (0, 165, 255),
                2
            )

        person = False
        reasons = []

        palm_open = open_palm_detected(
            frame, view
        )

        calibrated = baseline is not None

        # ================= PERSON DETECTION =================

        if (
            len(pr.boxes) > 0
            and pr.keypoints is not None
            and pr.keypoints.conf is not None
        ):

            wh = pr.boxes.xywh.cpu().numpy()

            i = int(
                np.argmax(wh[:, 2] * wh[:, 3])
            )

            kp = pr.keypoints.xy[i].cpu().numpy()
            c = pr.keypoints.conf[i].cpu().numpy()

            person = True

            ratio = head_drop_ratio(kp, c)

            # Initial calibration

            if baseline is None:

                if calib_start is None:
                    calib_start = now

                if ratio is not None:
                    calib_vals.append(ratio)

                if (
                    now - calib_start >= CALIB_SECONDS
                    and len(calib_vals) >= 10
                ):

                    baseline = float(
                        np.median(calib_vals)
                    )

                    calibrated = True

            else:

                # Looking-down detection

                is_down = (
                    ratio is not None
                    and ratio < baseline * DOWN_FACTOR
                )

                if is_down:
                    reasons.append("Looking down")

                # Drowsiness estimation

                if is_down and not was_down:
                    nods.append(now)

                was_down = is_down

                while (
                    nods
                    and now - nods[0] > NOD_WINDOW
                ):
                    nods.popleft()

                if len(nods) >= NODS_FOR_DROWSY:

                    drowsy_until = now + DROWSY_HOLD
                    nods.clear()

                if now < drowsy_until:
                    reasons.append("Possible drowsiness")

                # Head-turn detection

                if head_turned(kp, c):
                    reasons.append("Head turned")

                # Phone detection

                if phone_in_use(phones, kp, c):
                    reasons.append("Phone in use")

        else:

            if baseline is None:

                calib_start = None
                calib_vals.clear()

            was_down = False

        # ================= SIMPLE SOS GESTURE =================

        if palm_open:

            if sos_since is None:
                sos_since = now

            held = now - sos_since

            if (
                held >= SOS_HOLD
                and now >= sos_cooldown
                and not sos_latched
            ):

                sos_until = now + SOS_POPUP_SECONDS
                sos_cooldown = sos_until

                sos_latched = True
                sos_events += 1

                log_event("SOS", frame)

                # Immediate emergency beep

                beep(emergency=True)
                last_beep = now

                speak(
                    "Emergency alert triggered",
                    "sos"
                )

                print("\nSOS TRIGGERED!")
                print("Location:", location["text"])

        else:

            sos_since = None
            sos_latched = False

        sos_active = now < sos_until

        # ================= DISTRACTION TIMER =================

        distracted = (
            person
            and calibrated
            and len(reasons) > 0
        )

        if distracted:

            if distract_since is None:
                distract_since = now

            dur = now - distract_since

        else:

            distract_since = None
            episode_counted = False
            distract_logged = False
            dur = 0

        if person and calibrated:

            present_time += dt

            if distracted:
                distracted_time += dt

        reason = " + ".join(reasons)

        # ================= STATUS AND WARNINGS =================

        if not person:

            status = "NO PERSON IN VIEW"
            color = (100, 100, 100)

        elif not calibrated:

            left = (
                max(
                    0,
                    CALIB_SECONDS - (now - calib_start)
                )
                if calib_start is not None
                else CALIB_SECONDS
            )

            status = f"CALIBRATING... ({left:.0f}s)"
            color = (200, 120, 0)

        elif dur >= BEEP_AFTER:

            # RED ALERT AFTER 5 SECONDS

            status = f"ALERT! {reason}"
            color = (0, 0, 255)

            if not sos_active:

                # Repeating beep every second

                if now - last_beep >= BEEP_INTERVAL:

                    beep()
                    last_beep = now

                if "Possible drowsiness" in reasons:

                    speak(
                        "Drowsiness alert. Please take a break.",
                        "drowsy"
                    )

                elif "Phone in use" in reasons:

                    speak(
                        "Please stop using your phone.",
                        "phone"
                    )

                elif "Looking down" in reasons:

                    speak(
                        "Please look forward.",
                        "down"
                    )

                else:

                    speak(
                        "Distraction alert. Please focus.",
                        "alert"
                    )

            if not distract_logged:

                log_event("DISTRACTION", frame)
                distract_logged = True

        elif dur >= WARN_AFTER:

            # ORANGE WARNING AFTER 2 SECONDS

            status = f"WARNING: {reason}"
            color = (0, 140, 255)

            if not sos_active:

                speak(
                    "Please focus",
                    "warn"
                )

        elif distracted:

            status = f"CHECKING: {reason}"
            color = (0, 180, 180)

        else:

            # GREEN STATUS

            status = "FOCUSED"
            color = (0, 160, 0)

        # Count distraction events

        if (
            distracted
            and dur >= WARN_AFTER
            and not episode_counted
        ):

            episode_counted = True
            distraction_events += 1

            for r in reasons:

                reason_counts[r] = (
                    reason_counts.get(r, 0) + 1
                )

        # ================= STATUS BAR =================

        cv2.rectangle(
            view,
            (0, 0),
            (W, 50),
            color,
            -1
        )

        cv2.putText(
            view,
            status,
            (15, 32),
            FONT,
            0.55,
            (255, 255, 255),
            2
        )

        # ================= DISTRACTION TIMER DISPLAY =================

        if distracted:

            cv2.putText(
                view,
                f"Unfocused: {dur:.1f}s",
                (15, 110),
                FONT,
                0.65,
                (0, 165, 255),
                2
            )

        # ================= SOS INSTRUCTIONS =================

        hint = f"SOS: Show open palm for {SOS_HOLD:.0f}s"

        if sos_since is not None and not sos_latched:

            held = min(
                now - sos_since,
                SOS_HOLD
            )

            hint = (
                f"SOS: Hold palm "
                f"{held:.1f}/{SOS_HOLD:.0f}s"
            )

        if sos_active:
            hint = "SOS ALERT ACTIVATED"

        cv2.rectangle(
            view,
            (0, H - 65),
            (W, H),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            view,
            hint,
            (15, H - 40),
            FONT,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            view,
            "R: Recalibrate | Q: Quit",
            (15, H - 12),
            FONT,
            0.5,
            (255, 255, 255),
            1
        )

        # ================= SOS EMERGENCY POPUP =================

        if sos_active:

            cv2.rectangle(
                view,
                (0, 0),
                (W - 1, H - 1),
                (0, 0, 255),
                12
            )

            # Repeating emergency beep

            if now - last_beep >= BEEP_INTERVAL:

                beep(emergency=True)
                last_beep = now

            popup = np.zeros(
                (320, 800, 3),
                dtype=np.uint8
            )

            popup[:] = (0, 0, 180)

            cv2.putText(
                popup,
                "SOS - NEED HELP!",
                (40, 80),
                FONT,
                2,
                (255, 255, 255),
                4
            )

            cv2.putText(
                popup,
                f"Location: {location['text']}",
                (40, 145),
                FONT,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                popup,
                f"Time: {time.strftime('%H:%M:%S')}",
                (40, 190),
                FONT,
                0.8,
                (255, 255, 255),
                2
            )

            remaining = max(
                0,
                sos_until - now
            )

            cv2.putText(
                popup,
                f"Alert active: {remaining:.0f}s",
                (40, 235),
                FONT,
                0.7,
                (255, 255, 255),
                2
            )

            if location["link"]:

                cv2.putText(
                    popup,
                    location["link"],
                    (40, 280),
                    FONT,
                    0.6,
                    (255, 255, 255),
                    1
                )

            cv2.imshow("SOS ALERT", popup)

            popup_open = True

        elif popup_open:

            try:
                cv2.destroyWindow("SOS ALERT")
            except cv2.error:
                pass

            popup_open = False

        # ================= DISPLAY =================

        cv2.imshow(
            "Focus Guard + SOS",
            view
        )

        key = cv2.waitKey(1) & 0xFF

        # Quit application

        if key == ord("q"):
            break

        # Recalibrate

        if key == ord("r"):

            baseline = None
            calib_start = None
            calib_vals = []

            was_down = False
            nods.clear()
            drowsy_until = 0

            distract_since = None
            episode_counted = False
            distract_logged = False

            sos_since = None

            print("Recalibrating...")

except KeyboardInterrupt:

    print("\nSession stopped.")

finally:

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    print_summary()