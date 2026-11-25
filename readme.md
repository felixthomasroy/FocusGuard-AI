
# Focus Guard + SOS

An AI-powered real-time distraction detection and
gesture-based SOS alert system using computer vision.

## Features

- Real-time head-turn detection
- Mobile phone usage detection
- Looking-down detection
- Possible drowsiness detection
- Gesture-based SOS activation
- Audible beep and voice alerts
- Approximate location detection
- Automatic alert screenshots
- CSV-based alert history
- Session focus analysis

## Technologies Used

- Python 3.14.6
- YOLO11
- MediaPipe
- OpenCV
- NumPy
- Pyttsx3
- Computer Vision
- Deep Learning

## Installation

Clone the repository and open the project folder.

Create a virtual environment:

    python -m venv venv

Activate the environment on Windows:

    .\venv\Scripts\Activate.ps1

Install the dependencies:

    pip install -r requirements.txt

Download the MediaPipe hand landmark model:

    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"

The YOLO model weights are automatically downloaded
by Ultralytics when required.

## Usage

Run the application:

    python main.py

Look straight at the webcam for initial calibration.

## Alert Mechanism

| Situation | Response |
|---|---|
| Distraction for 2 seconds | Warning |
| Distraction for 5 seconds | Repeating beep |
| Open palm for 2 seconds | SOS alert |
| Focused | Normal status |

## Keyboard Controls

- R - Recalibrate
- Q - Quit and display session summary

## Privacy

Alert screenshots and session records are stored locally.
Model weights, virtual environments and private alert
records are excluded from version control.

Approximate location lookup uses an external IP-based
location service.

## Limitations

Detection accuracy depends on lighting, camera quality
and user positioning.

Drowsiness detection is heuristic. The SOS feature
generates local alerts and does not automatically
contact emergency services.

This project is an educational prototype.

## Author

Felix Thomas Roy
