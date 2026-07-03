# MediaPipe Hand Demos

A collection of real-time hand detection scripts built with OpenCV and MediaPipe. This is the foundation of my AI-powered Sign Language Translator project for the deaf community in Uzbekistan.

## demos

**webcam_test.py** — basic webcam feed with OpenCV. Tests camera input, frame flipping, and live video display.

**hand_detection.py** — real-time hand landmark detection using MediaPipe. Detects 21 hand landmarks and draws the skeleton overlay on a live webcam feed.

## setup

```bash
py -3.11 -m venv venv
.\venv\Scripts\activate
pip install opencv-python==4.9.0.80 mediapipe==0.10.14 numpy==1.26.4
```

## run

```bash
python webcam_test.py
python hand_detection.py
```

Press Q to exit.

## tech stack
- Python 3.11
- OpenCV 4.9
- MediaPipe 0.10.14

## part of
This repo is Phase 1 of the Sign Language Translator project — an AI app to help the deaf community in Uzbekistan communicate in real time.
