<<<<<<< HEAD
# IT-FID
=======
# Liveness Detection System

This project is a liveness detection system that uses a pipeline of actions to determine if a person is live. The system uses computer vision to detect facial gestures such as blinking and mouth opening.

## How it Works

The system uses a pipeline of actions, where each action is a specific facial gesture that needs to be performed within a certain time frame. The user is guided through the pipeline, and their facial gestures are monitored in real-time.

The core components of the system are:

- **Action Pipeline**: A pipeline that defines a sequence of actions to be performed.
- **Detectors**: Computer vision models that detect specific facial gestures (e.g., `BlinkDetector`, `MouthOpenDetector`).
- **Camera**: Captures video feed for real-time analysis.
- **UI**: Provides feedback to the user, showing the current action, progress, and results.

The `main.py` file contains the main application logic, including the action pipeline and UI. The `detector.py` file contains the implementation of the facial gesture detectors, which use the `mediapipe` library for facial landmark detection.

## Features

- **Configurable Action Pipeline**: Easily define a sequence of actions to be performed.
- **Real-time Feedback**: The UI provides real-time feedback to the user.
- **Extensible**: New detectors for other facial gestures can be easily added.

## Dependencies

- `mediapipe`
- `opencv-python`

## How to Run

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```
   python main.py
   ```
>>>>>>> 6593a5a ( Added core function of eye and mouth thracking)
