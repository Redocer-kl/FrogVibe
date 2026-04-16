# FrogVibe — AI-Powered Interactive Avatar

A real-time computer vision system that transforms human emotions and hand gestures into a branded digital character.

The goal of this project was to bridge computer vision technology with brand identity, creating a low-latency interactive avatar for educational videos and online programming lessons.

## Key Features

* **Emotion Recognition**: Real-time facial expression analysis (Happy, Sad, Neutral) using deep learning models.
* **Gesture Control**: Hand tracking and landmark detection to trigger specific avatar poses (Pointing, Peace sign).
* **State Priority Logic**: Intelligent processing that prioritizes intentional hand gestures over facial expressions for precise presentation control.
* **Branded UI**: Custom real-time interface with alpha-blending for high-quality character rendering and telemetry display.

## Tech Stack

* **Computer Vision**: OpenCV, MediaPipe (Hand Tracking)
* **Deep Learning**: DeepFace (Emotion Analysis), TensorFlow
* **Core Logic**: Python 3.11, NumPy (Matrix-based image processing)

## Architecture

The system operates on a dual-stream processing pipeline:

1.  **Geometric Stream**: MediaPipe extracts 21 hand landmarks to identify gestures via finger-state logic.
2.  **Neural Stream**: DeepFace analyzes facial regions every N frames to maintain high FPS while tracking emotional shifts.
3.  **Synthesis**: A priority-based controller merges both streams to update the avatar state.

---

### How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/Redocer-kl/FrogVibe.git](https://github.com/Redocer-kl/FrogVibe.git)
   ```

2. Install dependencies:

    ```Bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```Bash
    python image.py
    ```
