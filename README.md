# 🏋️ Sports Performance Analysis Prototype

## Overview

The Sports Performance Analysis Prototype is a real-time, camera-based exercise analysis system built in Python. It uses **MediaPipe Pose** for per-frame 2D human pose estimation and **OpenCV** for video capture and visualization. The system processes each webcam frame independently—extracting body keypoints, computing joint angles via trigonometric functions, and applying exercise-specific state-machine logic to count repetitions and deliver live corrective feedback as on-screen overlays. The architecture follows a modular, object-oriented design with an abstract base class for exercises, making it straightforward to add new movement patterns without modifying the core pipeline. No temporal tracking, trajectory analysis, or external camera calibration is required; the prototype operates under a fixed-viewpoint assumption and focuses on single-athlete, single-exercise analysis per session.

---

## Features

| Feature | Description |
|---|---|
| **Real-Time Pose Estimation** | Detects 33 body landmarks per frame using Google's MediaPipe Pose model. |
| **Per-Frame Joint Angle Computation** | Calculates angles between any three landmarks using `arctan2`-based trigonometry. |
| **Repetition Counting** | Tracks exercise stages (up / down) via a finite state machine and increments reps on valid transitions. |
| **Live Corrective Feedback** | Displays actionable cues ("Go Down", "Go Up") overlaid on the video feed. |
| **Supported Exercises** | Simple Squat · Advanced Squat (configurable side & angle thresholds) · Pushup |
| **Modular & Extensible** | New exercises are added by subclassing `Exercise` and overriding a single method. |
| **Configurable Camera** | Supports local webcam or remote IP camera (e.g., via DroidCam / IP Webcam). |

---

## System Architecture

```
┌──────────────┐       ┌──────────────────┐     ┌─────────────────────┐
│  Video Input │ ────▶ │  Frame Capture   │ ───▶│  Color Conversion   │
│  (Webcam /   │       │  (OpenCV)        │     │  BGR → RGB          │
│   IP Camera) │       └──────────────────┘     └────────┬────────────┘
└──────────────┘                                         │
                                                         ▼
                                              ┌─────────────────────┐
                                              │  MediaPipe Pose     │
                                              │  Estimation         │
                                              │  (33 Landmarks)     │
                                              └────────┬────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────────┐
                                              │  Exercise Analyzer  │
                                              │  (Angle Calc →      │
                                              │   State Machine →   │
                                              │   Rep Count)        │
                                              └────────┬────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────────┐
                                              │  Overlay Feedback   │
                                              │  & Render Frame     │
                                              │  (OpenCV)           │
                                              └─────────────────────┘
```

---

## Detailed Workflow

### Step 1 — Video Capture

The application initializes a video stream from a local webcam (`cv2.VideoCapture(0)`) at **1920 × 1080** resolution. An optional remote camera URL (e.g., `http://<ip>:8080/video`) is supported for mobile-phone-as-camera setups.

### Step 2 — Preprocessing

Each captured BGR frame is converted to RGB (`cv2.cvtColor`) before being passed to MediaPipe. The frame is temporarily marked as non-writable during inference to improve performance by avoiding unnecessary copies. After inference, the frame is converted back to BGR for OpenCV rendering.

### Step 3 — Pose Estimation

MediaPipe Pose processes the RGB frame and returns **33 2D landmarks** (x, y in normalized image coordinates [0, 1], plus a visibility score). The model runs with:
- `min_detection_confidence = 0.5`
- `min_tracking_confidence = 0.5`

Each frame is processed **independently**; there is no cross-frame trajectory or temporal smoothing.

### Step 4 — Exercise-Specific Analysis

The detected landmarks are forwarded to the selected exercise analyzer, which performs:

1. **Landmark Extraction** — Retrieves the relevant joint coordinates (e.g., hip, knee, ankle for squats).
2. **Angle Calculation** — Calls `calculate_angle(a, b, c)` from `utils.py` to compute the angle at the middle joint.
3. **State Machine Logic** — Compares the angle against configurable thresholds to determine the current exercise stage and detect stage transitions that count as valid repetitions.
4. **Feedback Generation** — Sets a feedback string (e.g., "Go Down", "Go Up") based on the current state.

### Step 5 — Visualization & Display

The processed frame is annotated with:
- A colored status box showing **rep count** and **feedback text**.
- The computed **joint angle** rendered near the relevant landmark.
- Full **skeleton overlay** with landmarks and connections drawn by MediaPipe's drawing utilities.

The annotated frame is displayed in a window titled *"Exercise Performance Analyzer"*. The loop continues until the user presses **ESC**.

---

## Methodology

### Angle Calculation (`utils.py`)

The angle at a joint B, formed by points A, B, C, is calculated using the two-argument arctangent:

```
θ = | atan2(Cy − By, Cx − Bx) − atan2(Ay − By, Ax − Bx) |
```

The result is converted from radians to degrees. If θ > 180°, it is remapped to 360° − θ to ensure the output is always in the range [0°, 180°].

> **Why `arctan2` over dot-product?**  `arctan2` is numerically stable for near-zero and near-180° angles, avoiding division-by-zero issues that can arise with the `arccos(dot / (mag × mag))` approach.

### Repetition State Machine

Each exercise uses a two-state finite state machine:

```
        angle > threshold_up
   ┌─────────────────────────────┐
   │                             │
   │                             ▼
┌──┴───┐                      ┌──────┐
│ DOWN │                      │  UP  │
└──────┘                      └──┬───┘
   ▲                             │
   │   angle < threshold_down    │
   │   (rep_counter += 1)        │
   └─────────────────────────────┘
```

- **UP → DOWN transition** (angle crosses below `threshold_down`): a valid repetition is counted.
- **DOWN → UP transition** (angle crosses above `threshold_up`): the system resets and waits for the next rep.
- No rep is counted unless a full UP → DOWN cycle completes, preventing double-counting.

### Object-Oriented Exercise Design

```
Exercise (base_exercise.py)          ← Abstract base class
 ├── rep_counter, stage, feedback    ← Shared state
 ├── process_frame()                 ← Override per exercise
 └── display_feedback()              ← Common HUD rendering
      │
      ├── SimpleSquat                ← Basic squat logic
      ├── AdvancedSquat              ← Configurable side, thresholds
      └── Pushup                     ← Elbow-angle-based analysis
```

Adding a new exercise requires only:
1. Creating a new file in `exercises/`.
2. Subclassing `Exercise`.
3. Implementing `process_frame()` with the joint selection, angle thresholds, and state transitions specific to that movement.

---

## Project Structure

```
SportsPerformanceAnalysisPrototype/
└── src/
    └── SportsPerformance/
        ├── main.py                    # Entry point — video loop & exercise selection
        ├── utils.py                   # Utility functions (angle calculation)
        └── exercises/
            ├── __init__.py            # Package initializer
            ├── base_exercise.py       # Abstract Exercise base class
            ├── simple_squat.py        # Simple squat analyzer
            ├── advanced_squat.py      # Advanced squat analyzer (configurable)
            └── pushup.py              # Pushup analyzer
```

---

## Supported Exercises

### Simple Squat
Tracks the knee angle on one side of the body using default thresholds.

### Advanced Squat
Extends the simple squat with configurable parameters:

| Parameter | Default | Description |
|---|---|---|
| `side` | `"left"` | Which leg to track (`"left"` or `"right"`) |
| `angle_up` | `160°` | Knee angle threshold for the standing position |
| `angle_down` | `90°` | Knee angle threshold for the bottom of the squat |

### Pushup
Tracks the **left elbow angle** (shoulder → elbow → wrist):
- **Up position**: elbow angle > 160°
- **Down position**: elbow angle < 90°

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- A working webcam (built-in or USB)
- GPU to run MediaPipe

### Installation

#### Option A — One-command (Windows)

From the project root, run:

```bat
run_windows.bat
```

What it does:
- Creates `.venv` if it doesn’t exist (`python -m venv .venv`).
- Activates the venv using PowerShell.
- Installs dependencies from `requirements.txt`.
- Runs `src/SportsPerformance/main.py`.

If activation fails on your machine, use **Option B** (manual activation) or run the script from an elevated PowerShell.

You can also double-click `run_windows.bat` in File Explorer.

#### Option B — Manual setup (any OS)

```bash
# Clone the repository
git clone https://github.com/ParthProLegend/SportsPerformanceAnalysisPrototype
cd SportsPerformanceAnalysisPrototype

# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

# Windows (cmd)
# .\.venv\Scripts\activate.bat

# macOS/Linux
# source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt
```

### Running the Application

```bash
cd src/SportsPerformance
python main.py
```

### Selecting an Exercise

Open `main.py` and modify the `exercise_selection` variable:

```python
exercise_selection = "advanced_squat"   # Options: "simple_squat", "advanced_squat", "pushup"
```

### Controls

| Key | Action |
|---|---|
| `ESC` | Exit the application |

---

## Adding a New Exercise

1. **Create** a new file, e.g., `exercises/lunge.py`.
2. **Subclass** `Exercise` and implement `process_frame()`:

```python
from .base_exercise import Exercise
from utils import calculate_angle

class Lunge(Exercise):
    def __init__(self):
        super().__init__()
        self.feedback = "Start"

    def process_frame(self, frame, landmarks, mp_pose):
        # 1. Extract landmarks
        # 2. Calculate angles
        # 3. State machine logic
        # 4. Call self.display_feedback(frame)
        return frame
```

3. **Register** the exercise in `main.py`:

```python
from exercises.lunge import Lunge

# Inside main():
elif exercise_selection == "lunge":
    exercise_analyzer = Lunge()
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Video capture, frame manipulation, and on-screen rendering |
| `mediapipe` | 2D human pose estimation (33 landmarks) |
| `numpy` | Numerical operations for angle computation |

---

## Limitations

- **Single-person only**: MediaPipe Pose estimates a single pose per frame; multi-person scenarios are not supported.
- **No temporal analysis**: Each frame is analyzed independently — no velocity, acceleration, or trajectory tracking is performed.
- **2D only**: Pose estimation is in 2D image coordinates; depth information is not used.
- **Fixed exercise selection**: The exercise must be set before running the application; runtime switching is not yet supported.

---

## Future Enhancements

- 📊 **Session analytics dashboard** — Log rep data and display post-session summaries with plots.
- 🏃 **Multi-exercise runtime switching** — Allow users to switch exercises via keyboard shortcuts during a session.
- 📹 **Video file input** — Analyze pre-recorded videos in addition to live camera feeds.
- 🦾 **Additional exercises** — Lunges, deadlifts, bicep curls, planks, etc.
- 📱 **Remote camera integration** — Seamless setup with IP Webcam / DroidCam.
- 🧠 **Form scoring** — Assign a quality score per rep based on angle consistency and range of motion.
- 📈 **Progress tracking** — Persist session data to JSON/CSV for longitudinal analysis.

---

## Acknowledgments

- [**MediaPipe**](https://mediapipe.dev/) — Google's open-source framework for real-time pose estimation.
- [**OpenCV**](https://opencv.org/) — Industry-standard library for computer vision and image processing.
- [**NumPy**](https://numpy.org/) — Fundamental package for numerical computation in Python.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
