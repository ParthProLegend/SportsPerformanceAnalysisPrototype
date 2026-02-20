import cv2
import numpy as np
from .base_exercise import Exercise
from utils import calculate_angle

class AdvancedSquat(Exercise):
    def __init__(self, side="left", angle_up=165, angle_down=95):
        """
        Initializes the Advanced Squat analyzer with customizable parameters.

        Args:
            side (str): The side of the body to track ("left" or "right").
            angle_up (int): The knee angle threshold for the 'up' position.
            angle_down (int): The knee angle threshold for the 'down' position.
        """
        super().__init__()
        self.side = side.upper()  # Ensure side is uppercase for landmark names
        self.angle_up = angle_up
        self.angle_down = angle_down
        self.feedback = "Start"

    def process_frame(self, frame, landmarks, mp_pose):
        # --- 1. Landmark Extraction ---
        # Dynamically get the required landmarks based on the selected side
        hip_landmark = getattr(mp_pose.PoseLandmark, f'{self.side}_HIP')
        knee_landmark = getattr(mp_pose.PoseLandmark, f'{self.side}_KNEE')
        ankle_landmark = getattr(mp_pose.PoseLandmark, f'{self.side}_ANKLE')

        hip = [landmarks[hip_landmark.value].x, landmarks[hip_landmark.value].y]
        knee = [landmarks[knee_landmark.value].x, landmarks[knee_landmark.value].y]
        ankle = [landmarks[ankle_landmark.value].x, landmarks[ankle_landmark.value].y]

        # --- 2. Angle Calculation ---
        angle = calculate_angle(hip, knee, ankle)

        # --- 3. Repetition Logic ---
        if angle > self.angle_up:
            self.stage = "up"
            self.feedback = "Go Down"

        if angle < self.angle_down and self.stage == 'up':
            self.stage = "down"
            self.rep_counter += 1
            self.feedback = "Go Up"

        # --- 4. Drawing and Display ---
        # Use the display_feedback method from the base class for the status box
        self.display_feedback(frame)

        # Draw the calculated angle on the knee for visualization
        h, w, _ = frame.shape
        knee_coords = tuple(np.multiply(knee, [w, h]).astype(int))
        cv2.putText(frame, str(int(angle)),
                    knee_coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        return frame

