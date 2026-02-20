from .base_exercise import Exercise
from utils import calculate_angle

class SimpleSquat(Exercise):
    """
    A simple squat analyzer with fixed angles and side.
    """
    def __init__(self):
        super().__init__()
        self.feedback = "Start"

    def process_frame(self, frame, landmarks, mp_pose):
        # Using hardcoded left side landmarks
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate angle
        angle = calculate_angle(hip, knee, ankle)

        # Simple rep logic with fixed angles (165 degrees for up, 95 for down)
        if angle > 165:
            self.stage = "up"
            self.feedback = "Go Down"

        if angle < 95 and self.stage == 'up':
            self.stage = "down"
            self.rep_counter += 1
            self.feedback = "Go Up"

        # Display feedback using the base class method
        self.display_feedback(frame)

        return frame

