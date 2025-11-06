import cv2
import numpy as np
from .base_exercise import Exercise
from utils import calculate_angle

class Pushup(Exercise):
    def __init__(self):
        super().__init__()
        self.feedback = "Start"

    def process_frame(self, frame, landmarks, mp_pose):
        # 1. Extract relevant landmarks for left side of the body
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # 2. Calculate elbow angle
        angle = calculate_angle(shoulder, elbow, wrist)

        # Visualize angle (optional)
        cv2.putText(frame, str(int(angle)),
                           tuple(np.multiply(elbow, [1280, 720]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # 3. Push-up logic (rep counting and feedback)
        if angle > 160:
            self.stage = "down"
            self.feedback = "Go Down"

        if angle < 90 and self.stage == 'down':
            self.stage = "up"
            self.rep_counter += 1
            self.feedback = "Go Up"

        # 4. Display feedback on the frame
        self.display_feedback(frame)

        return frame


























# from .base_exercise import Exercise
# from utils import calculate_angle # Assuming you have a utils file for helpers
# import cv2

# class Squat(Exercise):
#     def __init__(self, side="left", angle_up=150, angle_down=100):
#         super().__init__()
#         self.side = side
#         self.angle_up = angle_up
#         self.angle_down = angle_down

#     def process_frame(self, frame, landmarks, mp_pose):
#         # Select side
#         if self.side == "left":
#             hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
#                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#             knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
#                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
#             ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
#                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
#         else:
#             hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#             knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
#             ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
#                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

#         # Calculate angle
#         angle = calculate_angle(hip, knee, ankle)

#         # Rep logic
#         feedback = ""
#         if angle > self.angle_up:
#             self.stage = "up"
#         if angle < self.angle_down and self.stage == "up":
#             self.stage = "down"
#             self.rep_counter += 1
#             feedback = "Great Rep!"

#         # Draw info
#         h, w, _ = frame.shape
#         cv2.putText(frame, f'REPS: {self.rep_counter}',
#                     (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
#                     (0, 0, 0), 2, cv2.LINE_AA)
#         cv2.putText(frame, feedback, (15, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7,
#                     (0, 255, 0), 2, cv2.LINE_AA)

#         # Optional: Draw angle near the knee
#         cx, cy = int(knee[0] * w), int(knee[1] * h)
#         cv2.putText(frame, str(int(angle)), (cx, cy),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

#         return frame
