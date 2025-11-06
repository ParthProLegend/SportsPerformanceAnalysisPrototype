import cv2

class Exercise:
    """
    A base class for all exercise analyzers.
    This class defines the common structure and methods that every
    exercise-specific class should implement.
    """
    def __init__(self):
        """
        Initializes common attributes for exercise tracking.
        """
        self.rep_counter = 0
        self.stage = None  # e.g., 'up' or 'down'
        self.feedback = ""

    def process_frame(self, frame, landmarks, mp_pose):
        """
        This method must be implemented by each specific exercise class.
        It contains the core logic for analyzing the pose for that exercise.

        Args:
            frame: The current video frame from OpenCV.
            landmarks: The pose landmarks detected by MediaPipe.
            mp_pose: The MediaPipe pose solution instance.

        Returns:
            The processed frame with feedback and annotations drawn on it.
        """
        raise NotImplementedError("This method should be overridden by a subclass")

    def display_feedback(self, frame):
        """
        A helper function to display the rep counter and feedback on the screen.
        """
        # Status box
        cv2.rectangle(frame, (0, 0), (250, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(frame, 'REPS', (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.rep_counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Feedback
        cv2.putText(frame, 'FEEDBACK', (90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, self.feedback, (90, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return frame








# class Exercise:
#     def __init__(self):
#         # Common attributes like rep counter, stage, etc.
#         self.rep_counter = 0
#         self.stage = None # e.g., 'up' or 'down'

#     def process_frame(self, frame, landmarks):
#         """
#         This method must be implemented by each specific exercise class.
#         It will contain the core logic for analyzing the pose.
#         It should return the processed frame and any feedback text.
#         """
#         raise NotImplementedError("This method should be overridden by subclass")