import cv2
import mediapipe as mp
# import ssl

# Import the exercise modules
from exercises.simple_squat import SimpleSquat
from exercises.advanced_squat import AdvancedSquat
from exercises.pushup import Pushup

# ctx = ssl.create_default_context()
# ctx.check_hostname = False
# ctx.verify_mode = ssl.CERT_NONE

# url = 'http://192.168.1.5:8080/video'


def main():
    """
    Main function to run the pose estimation and exercise analysis application.
    """
    # --- SETUP ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # --- EXERCISE SELECTION ---
    # You can now choose "simple_squat", "advanced_squat", or "pushup"
    exercise_selection = "advanced_squat"

    if exercise_selection == "simple_squat":
        exercise_analyzer = SimpleSquat()
    elif exercise_selection == "advanced_squat":
        # Initialize the AdvancedSquat with custom parameters
        exercise_analyzer = AdvancedSquat(side="left", angle_up=160, angle_down=90)
    elif exercise_selection == "pushup":
        exercise_analyzer = Pushup()
    else:
        print(f"Error: Exercise '{exercise_selection}' not recognized.")
        return

    # cap = cv2.VideoCapture(url)  # Initialize remote webcam as webcam
    cap = cv2.VideoCapture(0)  # Initialize local webcam
    cap.set(3, 1920) # width
    cap.set(4, 1080)  # height

    print(f"Starting analysis for: {exercise_selection.upper()}")

    # --- MAIN VIDEO LOOP ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # --- ANALYSIS & FEEDBACK ---
        try:
            landmarks = results.pose_landmarks.landmark
            frame = exercise_analyzer.process_frame(frame, landmarks, mp_pose)
        except Exception:
            cv2.putText(frame, "No person detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- DRAWING LANDMARKS ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        cv2.imshow('Exercise Performance Analyzer', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

    # --- CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()