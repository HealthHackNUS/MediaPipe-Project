import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Trunk-Tibia angle thresholds
HIP_BIAS_THRESHOLD = 10  # Trunk-Tibia angle > 10° (Hip Bias)
KNEE_BIAS_THRESHOLD = -10  # Trunk-Tibia angle < -10° (Knee Bias)

def calculate_angle(p1, p2, p3):
    """Compute the angle between three points using dot product (returns acute angle)."""
    # Create vectors
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # Calculate the cosine of the angle using the dot product formula
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calculate cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Clip the value to avoid precision issues that can lead to cos_angle > 1 or < -1
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Return the acute angle in degrees
    angle = np.degrees(np.arccos(cos_angle))

    return angle

def calculate_normal_angle(p1, p2):
    """Calculate the angle of a line with respect to the horizontal (normal) in an acute form."""
    v = np.array(p1) - np.array(p2)
    angle = np.arctan2(v[1], v[0])  # Calculate angle with horizontal axis (y/x)
    angle = np.degrees(angle)

    # Make sure the angle is always positive and within 0 to 180 degrees
    if angle < 0:
        angle += 180

    return angle

# Start video capture
cap = cv2.VideoCapture(0)

# Flags for squat detection and counting
squat_count = 0
is_squatting = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        h, w, _ = frame.shape  # Get frame dimensions

        trunk_tibia_angle = None  # Initialize the variable for angle
        shoulder_hip_angle = None
        knee_ankle_angle = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract key points (LEFT side landmarks)
            shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h])
            hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h])
            knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h])
            ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h])

            # Compute shoulder-hip angle (trunk angle)
            shoulder_hip_angle = calculate_normal_angle(shoulder, hip)

            # Compute knee-ankle angle (tibia angle)
            knee_ankle_angle = calculate_normal_angle(knee, ankle)

            # Trunk-Tibia Angle (difference between trunk angle and tibia angle)
            trunk_tibia_angle = shoulder_hip_angle - knee_ankle_angle

            # Determine squat type based on angle
            if trunk_tibia_angle > HIP_BIAS_THRESHOLD:
                squat_type = "Hip Bias Squat"
                color = (0, 255, 0)  # Green
            elif trunk_tibia_angle < KNEE_BIAS_THRESHOLD:
                squat_type = "Knee Bias Squat"
                color = (255, 0, 0)  # Blue
            else:
                squat_type = "Neutral Squat"
                color = (0, 0, 255)  # Red

            # Display feedback
            cv2.putText(frame, f"{squat_type}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Display shoulder-hip angle and knee-ankle angle
            if shoulder_hip_angle is not None:
                cv2.putText(frame, f"Shoulder-Hip Angle: {shoulder_hip_angle:.2f}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if knee_ankle_angle is not None:
                cv2.putText(frame, f"Knee-Ankle Angle: {knee_ankle_angle:.2f}", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Count squats
            knee_angle = calculate_angle(hip, knee, ankle)

            if knee_angle > 160 and not is_squatting:  # Assuming 160 degrees as "standing" position
                squat_count += 1
                is_squatting = True
            elif knee_angle < 90:  # Squatting position
                is_squatting = False

            # Display squat count
            cv2.putText(frame, f"Squats: {squat_count}", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # If trunk_tibia_angle exists, display it
            if trunk_tibia_angle is not None:
                cv2.putText(frame, f"Trunk-Tibia Angle: {trunk_tibia_angle:.2f}", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow("Squat Analysis", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()