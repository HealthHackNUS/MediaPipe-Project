import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Define the pinch detection threshold (adjust as needed)
PINCH_THRESHOLD = 30  # Pixels


def calculate_distance(p1, p2):
    """Compute Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_angle(a, b, c):
    """Calculate angle at point B formed by A-B-C"""
    # a, b, c are tuples (x, y)
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    bc = c - b

    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


# Start video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        max_num_hands=2,  # Track up to 2 hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Get frame dimensions
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmark positions
                landmarks = hand_landmarks.landmark

                # Extract thumb tip and index finger tip
                thumb_tip = (int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * w),
                             int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * h))
                index_tip = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                             int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))

                # Draw landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate pinch distance
                distance = calculate_distance(thumb_tip, index_tip)

                # Display result
                if distance < PINCH_THRESHOLD:
                    cv2.putText(frame, "Pinching", (thumb_tip[0], thumb_tip[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    color = (0, 255, 0)  # Green for pinching
                else:
                    cv2.putText(frame, "Not Pinching", (thumb_tip[0], thumb_tip[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    color = (0, 0, 255)  # Red for not pinching

                # Draw a circle on the thumb and index finger
                cv2.circle(frame, thumb_tip, 8, color, -1)
                cv2.circle(frame, index_tip, 8, color, -1)

                # Get landmark positions in pixel coordinates
                thumb_mcp = (landmarks[2].x * w, landmarks[2].y * h)
                thumb_ip = (landmarks[3].x * w, landmarks[3].y * h)
                thumb_tip = (landmarks[4].x * w, landmarks[4].y * h)

                index_mcp = (landmarks[5].x * w, landmarks[5].y * h)
                index_pip = (landmarks[6].x * w, landmarks[6].y * h)
                index_dip = (landmarks[7].x * w, landmarks[7].y * h)
                index_tip = (landmarks[8].x * w, landmarks[8].y * h)

                #print(f"Thumb MCP: {thumb_mcp}, Thumb IP: {thumb_ip}, Thumb Tip: {thumb_tip}")
                #print(f"Index MCP: {index_mcp}, Index PIP: {index_pip}, Index DIP: {index_dip}")

                # MCP (Metacarpophalangeal joint) = knuckle joint
                # PIP (Proximal interphalangeal joint) = middle joint of finger
                # DIP (Distal interphalangeal joint) = tip joint of the finger

                # Thumb IP joint Angle ( between MCP, IP, TIP)
                thumb_ip_angle = calculate_angle(thumb_mcp, thumb_ip, thumb_tip)

                # Index Finger PIP Joint Angle (between MCP, PIP, DIP)
                index_pip_angle = calculate_angle(index_mcp, index_pip, index_dip)

                # Index Finger DIP Joint Angle (between PIP, DIP, TIP)
                index_dip_angle = calculate_angle(index_pip, index_dip, index_tip)

                # Display angles on frame
                #cv2.putText(frame, f"Thumb IP: {thumb_ip_angle:.1f}", (50, 50),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #cv2.putText(frame, f"Index PIP: {index_pip_angle:.1f}", (50, 70),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #cv2.putText(frame, f"Index DIP: {index_dip_angle:.1f}", (50, 90),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame
        cv2.imshow("Pinch Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
