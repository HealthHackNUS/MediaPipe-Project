import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
def draw_angle_connections(image, landmarks, connections, left_color, right_color):
    for connection in connections:
        # Determine connection type and color
        if connection[0] in [11, 13]:  # Left arm landmarks (Mediapipe indices)
            color = left_color
        else:
            color = right_color

        # Draw connection line
        start_point = landmarks[connection[0].value]
        end_point = landmarks[connection[1].value]
        
        start_px = mp_drawing._normalized_to_pixel_coordinates(
            start_point.x, start_point.y, image.shape[1], image.shape[0])
        end_px = mp_drawing._normalized_to_pixel_coordinates(
            end_point.x, end_point.y, image.shape[1], image.shape[0])
        
        if start_px and end_px:
            cv2.line(image, start_px, end_px, color, 2)

        # Draw landmarks with matching color
        for landmark in [start_point, end_point]:
            point_px = mp_drawing._normalized_to_pixel_coordinates(
                landmark.x, landmark.y, image.shape[1], image.shape[0])
            if point_px:
                cv2.circle(image, point_px, 4, color, -1)
#start video capture
cap = cv2.VideoCapture(0)

counter_left = 0
counter_right = 0
flex_threshold = 70
flex_state_left = False
flex_state_right = False

ANGLE_CONNECTIONS = frozenset([
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
])

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Convert image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            # Calculate angle
            angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
            angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
            
            if angle_L < flex_threshold:
                color_L = (0, 255, 0)  # Green when flexed
                if not flex_state_left:
                    flex_state_left = True
            else:
                color_L = (0, 0, 255)  # Red when extended
                if flex_state_left:
                    counter_left += 1
                    flex_state_left = False

            if angle_R < flex_threshold:
                color_R = (0, 255, 0)  # Green when flexed
                if not flex_state_right:
                    flex_state_right = True
            else:
                color_R = (0, 0, 255)  # Red when extended
                if flex_state_right:
                    counter_right += 1
                    flex_state_right = False

            # Visualize angle
            # Add counter display with background box
            cv2.rectangle(frame, (10, 10), (250, 150), (0, 0, 0), -1)  # Black background
            cv2.putText(frame, f"Left Flexions: {counter_left}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_L, 1)
            cv2.putText(frame, f"Right Flexions: {counter_right}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_R, 1)
            cv2.putText(frame, f"Threshold: {flex_threshold})", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            bar_length = 150
            current_angle = min(angle_L, angle_R)
            flex_progress = max(0, min(1, (flex_threshold - current_angle)/flex_threshold))
            cv2.rectangle(frame, (20, 100), (20 + bar_length, 120), (255,255,255), 1)
            cv2.rectangle(frame, (20, 100), (20 + int(bar_length * flex_progress), 120), 
                        (0,255,0), -1)

            # Modified angle display with dynamic colors
            cv2.putText(frame, str(int(angle_L)), 
                    tuple(np.multiply(elbow_L, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_L, 2, cv2.LINE_AA)
            cv2.putText(frame, str(int(angle_R)), 
                    tuple(np.multiply(elbow_R, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_R, 2, cv2.LINE_AA)
                       
        except:
            pass
        
        
        # Render detections
        if results.pose_landmarks:
            draw_angle_connections(
                frame,
                results.pose_landmarks.landmark,
                ANGLE_CONNECTIONS,
                color_L, 
                color_R
            )               
        
        cv2.imshow('Mediapipe Flexion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
