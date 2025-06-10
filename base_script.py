import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize pose estimation
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        
        #adding left arm
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
        

        #adding right arm
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        # Calculate elbow angle
        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        
        # Display angle
        #cv2.putText(frame, f'Elbow Angle: {int(l_angle)}', 
                   #tuple(np.multiply(l_elbow, [1, 1]).astype(int)), 
                   #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, f'L Elbow: {int(l_angle)}0', 
           (int(l_elbow[0]) - 50, int(l_elbow[1]) - 20), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f'R Elbow: {int(r_angle)}o', 
           (int(r_elbow[0]) + 10, int(r_elbow[1]) - 20), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    

    cv2.imshow('Pose Detection with Angles', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
