import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def verdict_predictor(l_angle, r_angle, l_knee_angle, r_knee_angle):
    verdict = "Good Shot Form"
    explanation = []

    if not (70 <= r_angle <= 120 or 70 <= l_angle <= 120):
        verdict = "Poor Shot Form"
        explanation.append("Elbow angle not in shooting range (70–120°)")

    if l_knee_angle> 150 or r_knee_angle>150:
        verdict = "Poor Shot Form"
        explanation.append("Leg not bent enough (<150° knee angle)")

    # Display verdict and explanation
    cv2.putText(frame, verdict, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 0, 255) if verdict == "Poor Shot Form" else (0, 255, 0), 3)
    
    y_offset = 60
    for line in explanation:
        cv2.putText(frame, line, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20


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
        
        #adding right leg
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
        
        #adding left leg
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

        
        # Calculate elbow angle
        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        
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
        
        cv2.putText(frame, f'Knee: {int(r_knee_angle)}°',
            (int(r_knee[0]) - 30, int(r_knee[1]) - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.putText(frame, f'Knee: {int(l_knee_angle)}°',
            (int(l_knee[0]) - 30, int(l_knee[1]) - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        verdict_predictor(l_angle=l_angle, r_angle=r_angle, l_knee_angle=l_knee_angle, r_knee_angle=r_knee_angle)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    

    cv2.imshow('Basketball Shot Form Analyzer', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
