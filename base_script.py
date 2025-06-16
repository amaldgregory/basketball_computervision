import cv2
import mediapipe as mp
import numpy as np
import math
import time
import argparse

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables for statistics
total_shots = 0
successful_shots = 0
form_scores = []
shot_data = []
ball_trajectory = []
previous_angles = {}
shot_success = None
last_shot_time = 0  # Track when last shot was detected
shot_cooldown = 3.0  # Increased cooldown for more accuracy
shot_state = "idle"  # States: idle, preparing, shooting, completed
shot_start_time = 0  # When the current shot started

def initialize_video(video_source):
    """Initialize video capture from file or camera"""
    if isinstance(video_source, str):
        # Video file input
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_source}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video loaded: {total_frames} frames, {width}x{height}, {fps} FPS")
    else:
        # Camera input
        cap = cv2.VideoCapture(video_source)
    
    return cap

def analyze_shot_success():
    """Analyze if the shot was successful based on ball trajectory"""
    global ball_trajectory, shot_success, successful_shots
    
    if len(ball_trajectory) < 10:
        return None
    
    trajectory_points = [pos for pos, _ in ball_trajectory]
    
    # Analyze trajectory arc
    y_positions = [pos[1] for pos in trajectory_points]
    
    # Check for parabolic motion (ball goes up then down)
    if len(y_positions) > 5:
        first_half = y_positions[:len(y_positions)//2]
        second_half = y_positions[len(y_positions)//2:]
        
        # Ball should go up (y decreases) then down (y increases)
        going_up = np.mean(np.diff(first_half)) < 0
        coming_down = np.mean(np.diff(second_half)) > 0
        
        # Simple heuristic: if ball follows arc and ends in lower portion of screen
        if going_up and coming_down:
            final_y = y_positions[-1]
            frame_height = 480  # Assuming frame height
            
            # If ball ends in upper portion, likely a miss
            # If ball ends in middle-lower portion, likely a make
            if final_y > frame_height * 0.3:
                shot_success = True
                successful_shots += 1
            else:
                shot_success = False
    
    return shot_success

def analyze_shot_form(angles):
    """Enhanced shot form analysis with scoring"""
    global previous_angles
    
    l_angle, r_angle, l_knee_angle, r_knee_angle = angles
    
    score = 100
    feedback = []
    
    # Elbow angle analysis
    optimal_elbow_range = (70, 120)
    shooting_elbow = r_angle  # Assuming right-handed
    
    if not (optimal_elbow_range[0] <= shooting_elbow <= optimal_elbow_range[1]):
        score -= 25
        feedback.append(f"Elbow angle {shooting_elbow:.1f}° outside optimal range")
    
    # Knee bend analysis
    if l_knee_angle > 150 or r_knee_angle > 150:
        score -= 20
        feedback.append("Insufficient knee bend for power generation")
    
    # Consistency analysis
    if previous_angles:
        angle_consistency = abs(shooting_elbow - previous_angles.get('elbow', shooting_elbow))
        if angle_consistency > 15:
            score -= 10
            feedback.append("Inconsistent shooting form")
    
    return score, feedback

def record_shot_data(angles, score, success):
    """Record shot data for analysis"""
    global total_shots, shot_data, form_scores
    
    shot_record = {
        'timestamp': time.time(),
        'angles': {
            'left_elbow': angles[0],
            'right_elbow': angles[1],
            'left_knee': angles[2],
            'right_knee': angles[3]
        },
        'form_score': score,
        'successful': success,
        'shot_number': total_shots
    }
    shot_data.append(shot_record)
    form_scores.append(score)

#returns (x,y coordinates)
def detect_ball(frame, l_wrist=None, r_wrist=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #  HSV RANGE TUNING FOR ORANGE-BROWN BALL 
    lower_orange = np.array([5, 40, 40])
    upper_orange = np.array([25, 255, 255])
    
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    cv2.imshow('Ball Mask', mask)
    
    # Morphological ops to clean noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 100
    hand_distance_threshold = 150  # Only accept circles near a hand (in pixels)
    
    if contours and (l_wrist is not None and r_wrist is not None):
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        center = np.array([x, y])
                        # Distance to both hands
                        dist_l = np.linalg.norm(center - np.array(l_wrist))
                        dist_r = np.linalg.norm(center - np.array(r_wrist))
                        if min(dist_l, dist_r) < hand_distance_threshold:
                            valid_contours.append((contour, center, int(radius)))
        if valid_contours:
            # Pick the largest valid contour
            largest = max(valid_contours, key=lambda tup: cv2.contourArea(tup[0]))
            _, center, radius = largest
            return tuple(center.astype(int)), radius
    return None, None

#returns boolean value
def is_shot_taken(ball_center, wrist, prev_ball_center, threshold=80):  #increased threshold for more accurate detection
    if prev_ball_center is None:
        return False
    distance = np.linalg.norm(np.array(ball_center) - np.array(prev_ball_center))
    hand_dist = np.linalg.norm(np.array(ball_center) - np.array(wrist))
    
    # More stringent conditions for shot detection
    # Ball must move significantly AND be away from hands
    return (distance > threshold and hand_dist > 100)  # Increased hand distance threshold

def verdict_predictor(l_angle, r_angle, l_knee_angle, r_knee_angle, frame):
    verdict = "Good Shot Form"
    explanation = []

    if not (70 <= r_angle <= 120 or 70 <= l_angle <= 120):
        verdict = "Poor Shot Form"
        explanation.append("Elbow angle not in shooting range (70-120o)")

    if l_knee_angle> 150 or r_knee_angle>150:
        verdict = "Poor Shot Form"
        explanation.append("Leg not bent enough (<150o knee angle)")

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

def detect_shot_phase(ball_center, l_wrist, r_wrist, prev_ball_center, current_time):
    """Detect different phases of a shot and manage shot state"""
    global shot_state, shot_start_time, last_shot_time
    
    if ball_center is None or prev_ball_center is None:
        return False
    
    distance = np.linalg.norm(np.array(ball_center) - np.array(prev_ball_center))
    
    # Calculate distance to both hands
    l_hand_dist = np.linalg.norm(np.array(ball_center) - np.array(l_wrist))
    r_hand_dist = np.linalg.norm(np.array(ball_center) - np.array(r_wrist))
    
    # Use the minimum distance to either hand (ball should be away from both hands for a shot)
    min_hand_dist = min(l_hand_dist, r_hand_dist)
    
    # State machine for shot detection
    if shot_state == "idle":
        # Check if enough time has passed since last shot
        if current_time - last_shot_time < shot_cooldown:
            return False
        
        # Detect shot initiation (ball moving away from both hands significantly)
        if distance > 50 and min_hand_dist > 70:
            shot_state = "preparing"
            shot_start_time = current_time
            return False
    
    elif shot_state == "preparing":
        # Ball is being prepared for shot
        if distance > 80 and min_hand_dist > 100:
            shot_state = "shooting"
            return False
        elif current_time - shot_start_time > 2.0:  # Timeout for preparation
            shot_state = "idle"
            return False
    
    elif shot_state == "shooting":
        # Ball is in flight
        if current_time - shot_start_time > 1.0:  # Minimum shot duration
            # Reset state and count as shot
            shot_state = "idle"
            last_shot_time = current_time
            return True
        elif min_hand_dist < 50:  # Ball back near either hand, not a shot
            shot_state = "idle"
            return False
    
    return False

def main():
    global total_shots, ball_trajectory, previous_angles, last_shot_time, shot_state, shot_start_time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Basketball Shot Form Analyzer')
    parser.add_argument('--video', type=str, help='Path to video file (optional, defaults to webcam)')
    args = parser.parse_args()
    
    # Initialize video source
    video_source = args.video if args.video else 0
    cap = initialize_video(video_source)
    
    # Initialize pose estimation
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    #initialize previous ball position - starting at null
    prev_ball_center = None

    desired_width = 720  
    desired_height = 480

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #making the frame bigger
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Initialize variables
        ball_center = None
        ball_radius = None
        l_wrist = None
        r_wrist = None
        
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

            #calculating angles
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            
            ball_center, ball_radius = detect_ball(frame, l_wrist, r_wrist)

            if ball_center:
                cv2.circle(frame, ball_center, ball_radius, (0, 140, 255), 2)
                cv2.putText(frame, "Ball", (ball_center[0]-10, ball_center[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
                
                # Track ball trajectory
                ball_trajectory.append((ball_center, time.time()))
                if len(ball_trajectory) > 30:  # Keep last 30 positions
                    ball_trajectory.pop(0)

            
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
            
            #for right now will only work for right handed players, will need to decide which hand to consider in the future
            current_time = time.time()
            if ball_center and detect_shot_phase(ball_center, l_wrist, r_wrist, prev_ball_center, current_time):
                print("Shot taken! Evaluate form now")
                total_shots += 1
                
                # Enhanced form analysis
                angles = (l_angle, r_angle, l_knee_angle, r_knee_angle)
                score, feedback = analyze_shot_form(angles)
                
                # Analyze shot success
                success = analyze_shot_success()
                
                # Record shot data
                record_shot_data(angles, score, success)
                
                # Update previous angles for consistency
                previous_angles['elbow'] = r_angle
                
                # Display enhanced feedback
                verdict_predictor(l_angle=l_angle, r_angle=r_angle, l_knee_angle=l_knee_angle, r_knee_angle=r_knee_angle, frame=frame)
                
                # Display score and feedback
                cv2.putText(frame, f'Form Score: {score}/100', (30, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                y_offset = 150
                for line in feedback:
                    cv2.putText(frame, line, (30, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 20

            prev_ball_center = ball_center

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display statistics
        cv2.putText(frame, f'Total Shots: {total_shots}', (30, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Successful: {successful_shots}', (30, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if form_scores:
            avg_score = sum(form_scores) / len(form_scores)
            cv2.putText(frame, f'Avg Form Score: {avg_score:.1f}', (30, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Debug information
        if shot_state != "idle":
            cv2.putText(frame, f'Shot State: {shot_state}', (30, 290), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show hand distances when ball is detected
        if ball_center and results.pose_landmarks:
            l_hand_dist = np.linalg.norm(np.array(ball_center) - np.array(l_wrist))
            r_hand_dist = np.linalg.norm(np.array(ball_center) - np.array(r_wrist))
            min_hand_dist = min(l_hand_dist, r_hand_dist)
            
            cv2.putText(frame, f'L Hand: {int(l_hand_dist)}px', (30, 320), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f'R Hand: {int(r_hand_dist)}px', (30, 340), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f'Min Dist: {int(min_hand_dist)}px', (30, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow('Basketball Shot Form Analyzer', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print(f"\n=== SESSION SUMMARY ===")
    print(f"Total Shots: {total_shots}")
    print(f"Successful Shots: {successful_shots}")
    if form_scores:
        print(f"Average Form Score: {sum(form_scores) / len(form_scores):.1f}/100")
        print(f"Best Form Score: {max(form_scores)}/100")

if __name__ == "__main__":
    main()
