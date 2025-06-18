import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import math
from ultralytics import YOLO

# Load custom-trained YOLOv8 model
model = YOLO("best.pt")

# MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Globals
total_shots = 0
successful_shots = 0
ball_pos = []
hoop_pos = []
up = False
down = False
up_frame = 0
down_frame = 0
fade_frames = 20
fade_counter = 0
overlay_text = ""
overlay_color = (0, 0, 0)
shot_ball_path = []  # stores trajectory during shot

# Additional globals for shot form analysis
form_scores = []
shot_data = []
ball_trajectory = []
previous_angles = {}
shot_success = None
last_shot_time = 0  # Track when last shot was detected
shot_cooldown = 3.0  # Increased cooldown for more accuracy
shot_state = "idle"  # States: idle, preparing, shooting, completed
shot_start_time = 0  # When the current shot started

# Global variables for persistent form score display
latest_form_score = None
latest_form_feedback = None
form_display_frames = 0
FORM_DISPLAY_DURATION = 60  # Show form score for 60 frames (about 2 seconds at 30fps)

def initialize_video(video_source):
    return cv2.VideoCapture(video_source) if isinstance(video_source, str) else cv2.VideoCapture(video_source)

def clean_positions(pos_list, max_age=1.0):
    current_time = time.time()
    return [p for p in pos_list if current_time - p[1] < max_age]

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

def detect_objects(frame):
    global ball_pos, hoop_pos

    results = model.predict(source=frame, conf=0.3, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        w, h = x2 - x1, y2 - y1
        center = (int(x1 + w / 2), int(y1 + h / 2))

        if cls == 0 and conf > 0.3:  # Ball
            ball_pos.append((center, time.time(), w, h, conf))
            cv2.circle(frame, center, 5, (0, 140, 255), 2)

        elif cls == 1 and conf > 0.5:  # Hoop
            hoop_pos.append((center, time.time(), w, h, conf))
            cv2.circle(frame, center, 12, (0, 255, 255), 3)
            cv2.putText(frame, "Hoop", (center[0] - 30, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

def detect_up(ball_pos, hoop_pos):
    if not hoop_pos or not ball_pos:
        return False
    return ball_pos[-1][0][1] < hoop_pos[-1][0][1] - 20

def detect_down(ball_pos, hoop_pos):
    if not hoop_pos or not ball_pos:
        return False
    return ball_pos[-1][0][1] > hoop_pos[-1][0][1] + 20

def score_shot(path, hoop_pos, frame):
    print("Entered function")
    if not hoop_pos or not path:
        print("getting stuck here")
        return False
    hoop_x, hoop_y = hoop_pos[-1][0]
    hoop_w, hoop_h = hoop_pos[-1][2], hoop_pos[-1][3]

    margin_x = int(hoop_w * 1.0)
    margin_y = int(hoop_h * 1.0)

    top_left = (int(hoop_x - margin_x), int(hoop_y - margin_y))
    bottom_right = (int(hoop_x + margin_x), int(hoop_y + margin_y))
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
    cv2.putText(frame, 'Scoring Zone', (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    for (x, y) in path:
        cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0), -1)  # blue dots for ball path
        print(f"Ball: ({x:.1f}, {y:.1f}) vs Hoop: ({hoop_x:.1f}, {hoop_y:.1f})")
        if abs(x - hoop_x) < margin_x and abs(y - hoop_y) < margin_y:
            print("shot detected")
            return True
    print("all the way through the function")
    return False

def process_shot_detection(frame, angles=None):
    global up, down, up_frame, down_frame, overlay_color, overlay_text, fade_counter
    global successful_shots, total_shots, shot_ball_path

    current_time = time.time()

    # Track when ball goes above the hoop
    if not up:
        up = detect_up(ball_pos, hoop_pos)
        if up:
            up_frame = current_time
            shot_ball_path.clear()

    if up and not down:
        down = detect_down(ball_pos, hoop_pos)
        if down:
            down_frame = current_time

    # If up and down happened, and in order
    if up and down and up_frame < down_frame:
        total_shots += 1
        up = down = False

        if score_shot(shot_ball_path, hoop_pos, frame):
            successful_shots += 1
            overlay_color = (0, 255, 0)
            overlay_text = "Make"
        else:
            overlay_color = (255, 0, 0)
            overlay_text = "Miss"

        # --- FORM FEEDBACK PRINT LOGIC ---
        if angles is not None:
            score, feedback = analyze_shot_form(angles)
            reason = feedback[0] if feedback else 'Good form'
            print(f"Form Score: {score}/100 - {reason}")
        else:
            print("Form Score: N/A - Could not detect player pose for this shot.")
        # --- END FORM FEEDBACK PRINT LOGIC ---

        fade_counter = fade_frames
        shot_ball_path.clear()

    if fade_counter > 0:
        alpha = 0.2 * (fade_counter / fade_frames)
        overlay = np.full_like(frame, overlay_color)
        frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        cv2.putText(frame, overlay_text, (frame.shape[1] - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
        fade_counter -= 1

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

def main():
    global ball_pos, hoop_pos, shot_ball_path, total_shots, ball_trajectory, previous_angles, last_shot_time, shot_state, shot_start_time

    parser = argparse.ArgumentParser(description='Basketball Shot Analyzer')
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    args = parser.parse_args()
    video_source = args.video if args.video else 0

    cap = initialize_video(video_source)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Initialize previous ball position
    prev_ball_center = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (720, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Initialize variables for pose detection
        ball_center = None
        ball_radius = None
        l_wrist = None
        r_wrist = None

        detect_objects(frame)
        ball_pos = clean_positions(ball_pos)
        hoop_pos = clean_positions(hoop_pos)

        if ball_pos:
            shot_ball_path.append(ball_pos[-1][0])  # Track center only
            ball_center = ball_pos[-1][0]  # Use detected ball position

        # Pose detection and angle calculations
        angles = None
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Left arm landmarks
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

            # Right arm landmarks
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
            
            # Right leg landmarks
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
            
            # Left leg landmarks
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

            # Calculate angles
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

            angles = (l_angle, r_angle, l_knee_angle, r_knee_angle)

            # Track ball trajectory for shot form analysis
            if ball_center:
                ball_trajectory.append((ball_center, time.time()))
                if len(ball_trajectory) > 30:  # Keep last 30 positions
                    ball_trajectory.pop(0)

            # Display angles on frame
            cv2.putText(frame, f'L Elbow: {int(l_angle)}°', 
               (int(l_elbow[0]) - 50, int(l_elbow[1]) - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(frame, f'R Elbow: {int(r_angle)}°', 
               (int(r_elbow[0]) + 10, int(r_elbow[1]) - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.putText(frame, f'R Knee: {int(r_knee_angle)}°',
                (int(r_knee[0]) - 30, int(r_knee[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.putText(frame, f'L Knee: {int(l_knee_angle)}°',
                (int(l_knee[0]) - 30, int(l_knee[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Shot detection and form analysis
            current_time = time.time()
            if ball_center and detect_shot_phase(ball_center, l_wrist, r_wrist, prev_ball_center, current_time):
                print("Shot taken! Evaluate form now")
                total_shots += 1
                
                # Enhanced form analysis
                score, feedback = analyze_shot_form(angles)
                
                # Analyze shot success
                success = analyze_shot_success()
                
                # Record shot data
                record_shot_data(angles, score, success)
                
                # Update previous angles for consistency
                previous_angles['elbow'] = r_angle
                
                # Print form score and explanation to terminal (single line)
                reason = feedback[0] if feedback else 'Good form'
                print(f"Form Score: {score}/100 - {reason}")
                
                # Display enhanced feedback (keep only verdict overlay, not form score/feedback)
                verdict_predictor(l_angle=l_angle, r_angle=r_angle, l_knee_angle=l_knee_angle, r_knee_angle=r_knee_angle, frame=frame)

            prev_ball_center = ball_center

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        process_shot_detection(frame, angles)

        # Display statistics
        cv2.putText(frame, f'Total Shots: {total_shots}', (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Successful: {successful_shots}', (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calculate and display FG%
        if total_shots > 0:
            fg_percentage = (successful_shots / total_shots) * 100
            cv2.putText(frame, f'FG%: {fg_percentage:.1f}%', (30, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display form statistics
        if form_scores:
            avg_score = sum(form_scores) / len(form_scores)
            cv2.putText(frame, f'Avg Form Score: {avg_score:.1f}', (30, 510), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Debug information
        if shot_state != "idle":
            cv2.putText(frame, f'Shot State: {shot_state}', (30, 540), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show hand distances when ball is detected
        if ball_center and results.pose_landmarks:
            l_hand_dist = np.linalg.norm(np.array(ball_center) - np.array(l_wrist))
            r_hand_dist = np.linalg.norm(np.array(ball_center) - np.array(r_wrist))
            min_hand_dist = min(l_hand_dist, r_hand_dist)
            
            cv2.putText(frame, f'L Hand: {int(l_hand_dist)}px', (30, 570), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f'R Hand: {int(r_hand_dist)}px', (30, 590), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f'Min Dist: {int(min_hand_dist)}px', (30, 610), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow('Basketball Shot Analyzer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n=== SESSION SUMMARY ===")
    print(f"Total Shots: {total_shots}")
    print(f"Successful Shots: {successful_shots}")
    if form_scores:
        print(f"Average Form Score: {sum(form_scores) / len(form_scores):.1f}/100")
        print(f"Best Form Score: {max(form_scores)}/100")

if __name__ == "__main__":
    main()
