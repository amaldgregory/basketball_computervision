'''
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import argparse

from ultralytics import YOLO

# Load custom-trained YOLOv8 model
model = YOLO("best.pt")

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
last_shot_time = 0
shot_cooldown = 3.0
shot_state = "idle"
shot_start_time = 0

# New shot detection variables
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

def initialize_video(video_source):
    cap = cv2.VideoCapture(video_source) if isinstance(video_source, str) else cv2.VideoCapture(video_source)
    return cap

def clean_positions(pos_list, max_age=1.0):
    current_time = time.time()
    return [p for p in pos_list if current_time - p[1] < max_age]

def detect_objects(frame, l_wrist=None, r_wrist=None):
    global ball_pos, hoop_pos

    results = model.predict(source=frame, conf=0.3, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        w, h = x2 - x1, y2 - y1
        center = (int(x1 + w / 2), int(y1 + h / 2))

        if cls == 0 and conf > 0.3:
            ball_pos.append((center, time.time(), w, h, conf))
            cv2.circle(frame, center, 5, (0, 140, 255), 2)

        elif cls == 1 and conf > 0.5:
            hoop_pos.append((center, time.time(), w, h, conf))
            cv2.circle(frame, center, 12, (0, 255, 255), 3)
            cv2.putText(frame, "Hoop", (center[0] - 30, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

def detect_up(ball_pos, hoop_pos):
    if not hoop_pos or not ball_pos:
        return False
    hoop_y = hoop_pos[-1][0][1]
    return ball_pos[-1][0][1] < hoop_y - 20

def detect_down(ball_pos, hoop_pos):
    if not hoop_pos or not ball_pos:
        return False
    hoop_y = hoop_pos[-1][0][1]
    return ball_pos[-1][0][1] > hoop_y + 20


def score_shot(ball_pos, hoop_pos):
    if not hoop_pos or not ball_pos:
        return False
    hoop_x, hoop_y = hoop_pos[-1][0]
    last_x, last_y = ball_pos[-1][0]
    return abs(last_x - hoop_x) < 40 and abs(last_y - hoop_y) < 40

#better one
def score_shot(path, hoop_pos):
    if not hoop_pos or not path:
        return False

    hoop_x, hoop_y = hoop_pos[-1][0]
    hoop_w, hoop_h = hoop_pos[-1][2], hoop_pos[-1][3]

    for (x, y) in path:
        if (hoop_x - hoop_w // 2) < x < (hoop_x + hoop_w // 2) and \
           (hoop_y - hoop_h // 2) < y < (hoop_y + hoop_h // 2):
            return True
    return False



def process_shot_detection(frame):
    global up, down, up_frame, down_frame, overlay_color, overlay_text, fade_counter, successful_shots, total_shots

    current_time = time.time()

    if not up:
        up = detect_up(ball_pos, hoop_pos)
        if up:
            up_frame = current_time

    if up and not down:
        down = detect_down(ball_pos, hoop_pos)
        if down:
            down_frame = current_time

    #if up and down and (0 < down_frame - up_frame < 2.0):
    if up and down and up_frame<down_frame:
        total_shots += 1
        up = down = False

        if score_shot(ball_pos, hoop_pos):
            successful_shots += 1
            overlay_color = (0, 255, 0)
            overlay_text = "Make"
        else:
            overlay_color = (255, 0, 0)
            overlay_text = "Miss"
        fade_counter = fade_frames

    if fade_counter > 0:
        alpha = 0.2 * (fade_counter / fade_frames)
        overlay = np.full_like(frame, overlay_color)
        frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        cv2.putText(frame, overlay_text, (frame.shape[1] - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
        fade_counter -= 1

def main():
    global ball_pos, hoop_pos

    parser = argparse.ArgumentParser(description='Basketball Shot Analyzer')
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    args = parser.parse_args()
    video_source = args.video if args.video else 0

    cap = initialize_video(video_source)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (720, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        l_wrist = r_wrist = None
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

        detect_objects(frame, l_wrist, r_wrist)
        ball_pos = clean_positions(ball_pos)
        hoop_pos = clean_positions(hoop_pos)

        process_shot_detection(frame)
        cv2.putText(frame, f'Total Shots: {total_shots}', (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Successful: {successful_shots}', (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Basketball Shot Analyzer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n=== SESSION SUMMARY ===")
    print(f"Total Shots: {total_shots}")
    print(f"Successful Shots: {successful_shots}")

if __name__ == "__main__":
    main()

'''

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
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

def initialize_video(video_source):
    return cv2.VideoCapture(video_source) if isinstance(video_source, str) else cv2.VideoCapture(video_source)

def clean_positions(pos_list, max_age=1.0):
    current_time = time.time()
    return [p for p in pos_list if current_time - p[1] < max_age]

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

    margin_x = int(hoop_w * 1.5)
    margin_y = int(hoop_h * 1.5)

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

def process_shot_detection(frame):
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

        fade_counter = fade_frames
        shot_ball_path.clear()

    if fade_counter > 0:
        alpha = 0.2 * (fade_counter / fade_frames)
        overlay = np.full_like(frame, overlay_color)
        frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        cv2.putText(frame, overlay_text, (frame.shape[1] - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
        fade_counter -= 1

def main():
    global ball_pos, hoop_pos, shot_ball_path

    parser = argparse.ArgumentParser(description='Basketball Shot Analyzer')
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    args = parser.parse_args()
    video_source = args.video if args.video else 0

    cap = initialize_video(video_source)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (720, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        detect_objects(frame)
        ball_pos = clean_positions(ball_pos)
        hoop_pos = clean_positions(hoop_pos)

        if ball_pos:
            shot_ball_path.append(ball_pos[-1][0])  # Track center only

        process_shot_detection(frame)

        cv2.putText(frame, f'Total Shots: {total_shots}', (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Successful: {successful_shots}', (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Basketball Shot Analyzer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n=== SESSION SUMMARY ===")
    print(f"Total Shots: {total_shots}")
    print(f"Successful Shots: {successful_shots}")

if __name__ == "__main__":
    main()
