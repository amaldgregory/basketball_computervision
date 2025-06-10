import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
import os


class PoseDataExtractor:
    """Handles YOLOv8 pose estimation and data extraction from videos/images"""
    
    def __init__(self, model_size='n'):
        """
        Initialize the pose extractor with YOLOv8 model
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        """
        self.model = YOLO(f'yolov8{model_size}-pose.pt')
        
        # COCO pose keypoints (17 points)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Basketball-specific key joints
        self.basketball_joints = {
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12
        }
    
    def extract_pose_from_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Extract pose data from a single frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of pose data for each detected person
        """
        results = self.model(frame)
        pose_data = []
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()
                
                for person_idx, (kpts, conf) in enumerate(zip(keypoints, confidences)):
                    person_data = {
                        'person_id': person_idx,
                        'keypoints': {},
                        'bbox': result.boxes.xyxy[person_idx].cpu().numpy() if result.boxes else None
                    }
                    
                    for i, (point, confidence) in enumerate(zip(kpts, conf)):
                        person_data['keypoints'][self.keypoint_names[i]] = {
                            'x': float(point[0]),
                            'y': float(point[1]),
                            'confidence': float(confidence)
                        }
                    
                    pose_data.append(person_data)
        
        return pose_data
    
    def load_video_for_pose_analysis(self, video_path: str) -> Tuple[List[np.ndarray], int, Tuple[int, int]]:
        """
        Load video and extract frames for pose analysis
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames, fps, dimensions)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        print(f"Loading video: {frame_count} frames at {fps} FPS")
        
        with tqdm(total=frame_count, desc="Loading frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                pbar.update(1)
        
        cap.release()
        return frames, fps, (width, height)
    
    def process_video_for_pose_data(self, video_path: str, skip_frames: int = 1) -> Dict:
        """
        Process entire video and extract pose data from each frame
        
        Args:
            video_path: Path to video file
            skip_frames: Process every nth frame (1 = all frames)
            
        Returns:
            Dictionary containing pose data for all frames
        """
        frames, fps, dimensions = self.load_video_for_pose_analysis(video_path)
        
        video_pose_data = {
            'metadata': {
                'fps': fps,
                'total_frames': len(frames),
                'processed_frames': len(frames) // skip_frames,
                'dimensions': dimensions,
                'duration_seconds': len(frames) / fps,
                'skip_frames': skip_frames
            },
            'frame_data': []
        }
        
        print(f"Processing {len(frames) // skip_frames} frames for pose detection...")
        
        with tqdm(total=len(frames) // skip_frames, desc="Processing poses") as pbar:
            for frame_idx in range(0, len(frames), skip_frames):
                frame = frames[frame_idx]
                pose_data = self.extract_pose_from_frame(frame)
                
                frame_info = {
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / fps,
                    'poses': pose_data
                }
                
                video_pose_data['frame_data'].append(frame_info)
                pbar.update(1)
        
        return video_pose_data
    
    def save_pose_data(self, pose_data: Dict, output_path: str) -> None:
        """
        Save pose data in multiple formats
        
        Args:
            pose_data: Pose data dictionary
            output_path: Base path for output files
        """
        # Save as JSON for human readability
        json_path = f"{output_path}_poses.json"
        with open(json_path, 'w') as f:
            json.dump(pose_data, f, indent=2, default=str)
        print(f"Pose data saved to: {json_path}")
        
        # Save as NumPy array for fast numerical processing
        numpy_data = []
        for frame_data in pose_data['frame_data']:
            frame_array = []
            for pose in frame_data['poses']:
                person_keypoints = []
                for joint_name in self.keypoint_names:
                    if joint_name in pose['keypoints']:
                        kpt = pose['keypoints'][joint_name]
                        person_keypoints.extend([kpt['x'], kpt['y'], kpt['confidence']])
                    else:
                        person_keypoints.extend([0, 0, 0])  # Missing keypoint
                frame_array.append(person_keypoints)
            numpy_data.append(frame_array)
        
        np_path = f"{output_path}_poses.npy"
        np.save(np_path, numpy_data)
        print(f"NumPy data saved to: {np_path}")
    
    def get_basketball_keypoints(self, pose_data: Dict) -> Optional[Dict]:
        """
        Extract basketball-specific keypoints from pose data
        
        Args:
            pose_data: Pose data for a single person
            
        Returns:
            Dictionary of basketball keypoints or None if insufficient data
        """
        basketball_kpts = {}
        
        for joint_name, joint_idx in self.basketball_joints.items():
            if joint_name in pose_data['keypoints']:
                kpt = pose_data['keypoints'][joint_name]
                if kpt['confidence'] > 0.5:  # Confidence threshold
                    basketball_kpts[joint_name] = {
                        'x': kpt['x'],
                        'y': kpt['y'],
                        'confidence': kpt['confidence']
                    }
        
        # Check if we have enough keypoints for analysis
        required_joints = ['right_shoulder', 'right_elbow', 'right_wrist']
        if all(joint in basketball_kpts for joint in required_joints):
            return basketball_kpts
        
        return None


class ShotFormAnalyzer:
    """Main class for analyzing basketball shot form using pose estimation"""
    
    def __init__(self, model_size='n'):
        """
        Initialize the shot form analyzer
        
        Args:
            model_size: YOLOv8 model size
        """
        self.pose_extractor = PoseDataExtractor(model_size)
        self.analysis_results = []
    
    def analyze_video(self, video_path: str, output_dir: str = "output") -> Dict:
        """
        Analyze a basketball shooting video
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save results
            
        Returns:
            Analysis results dictionary
        """
        print(f"🏀 Analyzing basketball shot form in: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract pose data from video
        pose_data = self.pose_extractor.process_video_for_pose_data(video_path)
        
        # Save raw pose data
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, base_name)
        self.pose_extractor.save_pose_data(pose_data, output_path)
        
        # Analyze each frame for shot form
        shot_analysis = self._analyze_shot_sequence(pose_data)
        
        # Compile results
        results = {
            'video_path': video_path,
            'pose_data': pose_data,
            'shot_analysis': shot_analysis,
            'summary': self._generate_summary(shot_analysis)
        }
        
        return results
    
    def _analyze_shot_sequence(self, pose_data: Dict) -> List[Dict]:
        """
        Analyze shot form across all frames
        
        Args:
            pose_data: Pose data from video
            
        Returns:
            List of shot analysis results for each frame
        """
        shot_analysis = []
        
        for frame_data in pose_data['frame_data']:
            frame_analysis = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp'],
                'poses_analyzed': 0,
                'shot_forms': []
            }
            
            for pose in frame_data['poses']:
                basketball_kpts = self.pose_extractor.get_basketball_keypoints(pose)
                
                if basketball_kpts:
                    frame_analysis['poses_analyzed'] += 1
                    # Basic analysis - will be enhanced with angle calculations
                    shot_form = {
                        'person_id': pose['person_id'],
                        'keypoints': basketball_kpts,
                        'analysis_ready': True
                    }
                    frame_analysis['shot_forms'].append(shot_form)
            
            shot_analysis.append(frame_analysis)
        
        return shot_analysis
    
    def _generate_summary(self, shot_analysis: List[Dict]) -> Dict:
        """
        Generate summary statistics from shot analysis
        
        Args:
            shot_analysis: List of frame analysis results
            
        Returns:
            Summary statistics
        """
        total_frames = len(shot_analysis)
        frames_with_poses = sum(1 for frame in shot_analysis if frame['poses_analyzed'] > 0)
        
        return {
            'total_frames': total_frames,
            'frames_with_poses': frames_with_poses,
            'pose_detection_rate': frames_with_poses / total_frames if total_frames > 0 else 0,
            'analysis_complete': True
        }
    
    def get_analysis_results(self) -> List[Dict]:
        """Get all analysis results"""
        return self.analysis_results


if __name__ == "__main__":
    # Test the pose analyzer
    analyzer = ShotFormAnalyzer()
    
    # Example usage (uncomment when you have a video file)
    # results = analyzer.analyze_video("path/to/your/basketball_video.mp4")
    # print("Analysis complete!") 