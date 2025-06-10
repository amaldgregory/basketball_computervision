import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
import logging
from datetime import datetime


class FileUtils:
    """Utility functions for file handling"""
    
    @staticmethod
    def validate_video_file(file_path: str) -> bool:
        """
        Validate if a file is a supported video format
        
        Args:
            file_path: Path to the video file
            
        Returns:
            True if valid video file, False otherwise
        """
        if not os.path.exists(file_path):
            return False
        
        # Supported video formats
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in supported_formats:
            return False
        
        # Try to open with OpenCV to verify it's a valid video
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False
            
            # Check if we can read at least one frame
            ret, _ = cap.read()
            cap.release()
            return ret
        except Exception:
            return False
    
    @staticmethod
    def validate_image_file(file_path: str) -> bool:
        """
        Validate if a file is a supported image format
        
        Args:
            file_path: Path to the image file
            
        Returns:
            True if valid image file, False otherwise
        """
        if not os.path.exists(file_path):
            return False
        
        # Supported image formats
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in supported_formats:
            return False
        
        # Try to open with OpenCV to verify it's a valid image
        try:
            img = cv2.imread(file_path)
            return img is not None
        except Exception:
            return False
    
    @staticmethod
    def get_video_info(file_path: str) -> Dict:
        """
        Get video file information
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Dictionary containing video information
        """
        if not FileUtils.validate_video_file(file_path):
            raise ValueError(f"Invalid video file: {file_path}")
        
        cap = cv2.VideoCapture(file_path)
        
        info = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    @staticmethod
    def create_output_directory(base_dir: str, session_name: Optional[str] = None) -> str:
        """
        Create output directory for analysis results
        
        Args:
            base_dir: Base directory path
            session_name: Optional session name
            
        Returns:
            Path to created directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if session_name:
            dir_name = f"{session_name}_{timestamp}"
        else:
            dir_name = f"analysis_{timestamp}"
        
        output_dir = Path(base_dir) / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(output_dir)


class DataUtils:
    """Utility functions for data processing and validation"""
    
    @staticmethod
    def validate_pose_data(pose_data: Dict) -> bool:
        """
        Validate pose data structure
        
        Args:
            pose_data: Pose data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['keypoints', 'person_id']
        
        if not isinstance(pose_data, dict):
            return False
        
        if not all(key in pose_data for key in required_keys):
            return False
        
        if 'keypoints' not in pose_data or not isinstance(pose_data['keypoints'], dict):
            return False
        
        # Check if we have at least some keypoints
        if len(pose_data['keypoints']) < 5:
            return False
        
        return True
    
    @staticmethod
    def filter_low_confidence_keypoints(keypoints: Dict, threshold: float = 0.5) -> Dict:
        """
        Filter out low confidence keypoints
        
        Args:
            keypoints: Dictionary of keypoints
            threshold: Confidence threshold
            
        Returns:
            Filtered keypoints dictionary
        """
        filtered = {}
        
        for joint_name, joint_data in keypoints.items():
            if isinstance(joint_data, dict) and 'confidence' in joint_data:
                if joint_data['confidence'] >= threshold:
                    filtered[joint_name] = joint_data
        
        return filtered
    
    @staticmethod
    def normalize_coordinates(keypoints: Dict, image_width: int, image_height: int) -> Dict:
        """
        Normalize keypoint coordinates to 0-1 range
        
        Args:
            keypoints: Dictionary of keypoints
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Dictionary with normalized coordinates
        """
        normalized = {}
        
        for joint_name, joint_data in keypoints.items():
            if isinstance(joint_data, dict) and 'x' in joint_data and 'y' in joint_data:
                normalized[joint_name] = {
                    'x': joint_data['x'] / image_width,
                    'y': joint_data['y'] / image_height,
                    'confidence': joint_data.get('confidence', 0.0)
                }
        
        return normalized
    
    @staticmethod
    def denormalize_coordinates(keypoints: Dict, image_width: int, image_height: int) -> Dict:
        """
        Convert normalized coordinates back to pixel coordinates
        
        Args:
            keypoints: Dictionary of normalized keypoints
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Dictionary with pixel coordinates
        """
        denormalized = {}
        
        for joint_name, joint_data in keypoints.items():
            if isinstance(joint_data, dict) and 'x' in joint_data and 'y' in joint_data:
                denormalized[joint_name] = {
                    'x': joint_data['x'] * image_width,
                    'y': joint_data['y'] * image_height,
                    'confidence': joint_data.get('confidence', 0.0)
                }
        
        return denormalized


class ImageUtils:
    """Utility functions for image processing"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Resized image
        """
        target_width, target_height = target_size
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Place resized image on canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    
    @staticmethod
    def draw_pose_skeleton(image: np.ndarray, keypoints: Dict, 
                          confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Draw pose skeleton on image
        
        Args:
            image: Input image
            keypoints: Dictionary of keypoints
            confidence_threshold: Minimum confidence for drawing
            
        Returns:
            Image with skeleton drawn
        """
        # Define skeleton connections (COCO format)
        skeleton_connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle')
        ]
        
        # Draw connections
        for connection in skeleton_connections:
            joint1, joint2 = connection
            
            if joint1 in keypoints and joint2 in keypoints:
                kpt1 = keypoints[joint1]
                kpt2 = keypoints[joint2]
                
                if (kpt1['confidence'] >= confidence_threshold and 
                    kpt2['confidence'] >= confidence_threshold):
                    
                    pt1 = (int(kpt1['x']), int(kpt1['y']))
                    pt2 = (int(kpt2['x']), int(kpt2['y']))
                    
                    cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoints
        for joint_name, joint_data in keypoints.items():
            if joint_data['confidence'] >= confidence_threshold:
                x, y = int(joint_data['x']), int(joint_data['y'])
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(image, joint_name.replace('_', ' '), (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    @staticmethod
    def create_analysis_overlay(image: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """
        Create overlay with analysis results on image
        
        Args:
            image: Input image
            analysis_result: Analysis result dictionary
            
        Returns:
            Image with analysis overlay
        """
        overlay = image.copy()
        height, width = overlay.shape[:2]
        
        # Create semi-transparent overlay
        overlay_alpha = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add analysis information
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Form score
        score = analysis_result.get('form_score', 0)
        score_text = f"Form Score: {score:.1f}%"
        cv2.putText(overlay, score_text, (10, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += 30
        
        # Classification
        classification = analysis_result.get('classification', 'Unknown')
        classification_text = f"Classification: {classification}"
        
        # Color based on classification
        if classification == 'Excellent':
            color = (0, 255, 0)  # Green
        elif classification == 'Good':
            color = (0, 255, 255)  # Yellow
        elif classification == 'Fair':
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        cv2.putText(overlay, classification_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += 30
        
        # Key angles
        if 'angles' in analysis_result:
            for angle_name, angle_data in analysis_result['angles'].items():
                if angle_data['is_valid']:
                    angle_text = f"{angle_name.replace('_', ' ').title()}: {angle_data['angle']:.1f}°"
                    cv2.putText(overlay, angle_text, (10, y_offset), font, 0.5, (255, 255, 255), 1)
                    y_offset += 20
        
        # Blend overlay with original image
        alpha = 0.8
        result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        
        return result


class LoggingUtils:
    """Utility functions for logging"""
    
    @staticmethod
    def setup_logging(log_file: Optional[str] = None, level: str = 'INFO') -> logging.Logger:
        """
        Setup logging configuration
        
        Args:
            log_file: Optional log file path
            level: Logging level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger('basketball_analyzer')
        logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


class ValidationUtils:
    """Utility functions for data validation"""
    
    @staticmethod
    def validate_angle_range(angle: float, min_angle: float = 0, max_angle: float = 180) -> bool:
        """
        Validate if angle is within reasonable range
        
        Args:
            angle: Angle in degrees
            min_angle: Minimum valid angle
            max_angle: Maximum valid angle
            
        Returns:
            True if valid, False otherwise
        """
        return min_angle <= angle <= max_angle
    
    @staticmethod
    def validate_confidence_score(confidence: float) -> bool:
        """
        Validate confidence score
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            True if valid, False otherwise
        """
        return 0.0 <= confidence <= 1.0
    
    @staticmethod
    def validate_coordinates(x: float, y: float, max_x: float = 1920, max_y: float = 1080) -> bool:
        """
        Validate coordinate values
        
        Args:
            x: X coordinate
            y: Y coordinate
            max_x: Maximum X value
            max_y: Maximum Y value
            
        Returns:
            True if valid, False otherwise
        """
        return 0 <= x <= max_x and 0 <= y <= max_y


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test file validation
    print("File validation tests:")
    print(f"Video validation: {FileUtils.validate_video_file('nonexistent.mp4')}")
    print(f"Image validation: {FileUtils.validate_image_file('nonexistent.jpg')}")
    
    # Test data validation
    print("\nData validation tests:")
    print(f"Angle validation: {ValidationUtils.validate_angle_range(90)}")
    print(f"Confidence validation: {ValidationUtils.validate_confidence_score(0.8)}")
    print(f"Coordinate validation: {ValidationUtils.validate_coordinates(100, 200)}")
    
    # Test pose data validation
    test_pose_data = {
        'person_id': 0,
        'keypoints': {
            'right_shoulder': {'x': 100, 'y': 50, 'confidence': 0.9},
            'right_elbow': {'x': 100, 'y': 80, 'confidence': 0.8}
        }
    }
    
    print(f"Pose data validation: {DataUtils.validate_pose_data(test_pose_data)}")
    
    print("Utility functions test completed!") 