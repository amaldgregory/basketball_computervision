import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AngleResult:
    """Container for angle calculation results"""
    angle: float
    confidence: float
    is_valid: bool
    description: str


class JointAngleCalculator:
    """Calculates basketball-specific joint angles from pose keypoints"""
    
    def __init__(self):
        # Optimal shooting form ranges (in degrees)
        self.optimal_ranges = {
            'shooting_elbow': (85, 95),      # Shoulder-Elbow-Wrist
            'shoulder_alignment': (170, 180), # Hip-Shoulder-Elbow
            'wrist_snap': (45, 65),          # Elbow-Wrist-Finger
            'knee_bend': (100, 140),         # Hip-Knee-Ankle
            'hip_shoulder_angle': (160, 180) # Hip-Shoulder-Elbow
        }
        
        # Minimum confidence threshold for angle calculations
        self.confidence_threshold = 0.5
    
    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """
        Calculate angle at point2 formed by point1-point2-point3
        
        Args:
            point1: First point (x, y)
            point2: Middle point (x, y) - angle vertex
            point3: Third point (x, y)
            
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        p1 = np.array([point1[0], point1[1]])
        p2 = np.array([point2[0], point2[1]])
        p3 = np.array([point3[0], point3[1]])
        
        # Vector from point2 to point1
        v1 = p1 - p2
        # Vector from point2 to point3
        v2 = p3 - p2
        
        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        cos_angle = dot_product / (norm_v1 * norm_v2)
        # Clamp to valid range for arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = math.degrees(math.acos(cos_angle))
        return angle
    
    def get_joint_coordinates(self, keypoints: Dict, joint_name: str) -> Optional[Tuple[float, float]]:
        """
        Extract joint coordinates from keypoints dictionary
        
        Args:
            keypoints: Dictionary of joint keypoints
            joint_name: Name of the joint
            
        Returns:
            Tuple of (x, y) coordinates or None if not found
        """
        if joint_name in keypoints:
            joint = keypoints[joint_name]
            if joint['confidence'] >= self.confidence_threshold:
                return (joint['x'], joint['y'])
        return None
    
    def calculate_shooting_elbow_angle(self, keypoints: Dict) -> AngleResult:
        """
        Calculate shooting elbow angle (shoulder-elbow-wrist)
        
        Args:
            keypoints: Dictionary of joint keypoints
            
        Returns:
            AngleResult with angle and validation info
        """
        shoulder = self.get_joint_coordinates(keypoints, 'right_shoulder')
        elbow = self.get_joint_coordinates(keypoints, 'right_elbow')
        wrist = self.get_joint_coordinates(keypoints, 'right_wrist')
        
        if all([shoulder, elbow, wrist]):
            angle = self.calculate_angle(shoulder, elbow, wrist)
            confidence = min(
                keypoints['right_shoulder']['confidence'],
                keypoints['right_elbow']['confidence'],
                keypoints['right_wrist']['confidence']
            )
            
            is_valid = self.optimal_ranges['shooting_elbow'][0] <= angle <= self.optimal_ranges['shooting_elbow'][1]
            
            return AngleResult(
                angle=angle,
                confidence=confidence,
                is_valid=is_valid,
                description="Shooting elbow angle (shoulder-elbow-wrist)"
            )
        
        return AngleResult(0.0, 0.0, False, "Insufficient keypoints for shooting elbow angle")
    
    def calculate_shoulder_alignment_angle(self, keypoints: Dict) -> AngleResult:
        """
        Calculate shoulder alignment angle (hip-shoulder-elbow)
        
        Args:
            keypoints: Dictionary of joint keypoints
            
        Returns:
            AngleResult with angle and validation info
        """
        hip = self.get_joint_coordinates(keypoints, 'right_hip')
        shoulder = self.get_joint_coordinates(keypoints, 'right_shoulder')
        elbow = self.get_joint_coordinates(keypoints, 'right_elbow')
        
        if all([hip, shoulder, elbow]):
            angle = self.calculate_angle(hip, shoulder, elbow)
            confidence = min(
                keypoints['right_hip']['confidence'],
                keypoints['right_shoulder']['confidence'],
                keypoints['right_elbow']['confidence']
            )
            
            is_valid = self.optimal_ranges['shoulder_alignment'][0] <= angle <= self.optimal_ranges['shoulder_alignment'][1]
            
            return AngleResult(
                angle=angle,
                confidence=confidence,
                is_valid=is_valid,
                description="Shoulder alignment angle (hip-shoulder-elbow)"
            )
        
        return AngleResult(0.0, 0.0, False, "Insufficient keypoints for shoulder alignment angle")
    
    def calculate_knee_bend_angle(self, keypoints: Dict) -> AngleResult:
        """
        Calculate knee bend angle (hip-knee-ankle)
        
        Args:
            keypoints: Dictionary of joint keypoints
            
        Returns:
            AngleResult with angle and validation info
        """
        hip = self.get_joint_coordinates(keypoints, 'right_hip')
        knee = self.get_joint_coordinates(keypoints, 'right_knee')
        ankle = self.get_joint_coordinates(keypoints, 'right_ankle')
        
        if all([hip, knee, ankle]):
            angle = self.calculate_angle(hip, knee, ankle)
            confidence = min(
                keypoints['right_hip']['confidence'],
                keypoints['right_knee']['confidence'],
                keypoints['right_ankle']['confidence']
            )
            
            is_valid = self.optimal_ranges['knee_bend'][0] <= angle <= self.optimal_ranges['knee_bend'][1]
            
            return AngleResult(
                angle=angle,
                confidence=confidence,
                is_valid=is_valid,
                description="Knee bend angle (hip-knee-ankle)"
            )
        
        return AngleResult(0.0, 0.0, False, "Insufficient keypoints for knee bend angle")
    
    def calculate_hip_shoulder_angle(self, keypoints: Dict) -> AngleResult:
        """
        Calculate hip-shoulder angle for posture analysis
        
        Args:
            keypoints: Dictionary of joint keypoints
            
        Returns:
            AngleResult with angle and validation info
        """
        left_hip = self.get_joint_coordinates(keypoints, 'left_hip')
        right_hip = self.get_joint_coordinates(keypoints, 'right_hip')
        left_shoulder = self.get_joint_coordinates(keypoints, 'left_shoulder')
        right_shoulder = self.get_joint_coordinates(keypoints, 'right_shoulder')
        
        # Use midpoint of hips and shoulders
        if all([left_hip, right_hip, left_shoulder, right_shoulder]):
            hip_midpoint = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            shoulder_midpoint = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
            
            # Calculate angle using vertical reference
            vertical_point = (hip_midpoint[0], hip_midpoint[1] - 100)  # Point above hip
            angle = self.calculate_angle(vertical_point, hip_midpoint, shoulder_midpoint)
            
            confidence = min(
                keypoints['left_hip']['confidence'],
                keypoints['right_hip']['confidence'],
                keypoints['left_shoulder']['confidence'],
                keypoints['right_shoulder']['confidence']
            )
            
            is_valid = self.optimal_ranges['hip_shoulder_angle'][0] <= angle <= self.optimal_ranges['hip_shoulder_angle'][1]
            
            return AngleResult(
                angle=angle,
                confidence=confidence,
                is_valid=is_valid,
                description="Hip-shoulder alignment angle"
            )
        
        return AngleResult(0.0, 0.0, False, "Insufficient keypoints for hip-shoulder angle")
    
    def analyze_shooting_form(self, keypoints: Dict) -> Dict[str, AngleResult]:
        """
        Analyze all relevant shooting form angles
        
        Args:
            keypoints: Dictionary of joint keypoints
            
        Returns:
            Dictionary of angle analysis results
        """
        angles = {}
        
        # Calculate all relevant angles
        angles['shooting_elbow'] = self.calculate_shooting_elbow_angle(keypoints)
        angles['shoulder_alignment'] = self.calculate_shoulder_alignment_angle(keypoints)
        angles['knee_bend'] = self.calculate_knee_bend_angle(keypoints)
        angles['hip_shoulder'] = self.calculate_hip_shoulder_angle(keypoints)
        
        return angles
    
    def get_form_score(self, angles: Dict[str, AngleResult]) -> Tuple[float, List[str]]:
        """
        Calculate overall form score based on angle analysis
        
        Args:
            angles: Dictionary of angle results
            
        Returns:
            Tuple of (score, feedback_messages)
        """
        valid_angles = [angle for angle in angles.values() if angle.is_valid]
        total_angles = len(angles)
        
        if total_angles == 0:
            return 0.0, ["No valid angles found for analysis"]
        
        score = len(valid_angles) / total_angles * 100
        feedback = []
        
        for angle_name, angle_result in angles.items():
            if angle_result.is_valid:
                feedback.append(f"✓ Good {angle_name.replace('_', ' ')}: {angle_result.angle:.1f}°")
            else:
                if angle_result.angle > 0:  # Angle was calculated but outside optimal range
                    feedback.append(f"✗ Adjust {angle_name.replace('_', ' ')}: {angle_result.angle:.1f}°")
                else:
                    feedback.append(f"✗ Missing data for {angle_name.replace('_', ' ')}")
        
        return score, feedback
    
    def get_optimal_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get optimal angle ranges for reference"""
        return self.optimal_ranges.copy()


class ShotFormAnalyzer:
    """Enhanced shot form analyzer with angle calculations"""
    
    def __init__(self):
        self.angle_calculator = JointAngleCalculator()
    
    def analyze_frame(self, keypoints: Dict) -> Dict:
        """
        Analyze shooting form for a single frame
        
        Args:
            keypoints: Joint keypoints from pose detection
            
        Returns:
            Complete analysis results
        """
        # Calculate all angles
        angles = self.angle_calculator.analyze_shooting_form(keypoints)
        
        # Get form score and feedback
        score, feedback = self.angle_calculator.get_form_score(angles)
        
        # Classify shot form
        classification = self._classify_shot_form(score)
        
        return {
            'angles': {name: {
                'angle': result.angle,
                'confidence': result.confidence,
                'is_valid': result.is_valid,
                'description': result.description
            } for name, result in angles.items()},
            'form_score': score,
            'classification': classification,
            'feedback': feedback,
            'timestamp': None  # Will be set by caller
        }
    
    def _classify_shot_form(self, score: float) -> str:
        """
        Classify shot form based on score
        
        Args:
            score: Form score (0-100)
            
        Returns:
            Classification string
        """
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Needs Work"
    
    def analyze_shot_sequence(self, pose_data: List[Dict]) -> List[Dict]:
        """
        Analyze a sequence of frames for shot form
        
        Args:
            pose_data: List of pose data for each frame
            
        Returns:
            List of analysis results for each frame
        """
        analysis_results = []
        
        for frame_data in pose_data:
            if 'keypoints' in frame_data and frame_data['keypoints']:
                analysis = self.analyze_frame(frame_data['keypoints'])
                analysis['timestamp'] = frame_data.get('timestamp', 0)
                analysis['frame_number'] = frame_data.get('frame_number', 0)
                analysis_results.append(analysis)
        
        return analysis_results


if __name__ == "__main__":
    # Test the angle calculator
    calculator = JointAngleCalculator()
    
    # Example keypoints (mock data)
    test_keypoints = {
        'right_shoulder': {'x': 100, 'y': 50, 'confidence': 0.9},
        'right_elbow': {'x': 100, 'y': 80, 'confidence': 0.8},
        'right_wrist': {'x': 100, 'y': 110, 'confidence': 0.7},
        'right_hip': {'x': 100, 'y': 120, 'confidence': 0.9}
    }
    
    angles = calculator.analyze_shooting_form(test_keypoints)
    score, feedback = calculator.get_form_score(angles)
    
    print("Angle Analysis Results:")
    for name, result in angles.items():
        print(f"{name}: {result.angle:.1f}° (Valid: {result.is_valid})")
    
    print(f"\nForm Score: {score:.1f}%")
    print("Feedback:")
    for msg in feedback:
        print(f"  {msg}") 