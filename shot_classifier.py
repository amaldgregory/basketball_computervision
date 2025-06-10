from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class ShotClassification:
    """Container for shot classification results"""
    classification: str
    confidence: float
    score: float
    feedback: List[str]
    timestamp: float
    frame_number: int


class ShotClassifier:
    """Classifies basketball shots based on form analysis"""
    
    def __init__(self):
        # Classification thresholds
        self.classification_thresholds = {
            'excellent': 80,
            'good': 60,
            'fair': 40,
            'needs_work': 0
        }
        
        # Weight factors for different aspects of shooting form
        self.form_weights = {
            'shooting_elbow': 0.3,
            'shoulder_alignment': 0.25,
            'knee_bend': 0.2,
            'hip_shoulder': 0.15,
            'overall_balance': 0.1
        }
        
        # Optimal ranges for each form aspect
        self.optimal_ranges = {
            'shooting_elbow': (85, 95),
            'shoulder_alignment': (170, 180),
            'knee_bend': (100, 140),
            'hip_shoulder': (160, 180)
        }
    
    def classify_shot(self, angles: Dict, confidence_scores: Dict) -> ShotClassification:
        """
        Classify a shot based on angle analysis
        
        Args:
            angles: Dictionary of calculated angles
            confidence_scores: Dictionary of confidence scores for each angle
            
        Returns:
            ShotClassification object
        """
        # Calculate weighted score
        total_score = 0
        total_weight = 0
        feedback = []
        
        for angle_name, angle_value in angles.items():
            if angle_name in self.form_weights and angle_name in self.optimal_ranges:
                weight = self.form_weights[angle_name]
                min_angle, max_angle = self.optimal_ranges[angle_name]
                
                # Calculate how well this angle fits the optimal range
                if min_angle <= angle_value <= max_angle:
                    # Perfect range
                    score = 100
                    feedback.append(f"✓ Excellent {angle_name.replace('_', ' ')}: {angle_value:.1f}°")
                else:
                    # Calculate distance from optimal range
                    if angle_value < min_angle:
                        distance = min_angle - angle_value
                        max_distance = min_angle - 0  # Assuming 0 is minimum possible
                    else:
                        distance = angle_value - max_angle
                        max_distance = 180 - max_angle  # Assuming 180 is maximum possible
                    
                    # Score decreases with distance from optimal range
                    score = max(0, 100 - (distance / max_distance) * 100)
                    
                    if score >= 80:
                        feedback.append(f"✓ Good {angle_name.replace('_', ' ')}: {angle_value:.1f}°")
                    elif score >= 60:
                        feedback.append(f"⚠ Fair {angle_name.replace('_', ' ')}: {angle_value:.1f}°")
                    else:
                        feedback.append(f"✗ Poor {angle_name.replace('_', ' ')}: {angle_value:.1f}°")
                
                # Apply confidence weighting
                confidence = confidence_scores.get(angle_name, 0.5)
                weighted_score = score * confidence
                
                total_score += weighted_score * weight
                total_weight += weight
        
        # Calculate final score
        final_score = total_score / total_weight if total_weight > 0 else 0
        
        # Determine classification
        classification = self._get_classification(final_score)
        
        # Calculate overall confidence
        overall_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.5
        
        return ShotClassification(
            classification=classification,
            confidence=overall_confidence,
            score=final_score,
            feedback=feedback,
            timestamp=datetime.now().timestamp(),
            frame_number=0  # Will be set by caller
        )
    
    def _get_classification(self, score: float) -> str:
        """
        Get classification string based on score
        
        Args:
            score: Form score (0-100)
            
        Returns:
            Classification string
        """
        if score >= self.classification_thresholds['excellent']:
            return "Excellent"
        elif score >= self.classification_thresholds['good']:
            return "Good"
        elif score >= self.classification_thresholds['fair']:
            return "Fair"
        else:
            return "Needs Work"
    
    def analyze_shot_sequence(self, angle_data: List[Dict]) -> List[ShotClassification]:
        """
        Analyze a sequence of shots
        
        Args:
            angle_data: List of angle analysis results
            
        Returns:
            List of shot classifications
        """
        classifications = []
        
        for i, frame_data in enumerate(angle_data):
            if 'angles' in frame_data:
                angles = {name: data['angle'] for name, data in frame_data['angles'].items()}
                confidences = {name: data['confidence'] for name, data in frame_data['angles'].items()}
                
                classification = self.classify_shot(angles, confidences)
                classification.frame_number = frame_data.get('frame_number', i)
                classification.timestamp = frame_data.get('timestamp', 0)
                
                classifications.append(classification)
        
        return classifications
    
    def get_shot_statistics(self, classifications: List[ShotClassification]) -> Dict:
        """
        Generate statistics from shot classifications
        
        Args:
            classifications: List of shot classifications
            
        Returns:
            Statistics dictionary
        """
        if not classifications:
            return {}
        
        scores = [c.score for c in classifications]
        confidences = [c.confidence for c in classifications]
        
        # Count classifications
        class_counts = {}
        for classification in classifications:
            class_counts[classification.classification] = class_counts.get(classification.classification, 0) + 1
        
        # Calculate averages
        avg_score = np.mean(scores)
        avg_confidence = np.mean(confidences)
        
        # Find best and worst shots
        best_shot = max(classifications, key=lambda x: x.score)
        worst_shot = min(classifications, key=lambda x: x.score)
        
        return {
            'total_shots': len(classifications),
            'average_score': avg_score,
            'average_confidence': avg_confidence,
            'classification_distribution': class_counts,
            'best_shot': {
                'frame': best_shot.frame_number,
                'score': best_shot.score,
                'classification': best_shot.classification
            },
            'worst_shot': {
                'frame': worst_shot.frame_number,
                'score': worst_shot.score,
                'classification': worst_shot.classification
            },
            'score_std': np.std(scores),
            'consistency_score': 100 - np.std(scores)  # Higher consistency = lower std
        }
    
    def get_improvement_recommendations(self, classifications: List[ShotClassification]) -> List[str]:
        """
        Generate improvement recommendations based on analysis
        
        Args:
            classifications: List of shot classifications
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        if not classifications:
            return ["No shot data available for recommendations"]
        
        # Analyze common issues
        all_feedback = []
        for classification in classifications:
            all_feedback.extend(classification.feedback)
        
        # Count feedback types
        feedback_counts = {}
        for feedback in all_feedback:
            if feedback.startswith("✗"):
                feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
        
        # Generate recommendations based on most common issues
        if feedback_counts:
            most_common_issues = sorted(feedback_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for issue, count in most_common_issues:
                percentage = (count / len(classifications)) * 100
                if percentage > 50:  # Issue appears in more than 50% of shots
                    recommendations.append(f"Focus on improving: {issue.split(':')[0].replace('✗', '').strip()}")
        
        # Add general recommendations based on average score
        avg_score = np.mean([c.score for c in classifications])
        
        if avg_score < 40:
            recommendations.append("Consider working with a coach to improve fundamental shooting mechanics")
        elif avg_score < 60:
            recommendations.append("Focus on consistency and repetition to improve form")
        elif avg_score < 80:
            recommendations.append("Minor adjustments needed - you're close to excellent form!")
        else:
            recommendations.append("Excellent form! Focus on maintaining consistency")
        
        return recommendations


class ShotQualityAnalyzer:
    """Advanced shot quality analysis with temporal patterns"""
    
    def __init__(self):
        self.classifier = ShotClassifier()
    
    def analyze_shot_quality_trends(self, classifications: List[ShotClassification]) -> Dict:
        """
        Analyze trends in shot quality over time
        
        Args:
            classifications: List of shot classifications
            
        Returns:
            Trend analysis results
        """
        if len(classifications) < 2:
            return {"error": "Need at least 2 shots for trend analysis"}
        
        scores = [c.score for c in classifications]
        timestamps = [c.timestamp for c in classifications]
        
        # Calculate trend
        if len(scores) > 1:
            # Simple linear trend
            x = np.arange(len(scores))
            trend_coefficient = np.polyfit(x, scores, 1)[0]
            
            trend_direction = "improving" if trend_coefficient > 0 else "declining" if trend_coefficient < 0 else "stable"
            trend_strength = abs(trend_coefficient)
        else:
            trend_direction = "insufficient_data"
            trend_strength = 0
        
        # Analyze consistency
        score_variance = np.var(scores)
        consistency_level = "high" if score_variance < 100 else "medium" if score_variance < 400 else "low"
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'consistency_level': consistency_level,
            'score_variance': score_variance,
            'total_shots': len(classifications)
        }
    
    def identify_key_frames(self, classifications: List[ShotClassification]) -> Dict:
        """
        Identify key frames in the shot sequence
        
        Args:
            classifications: List of shot classifications
            
        Returns:
            Key frames analysis
        """
        if not classifications:
            return {}
        
        # Find transition points (significant changes in score)
        key_frames = []
        scores = [c.score for c in classifications]
        
        for i in range(1, len(scores)):
            score_change = abs(scores[i] - scores[i-1])
            if score_change > 20:  # Significant change threshold
                key_frames.append({
                    'frame': classifications[i].frame_number,
                    'score_change': score_change,
                    'previous_score': scores[i-1],
                    'current_score': scores[i],
                    'type': 'improvement' if scores[i] > scores[i-1] else 'decline'
                })
        
        return {
            'key_frames': key_frames,
            'total_transitions': len(key_frames)
        }


if __name__ == "__main__":
    # Test the shot classifier
    classifier = ShotClassifier()
    
    # Example angle data
    test_angles = {
        'shooting_elbow': 90,
        'shoulder_alignment': 175,
        'knee_bend': 120,
        'hip_shoulder': 170
    }
    
    test_confidences = {
        'shooting_elbow': 0.9,
        'shoulder_alignment': 0.8,
        'knee_bend': 0.7,
        'hip_shoulder': 0.6
    }
    
    classification = classifier.classify_shot(test_angles, test_confidences)
    
    print(f"Shot Classification: {classification.classification}")
    print(f"Score: {classification.score:.1f}")
    print(f"Confidence: {classification.confidence:.2f}")
    print("Feedback:")
    for feedback in classification.feedback:
        print(f"  {feedback}") 