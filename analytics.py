import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from datetime import datetime
import json


class ShotAnalytics:
    """Handles shot statistics and analytics"""
    
    def __init__(self):
        self.shot_data = []
        self.session_data = []
        
        # Set style for better looking charts
        plt.style.use('default')
        sns.set_palette("husl")
    
    def record_shot(self, shot_data: Dict) -> None:
        """
        Record a shot for analytics
        
        Args:
            shot_data: Dictionary containing shot information
        """
        self.shot_data.append({
            'timestamp': datetime.now().isoformat(),
            **shot_data
        })
    
    def record_session(self, session_data: Dict) -> None:
        """
        Record a complete session
        
        Args:
            session_data: Dictionary containing session information
        """
        self.session_data.append({
            'timestamp': datetime.now().isoformat(),
            **session_data
        })
    
    def create_hexagon_chart(self, shot_data: List[Dict], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a hexagon performance chart
        
        Args:
            shot_data: List of shot analysis data
            save_path: Optional path to save the chart
            
        Returns:
            Matplotlib figure object
        """
        if not shot_data:
            raise ValueError("No shot data provided for hexagon chart")
        
        # Extract metrics for hexagon chart
        metrics = ['shooting_elbow', 'shoulder_alignment', 'knee_bend', 'hip_shoulder']
        
        # Calculate average scores for each metric
        metric_scores = {}
        for metric in metrics:
            scores = []
            for shot in shot_data:
                if 'angles' in shot and metric in shot['angles']:
                    angle_data = shot['angles'][metric]
                    if angle_data['is_valid']:
                        scores.append(100)  # Perfect score for valid angles
                    else:
                        # Calculate score based on distance from optimal range
                        angle = angle_data['angle']
                        if metric == 'shooting_elbow':
                            optimal_range = (85, 95)
                        elif metric == 'shoulder_alignment':
                            optimal_range = (170, 180)
                        elif metric == 'knee_bend':
                            optimal_range = (100, 140)
                        elif metric == 'hip_shoulder':
                            optimal_range = (160, 180)
                        else:
                            optimal_range = (0, 180)
                        
                        min_angle, max_angle = optimal_range
                        if min_angle <= angle <= max_angle:
                            scores.append(100)
                        else:
                            # Calculate distance from optimal range
                            if angle < min_angle:
                                distance = min_angle - angle
                            else:
                                distance = angle - max_angle
                            
                            # Score decreases with distance
                            score = max(0, 100 - distance * 2)
                            scores.append(score)
            
            metric_scores[metric] = np.mean(scores) if scores else 0
        
        # Create hexagon chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Prepare data for hexagon
        angles_hex = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        values = [metric_scores[metric] for metric in metrics]
        
        # Complete the hexagon
        values += values[:1]
        angles_hex = np.concatenate((angles_hex, [angles_hex[0]]))
        
        # Plot hexagon
        ax.plot(angles_hex, values, 'o-', linewidth=3, markersize=8, color='#2E86AB')
        ax.fill(angles_hex, values, alpha=0.25, color='#2E86AB')
        
        # Set up the plot
        ax.set_xticks(angles_hex[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics], fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add title and styling
        plt.title('Shot Form Analysis - Hexagon Chart', fontsize=16, fontweight='bold', pad=20)
        
        # Add performance indicators
        avg_score = np.mean(values[:-1])
        if avg_score >= 80:
            performance = "Excellent"
            color = "green"
        elif avg_score >= 60:
            performance = "Good"
            color = "orange"
        else:
            performance = "Needs Improvement"
            color = "red"
        
        plt.figtext(0.5, 0.02, f'Average Performance: {performance} ({avg_score:.1f}%)', 
                   ha='center', fontsize=14, fontweight='bold', color=color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hexagon chart saved to: {save_path}")
        
        return fig
    
    def create_performance_timeline(self, shot_data: List[Dict], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a timeline chart showing performance over time
        
        Args:
            shot_data: List of shot analysis data
            save_path: Optional path to save the chart
            
        Returns:
            Matplotlib figure object
        """
        if not shot_data:
            raise ValueError("No shot data provided for timeline chart")
        
        # Extract scores and timestamps
        scores = []
        timestamps = []
        classifications = []
        
        for shot in shot_data:
            if 'form_score' in shot:
                scores.append(shot['form_score'])
                timestamps.append(shot.get('timestamp', 0))
                classifications.append(shot.get('classification', 'Unknown'))
        
        if not scores:
            raise ValueError("No valid scores found in shot data")
        
        # Create timeline chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Score over time
        x = range(len(scores))
        ax1.plot(x, scores, 'o-', linewidth=2, markersize=6, color='#2E86AB')
        ax1.set_ylabel('Form Score', fontsize=12)
        ax1.set_title('Shot Form Performance Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Add trend line
        if len(scores) > 1:
            z = np.polyfit(x, scores, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x), "--", alpha=0.8, color='red', label='Trend')
            ax1.legend()
        
        # Plot 2: Classification distribution
        class_counts = {}
        for classification in classifications:
            class_counts[classification] = class_counts.get(classification, 0) + 1
        
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            colors = ['green', 'orange', 'yellow', 'red']
            
            ax2.bar(classes, counts, color=colors[:len(classes)], alpha=0.7)
            ax2.set_ylabel('Number of Shots', fontsize=12)
            ax2.set_title('Shot Classification Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timeline chart saved to: {save_path}")
        
        return fig
    
    def create_angle_distribution_chart(self, shot_data: List[Dict], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create distribution charts for individual angles
        
        Args:
            shot_data: List of shot analysis data
            save_path: Optional path to save the chart
            
        Returns:
            Matplotlib figure object
        """
        if not shot_data:
            raise ValueError("No shot data provided for angle distribution chart")
        
        # Extract angle data
        angle_data = {}
        metrics = ['shooting_elbow', 'shoulder_alignment', 'knee_bend', 'hip_shoulder']
        
        for metric in metrics:
            angles = []
            for shot in shot_data:
                if 'angles' in shot and metric in shot['angles']:
                    angle = shot['angles'][metric]['angle']
                    if angle > 0:  # Valid angle
                        angles.append(angle)
            angle_data[metric] = angles
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                angles = angle_data[metric]
                if angles:
                    # Create histogram
                    axes[i].hist(angles, bins=15, alpha=0.7, color='#2E86AB', edgecolor='black')
                    
                    # Add optimal range lines
                    if metric == 'shooting_elbow':
                        optimal_range = (85, 95)
                    elif metric == 'shoulder_alignment':
                        optimal_range = (170, 180)
                    elif metric == 'knee_bend':
                        optimal_range = (100, 140)
                    elif metric == 'hip_shoulder':
                        optimal_range = (160, 180)
                    else:
                        optimal_range = (0, 180)
                    
                    axes[i].axvline(optimal_range[0], color='red', linestyle='--', alpha=0.8, label='Optimal Range')
                    axes[i].axvline(optimal_range[1], color='red', linestyle='--', alpha=0.8)
                    
                    # Add mean line
                    mean_angle = np.mean(angles)
                    axes[i].axvline(mean_angle, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_angle:.1f}°')
                    
                    axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution', fontweight='bold')
                    axes[i].set_xlabel('Angle (degrees)')
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Angle distribution chart saved to: {save_path}")
        
        return fig
    
    def generate_summary_statistics(self, shot_data: List[Dict]) -> Dict:
        """
        Generate comprehensive summary statistics
        
        Args:
            shot_data: List of shot analysis data
            
        Returns:
            Dictionary containing summary statistics
        """
        if not shot_data:
            return {"error": "No shot data available"}
        
        # Extract scores and classifications
        scores = [shot.get('form_score', 0) for shot in shot_data]
        classifications = [shot.get('classification', 'Unknown') for shot in shot_data]
        
        # Calculate basic statistics
        total_shots = len(shot_data)
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Classification distribution
        class_distribution = {}
        for classification in classifications:
            class_distribution[classification] = class_distribution.get(classification, 0) + 1
        
        # Calculate percentages
        class_percentages = {}
        for classification, count in class_distribution.items():
            class_percentages[classification] = (count / total_shots) * 100
        
        # Find best and worst shots
        best_shot_idx = np.argmax(scores)
        worst_shot_idx = np.argmin(scores)
        
        # Analyze angle consistency
        angle_consistency = {}
        metrics = ['shooting_elbow', 'shoulder_alignment', 'knee_bend', 'hip_shoulder']
        
        for metric in metrics:
            angles = []
            for shot in shot_data:
                if 'angles' in shot and metric in shot['angles']:
                    angle = shot['angles'][metric]['angle']
                    if angle > 0:
                        angles.append(angle)
            
            if angles:
                angle_consistency[metric] = {
                    'mean': np.mean(angles),
                    'std': np.std(angles),
                    'consistency_score': max(0, 100 - np.std(angles))
                }
        
        return {
            'total_shots': total_shots,
            'average_score': avg_score,
            'score_std': std_score,
            'score_range': (min(scores), max(scores)),
            'classification_distribution': class_distribution,
            'classification_percentages': class_percentages,
            'best_shot': {
                'index': best_shot_idx,
                'score': scores[best_shot_idx],
                'classification': classifications[best_shot_idx]
            },
            'worst_shot': {
                'index': worst_shot_idx,
                'score': scores[worst_shot_idx],
                'classification': classifications[worst_shot_idx]
            },
            'angle_consistency': angle_consistency,
            'overall_consistency': max(0, 100 - std_score),
            'performance_level': self._get_performance_level(avg_score)
        }
    
    def _get_performance_level(self, avg_score: float) -> str:
        """Get performance level based on average score"""
        if avg_score >= 80:
            return "Elite"
        elif avg_score >= 70:
            return "Advanced"
        elif avg_score >= 60:
            return "Intermediate"
        elif avg_score >= 50:
            return "Beginner"
        else:
            return "Novice"
    
    def export_analytics_data(self, shot_data: List[Dict], filepath: str) -> None:
        """
        Export analytics data to JSON file
        
        Args:
            shot_data: List of shot analysis data
            filepath: Path to save the JSON file
        """
        analytics_data = {
            'export_timestamp': datetime.now().isoformat(),
            'summary_statistics': self.generate_summary_statistics(shot_data),
            'shot_data': shot_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(analytics_data, f, indent=2, default=str)
        
        print(f"Analytics data exported to: {filepath}")


if __name__ == "__main__":
    # Test the analytics module
    analytics = ShotAnalytics()
    
    # Example shot data
    test_shot_data = [
        {
            'form_score': 85,
            'classification': 'Excellent',
            'angles': {
                'shooting_elbow': {'angle': 90, 'is_valid': True},
                'shoulder_alignment': {'angle': 175, 'is_valid': True},
                'knee_bend': {'angle': 120, 'is_valid': True},
                'hip_shoulder': {'angle': 170, 'is_valid': True}
            }
        },
        {
            'form_score': 75,
            'classification': 'Good',
            'angles': {
                'shooting_elbow': {'angle': 88, 'is_valid': True},
                'shoulder_alignment': {'angle': 172, 'is_valid': True},
                'knee_bend': {'angle': 125, 'is_valid': True},
                'hip_shoulder': {'angle': 165, 'is_valid': False}
            }
        }
    ]
    
    # Generate summary statistics
    summary = analytics.generate_summary_statistics(test_shot_data)
    print("Summary Statistics:")
    print(json.dumps(summary, indent=2))
    
    # Create hexagon chart
    fig = analytics.create_hexagon_chart(test_shot_data)
    plt.show() 