#!/usr/bin/env python3
"""
Basketball Shot Form Analyzer - Main Application
A comprehensive tool for analyzing basketball shooting form using YOLOv8 pose estimation.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
import argparse

# Import our modules
from pose_analyzer import ShotFormAnalyzer as PoseAnalyzer
from angle_calculator import ShotFormAnalyzer as AngleAnalyzer
from shot_classifier import ShotClassifier, ShotQualityAnalyzer
from analytics import ShotAnalytics
from heatmap import ShotHeatMap
from exporter import ResultsExporter
from utils import FileUtils, LoggingUtils


class BasketballShotAnalyzer:
    """Main application class for basketball shot form analysis"""
    
    def __init__(self):
        """Initialize the basketball shot analyzer"""
        self.pose_analyzer = PoseAnalyzer()
        self.angle_analyzer = AngleAnalyzer()
        self.shot_classifier = ShotClassifier()
        self.quality_analyzer = ShotQualityAnalyzer()
        self.analytics = ShotAnalytics()
        self.heatmap = ShotHeatMap()
        self.exporter = ResultsExporter()
        
        # Setup logging
        self.logger = LoggingUtils.setup_logging()
        
        # Current session data
        self.current_session_data = []
        self.current_video_path = None
        
        print("🏀 Basketball Shot Form Analyzer")
        print("=" * 50)
    
    def analyze_video(self, video_path: str, output_dir: str = "output") -> Dict:
        """
        Analyze a basketball shooting video
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save results
            
        Returns:
            Analysis results dictionary
        """
        print(f"\n🎥 Analyzing video: {video_path}")
        
        # Validate video file
        if not FileUtils.validate_video_file(video_path):
            print(f"❌ Error: Invalid video file: {video_path}")
            return {}
        
        try:
            # Get video information
            video_info = FileUtils.get_video_info(video_path)
            print(f"📊 Video Info: {video_info['frame_count']} frames, "
                  f"{video_info['fps']:.1f} FPS, {video_info['duration']:.1f}s")
            
            # Analyze pose data
            print("🔍 Extracting pose data...")
            pose_results = self.pose_analyzer.analyze_video(video_path, output_dir)
            
            # Process each frame for angle analysis
            print("📐 Calculating joint angles...")
            all_analysis_results = []
            
            for frame_data in pose_results['shot_analysis']:
                for shot_form in frame_data['shot_forms']:
                    if shot_form['analysis_ready']:
                        # Analyze angles
                        angle_analysis = self.angle_analyzer.analyze_frame(shot_form['keypoints'])
                        angle_analysis['frame_number'] = frame_data['frame_number']
                        angle_analysis['timestamp'] = frame_data['timestamp']
                        
                        # Classify shot
                        classification = self.shot_classifier.classify_shot(
                            {name: data['angle'] for name, data in angle_analysis['angles'].items()},
                            {name: data['confidence'] for name, data in angle_analysis['angles'].items()}
                        )
                        
                        # Combine results
                        combined_result = {
                            **angle_analysis,
                            'classification': classification.classification,
                            'confidence': classification.confidence,
                            'feedback': classification.feedback
                        }
                        
                        all_analysis_results.append(combined_result)
            
            # Store session data
            self.current_session_data = all_analysis_results
            self.current_video_path = video_path
            
            # Generate analytics
            print("📈 Generating analytics...")
            summary_stats = self.analytics.generate_summary_statistics(all_analysis_results)
            
            # Compile final results
            results = {
                'video_path': video_path,
                'video_info': video_info,
                'pose_results': pose_results,
                'analysis_results': all_analysis_results,
                'summary_statistics': summary_stats,
                'total_shots_analyzed': len(all_analysis_results)
            }
            
            print(f"✅ Analysis complete! Analyzed {len(all_analysis_results)} shots.")
            return results
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            self.logger.error(f"Analysis error: {str(e)}")
            return {}
    
    def show_statistics(self) -> None:
        """Display current session statistics"""
        if not self.current_session_data:
            print("❌ No analysis data available. Please analyze a video first.")
            return
        
        print("\n📊 SESSION STATISTICS")
        print("-" * 30)
        
        summary = self.analytics.generate_summary_statistics(self.current_session_data)
        
        print(f"Total Shots: {summary['total_shots']}")
        print(f"Average Score: {summary['average_score']:.1f}%")
        print(f"Best Score: {summary['best_score']:.1f}%")
        print(f"Worst Score: {summary['worst_score']:.1f}%")
        print(f"Consistency: {summary['overall_consistency']:.1f}%")
        print(f"Performance Level: {summary['performance_level']}")
        
        print("\nClassification Distribution:")
        for classification, count in summary['classification_distribution'].items():
            percentage = (count / summary['total_shots']) * 100
            print(f"  {classification}: {count} ({percentage:.1f}%)")
        
        print("\nAngle Consistency:")
        for angle_name, angle_stats in summary['angle_consistency'].items():
            print(f"  {angle_name.replace('_', ' ').title()}: "
                  f"{angle_stats['mean']:.1f}° ± {angle_stats['std']:.1f}° "
                  f"(Consistency: {angle_stats['consistency_score']:.1f}%)")
    
    def generate_hexagon_chart(self, save_path: Optional[str] = None) -> None:
        """Generate and display hexagon performance chart"""
        if not self.current_session_data:
            print("❌ No analysis data available. Please analyze a video first.")
            return
        
        print("📊 Generating hexagon chart...")
        
        try:
            fig = self.analytics.create_hexagon_chart(self.current_session_data, save_path)
            plt.show()
            print("✅ Hexagon chart generated successfully!")
        except Exception as e:
            print(f"❌ Error generating hexagon chart: {str(e)}")
    
    def generate_timeline_chart(self, save_path: Optional[str] = None) -> None:
        """Generate and display performance timeline chart"""
        if not self.current_session_data:
            print("❌ No analysis data available. Please analyze a video first.")
            return
        
        print("📈 Generating timeline chart...")
        
        try:
            fig = self.analytics.create_performance_timeline(self.current_session_data, save_path)
            plt.show()
            print("✅ Timeline chart generated successfully!")
        except Exception as e:
            print(f"❌ Error generating timeline chart: {str(e)}")
    
    def generate_angle_distribution_chart(self, save_path: Optional[str] = None) -> None:
        """Generate and display angle distribution charts"""
        if not self.current_session_data:
            print("❌ No analysis data available. Please analyze a video first.")
            return
        
        print("📐 Generating angle distribution charts...")
        
        try:
            fig = self.analytics.create_angle_distribution_chart(self.current_session_data, save_path)
            plt.show()
            print("✅ Angle distribution charts generated successfully!")
        except Exception as e:
            print(f"❌ Error generating angle distribution charts: {str(e)}")
    
    def generate_heatmap(self, save_path: Optional[str] = None) -> None:
        """Generate shot location heatmap"""
        if not self.current_session_data:
            print("❌ No analysis data available. Please analyze a video first.")
            return
        
        print("🔥 Generating shot heatmap...")
        
        try:
            # Add shot locations to heatmap (simplified - using frame positions)
            for i, shot_data in enumerate(self.current_session_data):
                # Use frame number as approximate location
                x = (i / len(self.current_session_data)) * 94  # Court width
                y = 25  # Mid-court
                made = shot_data['form_score'] > 60  # Assume good form = made shot
                
                self.heatmap.add_shot_location(x, y, made, shot_data['confidence'])
            
            # Generate heatmaps
            fig1 = self.heatmap.create_shot_frequency_heatmap()
            plt.show()
            
            fig2 = self.heatmap.create_accuracy_heatmap()
            plt.show()
            
            print("✅ Heatmaps generated successfully!")
        except Exception as e:
            print(f"❌ Error generating heatmaps: {str(e)}")
    
    def export_results(self, format: str = 'csv') -> None:
        """Export analysis results"""
        if not self.current_session_data:
            print("❌ No analysis data available. Please analyze a video first.")
            return
        
        print(f"💾 Exporting results in {format.upper()} format...")
        
        try:
            filepath = self.exporter.export_shot_data(self.current_session_data, 
                                                     f"shot_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                                                     format)
            print(f"✅ Results exported to: {filepath}")
        except Exception as e:
            print(f"❌ Error exporting results: {str(e)}")
    
    def export_comprehensive_report(self) -> None:
        """Export comprehensive analysis report"""
        if not self.current_session_data:
            print("❌ No analysis data available. Please analyze a video first.")
            return
        
        print("📋 Generating comprehensive report...")
        
        try:
            summary_stats = self.analytics.generate_summary_statistics(self.current_session_data)
            filepath = self.exporter.create_comprehensive_report(
                self.current_session_data, 
                summary_stats
            )
            print(f"✅ Comprehensive report exported to: {filepath}")
        except Exception as e:
            print(f"❌ Error generating comprehensive report: {str(e)}")
    
    def show_improvement_recommendations(self) -> None:
        """Show improvement recommendations"""
        if not self.current_session_data:
            print("❌ No analysis data available. Please analyze a video first.")
            return
        
        print("\n💡 IMPROVEMENT RECOMMENDATIONS")
        print("-" * 35)
        
        # Get classifications
        classifications = []
        for shot_data in self.current_session_data:
            classifications.append(shot_data)
        
        recommendations = self.shot_classifier.get_improvement_recommendations(classifications)
        
        for i, recommendation in enumerate(recommendations, 1):
            print(f"{i}. {recommendation}")
    
    def interactive_menu(self) -> None:
        """Display interactive menu"""
        while True:
            print("\n" + "=" * 50)
            print("🏀 BASKETBALL SHOT FORM ANALYZER")
            print("=" * 50)
            print("1. Analyze Video")
            print("2. View Statistics")
            print("3. Generate Hexagon Chart")
            print("4. Generate Timeline Chart")
            print("5. Generate Angle Distribution Charts")
            print("6. Generate Heatmap")
            print("7. Show Improvement Recommendations")
            print("8. Export Results (CSV)")
            print("9. Export Results (JSON)")
            print("10. Export Results (Excel)")
            print("11. Export Comprehensive Report")
            print("12. Exit")
            print("-" * 50)
            
            choice = input("Select an option (1-12): ").strip()
            
            try:
                if choice == "1":
                    video_path = input("Enter path to video file: ").strip()
                    if video_path:
                        self.analyze_video(video_path)
                
                elif choice == "2":
                    self.show_statistics()
                
                elif choice == "3":
                    save_option = input("Save chart to file? (y/n): ").strip().lower()
                    save_path = None
                    if save_option == 'y':
                        save_path = input("Enter save path (or press Enter for default): ").strip()
                        if not save_path:
                            save_path = f"hexagon_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    self.generate_hexagon_chart(save_path)
                
                elif choice == "4":
                    save_option = input("Save chart to file? (y/n): ").strip().lower()
                    save_path = None
                    if save_option == 'y':
                        save_path = input("Enter save path (or press Enter for default): ").strip()
                        if not save_path:
                            save_path = f"timeline_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    self.generate_timeline_chart(save_path)
                
                elif choice == "5":
                    save_option = input("Save charts to file? (y/n): ").strip().lower()
                    save_path = None
                    if save_option == 'y':
                        save_path = input("Enter save path (or press Enter for default): ").strip()
                        if not save_path:
                            save_path = f"angle_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    self.generate_angle_distribution_chart(save_path)
                
                elif choice == "6":
                    self.generate_heatmap()
                
                elif choice == "7":
                    self.show_improvement_recommendations()
                
                elif choice == "8":
                    self.export_results('csv')
                
                elif choice == "9":
                    self.export_results('json')
                
                elif choice == "10":
                    self.export_results('excel')
                
                elif choice == "11":
                    self.export_comprehensive_report()
                
                elif choice == "12":
                    print("👋 Thank you for using Basketball Shot Form Analyzer!")
                    break
                
                else:
                    print("❌ Invalid option. Please select 1-12.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                self.logger.error(f"Menu error: {str(e)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Basketball Shot Form Analyzer")
    parser.add_argument("--video", "-v", help="Path to video file for analysis")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BasketballShotAnalyzer()
    
    if args.video:
        # Analyze specified video
        results = analyzer.analyze_video(args.video, args.output)
        if results:
            print("\n✅ Analysis completed successfully!")
            print("Run with --interactive to explore results and generate visualizations.")
    elif args.interactive:
        # Run interactive mode
        analyzer.interactive_menu()
    else:
        # Show help and start interactive mode
        print("No video specified. Starting interactive mode...")
        print("Use --help for command line options.")
        analyzer.interactive_menu()


if __name__ == "__main__":
    # Import matplotlib here to avoid issues with backend
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    main() 