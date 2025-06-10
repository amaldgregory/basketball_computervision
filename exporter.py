import json
import csv
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from pathlib import Path


class ResultsExporter:
    """Handles exporting shot analysis results in various formats"""
    
    def __init__(self, output_dir: str = "exports"):
        """
        Initialize the exporter
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Supported export formats
        self.supported_formats = ['csv', 'json', 'excel', 'txt']
    
    def export_shot_data(self, shot_data: List[Dict], filename: Optional[str] = None, 
                        format: str = 'csv') -> str:
        """
        Export shot analysis data
        
        Args:
            shot_data: List of shot analysis results
            filename: Optional filename (without extension)
            format: Export format ('csv', 'json', 'excel')
            
        Returns:
            Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shot_analysis_{timestamp}"
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")
        
        filepath = self.output_dir / f"{filename}.{format}"
        
        if format == 'csv':
            return self._export_to_csv(shot_data, filepath)
        elif format == 'json':
            return self._export_to_json(shot_data, filepath)
        elif format == 'excel':
            return self._export_to_excel(shot_data, filepath)
        elif format == 'txt':
            return self._export_to_txt(shot_data, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_to_csv(self, shot_data: List[Dict], filepath: Path) -> str:
        """Export shot data to CSV format"""
        if not shot_data:
            raise ValueError("No shot data to export")
        
        # Flatten the data structure for CSV
        flattened_data = []
        
        for shot in shot_data:
            row = {
                'timestamp': shot.get('timestamp', ''),
                'frame_number': shot.get('frame_number', ''),
                'form_score': shot.get('form_score', ''),
                'classification': shot.get('classification', ''),
                'confidence': shot.get('confidence', '')
            }
            
            # Add angle data
            if 'angles' in shot:
                for angle_name, angle_data in shot['angles'].items():
                    row[f'{angle_name}_angle'] = angle_data.get('angle', '')
                    row[f'{angle_name}_confidence'] = angle_data.get('confidence', '')
                    row[f'{angle_name}_is_valid'] = angle_data.get('is_valid', '')
            
            # Add feedback
            if 'feedback' in shot:
                row['feedback'] = '; '.join(shot['feedback'])
            
            flattened_data.append(row)
        
        # Write to CSV
        df = pd.DataFrame(flattened_data)
        df.to_csv(filepath, index=False)
        
        print(f"Shot data exported to CSV: {filepath}")
        return str(filepath)
    
    def _export_to_json(self, shot_data: List[Dict], filepath: Path) -> str:
        """Export shot data to JSON format"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_shots': len(shot_data),
            'shot_data': shot_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Shot data exported to JSON: {filepath}")
        return str(filepath)
    
    def _export_to_excel(self, shot_data: List[Dict], filepath: Path) -> str:
        """Export shot data to Excel format with multiple sheets"""
        if not shot_data:
            raise ValueError("No shot data to export")
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main shot data sheet
            flattened_data = []
            for shot in shot_data:
                row = {
                    'Timestamp': shot.get('timestamp', ''),
                    'Frame Number': shot.get('frame_number', ''),
                    'Form Score': shot.get('form_score', ''),
                    'Classification': shot.get('classification', ''),
                    'Confidence': shot.get('confidence', '')
                }
                
                if 'angles' in shot:
                    for angle_name, angle_data in shot['angles'].items():
                        row[f'{angle_name.replace("_", " ").title()} Angle'] = angle_data.get('angle', '')
                        row[f'{angle_name.replace("_", " ").title()} Confidence'] = angle_data.get('confidence', '')
                        row[f'{angle_name.replace("_", " ").title()} Valid'] = angle_data.get('is_valid', '')
                
                if 'feedback' in shot:
                    row['Feedback'] = '; '.join(shot['feedback'])
                
                flattened_data.append(row)
            
            df_main = pd.DataFrame(flattened_data)
            df_main.to_excel(writer, sheet_name='Shot Data', index=False)
            
            # Summary statistics sheet
            if shot_data:
                summary_data = self._generate_summary_data(shot_data)
                df_summary = pd.DataFrame([summary_data])
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Angle analysis sheet
            angle_data = self._extract_angle_data(shot_data)
            if angle_data:
                df_angles = pd.DataFrame(angle_data)
                df_angles.to_excel(writer, sheet_name='Angle Analysis', index=False)
        
        print(f"Shot data exported to Excel: {filepath}")
        return str(filepath)
    
    def _export_to_txt(self, shot_data: List[Dict], filepath: Path) -> str:
        """Export shot data to human-readable text format"""
        with open(filepath, 'w') as f:
            f.write("BASKETBALL SHOT FORM ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Shots Analyzed: {len(shot_data)}\n\n")
            
            if shot_data:
                # Summary statistics
                summary = self._generate_summary_data(shot_data)
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 20 + "\n")
                for key, value in summary.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
                
                # Individual shot details
                f.write("DETAILED SHOT ANALYSIS\n")
                f.write("-" * 25 + "\n")
                
                for i, shot in enumerate(shot_data, 1):
                    f.write(f"\nShot #{i}\n")
                    f.write(f"Frame: {shot.get('frame_number', 'N/A')}\n")
                    f.write(f"Score: {shot.get('form_score', 'N/A'):.1f}\n")
                    f.write(f"Classification: {shot.get('classification', 'N/A')}\n")
                    
                    if 'angles' in shot:
                        f.write("Angles:\n")
                        for angle_name, angle_data in shot['angles'].items():
                            f.write(f"  {angle_name.replace('_', ' ').title()}: "
                                   f"{angle_data.get('angle', 'N/A'):.1f}° "
                                   f"({'Valid' if angle_data.get('is_valid') else 'Invalid'})\n")
                    
                    if 'feedback' in shot:
                        f.write("Feedback:\n")
                        for feedback in shot['feedback']:
                            f.write(f"  {feedback}\n")
        
        print(f"Shot data exported to text: {filepath}")
        return str(filepath)
    
    def _generate_summary_data(self, shot_data: List[Dict]) -> Dict:
        """Generate summary statistics for export"""
        scores = [shot.get('form_score', 0) for shot in shot_data]
        classifications = [shot.get('classification', 'Unknown') for shot in shot_data]
        
        # Classification counts
        class_counts = {}
        for classification in classifications:
            class_counts[classification] = class_counts.get(classification, 0) + 1
        
        return {
            'total_shots': len(shot_data),
            'average_score': np.mean(scores) if scores else 0,
            'best_score': max(scores) if scores else 0,
            'worst_score': min(scores) if scores else 0,
            'score_std': np.std(scores) if scores else 0,
            'excellent_shots': class_counts.get('Excellent', 0),
            'good_shots': class_counts.get('Good', 0),
            'fair_shots': class_counts.get('Fair', 0),
            'needs_work_shots': class_counts.get('Needs Work', 0)
        }
    
    def _extract_angle_data(self, shot_data: List[Dict]) -> List[Dict]:
        """Extract angle data for separate analysis"""
        angle_data = []
        
        for shot in shot_data:
            if 'angles' in shot:
                for angle_name, angle_info in shot['angles'].items():
                    angle_data.append({
                        'shot_number': shot.get('frame_number', ''),
                        'angle_type': angle_name,
                        'angle_value': angle_info.get('angle', ''),
                        'confidence': angle_info.get('confidence', ''),
                        'is_valid': angle_info.get('is_valid', ''),
                        'form_score': shot.get('form_score', ''),
                        'classification': shot.get('classification', '')
                    })
        
        return angle_data
    
    def export_analytics_report(self, analytics_data: Dict, filename: Optional[str] = None) -> str:
        """
        Export comprehensive analytics report
        
        Args:
            analytics_data: Analytics data dictionary
            filename: Optional filename
            
        Returns:
            Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_report_{timestamp}"
        
        filepath = self.output_dir / f"{filename}.json"
        
        export_data = {
            'report_timestamp': datetime.now().isoformat(),
            'analytics_data': analytics_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Analytics report exported: {filepath}")
        return str(filepath)
    
    def export_heatmap_data(self, heatmap_data: Dict, filename: Optional[str] = None) -> str:
        """
        Export heatmap data
        
        Args:
            heatmap_data: Heatmap data dictionary
            filename: Optional filename
            
        Returns:
            Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heatmap_data_{timestamp}"
        
        filepath = self.output_dir / f"{filename}.json"
        
        with open(filepath, 'w') as f:
            json.dump(heatmap_data, f, indent=2, default=str)
        
        print(f"Heatmap data exported: {filepath}")
        return str(filepath)
    
    def create_comprehensive_report(self, shot_data: List[Dict], analytics_data: Dict, 
                                  heatmap_data: Optional[Dict] = None) -> str:
        """
        Create a comprehensive report with all analysis data
        
        Args:
            shot_data: Shot analysis data
            analytics_data: Analytics data
            heatmap_data: Optional heatmap data
            
        Returns:
            Path to the comprehensive report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_report_{timestamp}"
        
        # Create comprehensive data structure
        comprehensive_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_shots': len(shot_data),
                'report_version': '1.0'
            },
            'shot_analysis': shot_data,
            'analytics_summary': analytics_data,
            'heatmap_data': heatmap_data or {},
            'export_formats': {
                'csv_ready': True,
                'json_ready': True,
                'excel_ready': True
            }
        }
        
        # Export in multiple formats
        files_created = []
        
        # JSON format (complete data)
        json_file = self.export_shot_data(shot_data, filename, 'json')
        files_created.append(json_file)
        
        # Excel format (structured data)
        excel_file = self.export_shot_data(shot_data, filename, 'excel')
        files_created.append(excel_file)
        
        # CSV format (flattened data)
        csv_file = self.export_shot_data(shot_data, filename, 'csv')
        files_created.append(csv_file)
        
        # Text format (human readable)
        txt_file = self.export_shot_data(shot_data, filename, 'txt')
        files_created.append(txt_file)
        
        print(f"Comprehensive report created with {len(files_created)} files:")
        for file in files_created:
            print(f"  - {file}")
        
        return json_file  # Return the main JSON file path
    
    def get_export_history(self) -> List[str]:
        """Get list of exported files"""
        if not self.output_dir.exists():
            return []
        
        files = []
        for file in self.output_dir.iterdir():
            if file.is_file():
                files.append(str(file))
        
        return sorted(files, reverse=True)  # Most recent first
    
    def cleanup_old_exports(self, days_old: int = 30) -> int:
        """
        Clean up old export files
        
        Args:
            days_old: Remove files older than this many days
            
        Returns:
            Number of files removed
        """
        if not self.output_dir.exists():
            return 0
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        removed_count = 0
        
        for file in self.output_dir.iterdir():
            if file.is_file():
                if file.stat().st_mtime < cutoff_time:
                    file.unlink()
                    removed_count += 1
        
        print(f"Removed {removed_count} old export files")
        return removed_count


if __name__ == "__main__":
    # Test the exporter
    exporter = ResultsExporter()
    
    # Example shot data
    test_shot_data = [
        {
            'timestamp': datetime.now().isoformat(),
            'frame_number': 1,
            'form_score': 85.5,
            'classification': 'Excellent',
            'confidence': 0.9,
            'angles': {
                'shooting_elbow': {'angle': 90, 'confidence': 0.9, 'is_valid': True},
                'shoulder_alignment': {'angle': 175, 'confidence': 0.8, 'is_valid': True}
            },
            'feedback': ['✓ Good shooting elbow: 90.0°', '✓ Good shoulder alignment: 175.0°']
        },
        {
            'timestamp': datetime.now().isoformat(),
            'frame_number': 2,
            'form_score': 75.0,
            'classification': 'Good',
            'confidence': 0.8,
            'angles': {
                'shooting_elbow': {'angle': 88, 'confidence': 0.9, 'is_valid': True},
                'shoulder_alignment': {'angle': 172, 'confidence': 0.7, 'is_valid': True}
            },
            'feedback': ['✓ Good shooting elbow: 88.0°', '✓ Good shoulder alignment: 172.0°']
        }
    ]
    
    # Test different export formats
    print("Testing export functionality...")
    
    # Export to CSV
    csv_file = exporter.export_shot_data(test_shot_data, 'test_export', 'csv')
    print(f"CSV export: {csv_file}")
    
    # Export to JSON
    json_file = exporter.export_shot_data(test_shot_data, 'test_export', 'json')
    print(f"JSON export: {json_file}")
    
    # Export to Excel
    excel_file = exporter.export_shot_data(test_shot_data, 'test_export', 'excel')
    print(f"Excel export: {excel_file}")
    
    # Export to text
    txt_file = exporter.export_shot_data(test_shot_data, 'test_export', 'txt')
    print(f"Text export: {txt_file}")
    
    # Show export history
    print("\nExport history:")
    for file in exporter.get_export_history():
        print(f"  {file}") 