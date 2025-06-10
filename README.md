# 🏀 Basketball Shot Form Analyzer

A comprehensive basketball shot form analysis tool using YOLOv8 pose estimation to analyze shooting mechanics, classify shot quality, and provide actionable feedback.

## Features

- **Pose Estimation**: YOLOv8-based human pose detection
- **Shot Form Analysis**: Joint angle calculations for shooting mechanics
- **Shot Classification**: Automatic classification of shot quality (Excellent/Good/Needs Work)
- **Statistics Tracking**: Comprehensive shot data collection and analysis
- **Hexagon Charts**: Visual performance metrics display
- **Export Capabilities**: CSV and JSON export for data analysis
- **Heat Map Visualization**: Shot location and accuracy mapping (optional)

## Key Joint Angles Analyzed

- **Shooting Elbow Angle**: Shoulder-Elbow-Wrist alignment
- **Shoulder Alignment**: Hip-Shoulder-Elbow positioning
- **Wrist Snap**: Follow-through mechanics

## Installation

1. Ensure you have Python 3.11+ installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

## Project Structure

```
basketball/
├── main.py                 # Main application entry point
├── pose_analyzer.py        # YOLOv8 pose estimation and analysis
├── angle_calculator.py     # Joint angle calculations
├── shot_classifier.py      # Shot quality classification
├── analytics.py           # Statistics and visualization
├── heatmap.py             # Heat map generation
├── exporter.py            # Data export functionality
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## How It Works

1. **Video Processing**: Upload basketball shooting videos
2. **Pose Detection**: YOLOv8 extracts 17 key body points
3. **Angle Calculation**: Compute critical shooting angles
4. **Form Analysis**: Compare against optimal shooting mechanics
5. **Classification**: Rate shot quality and provide feedback
6. **Visualization**: Generate charts and heat maps
7. **Export**: Save results for further analysis

## Optimal Shooting Form Criteria

- **Shooting Elbow**: 85-95° for optimal arc
- **Shoulder Alignment**: 170-180° for proper alignment
- **Wrist Snap**: 45-65° for proper follow-through

## License

MIT License - feel free to use and modify for your basketball training needs! 