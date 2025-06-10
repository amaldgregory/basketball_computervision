#!/usr/bin/env python3
"""
Test script to verify Basketball Shot Form Analyzer setup
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing module imports...")
    
    required_modules = [
        'ultralytics',
        'cv2',
        'numpy',
        'matplotlib',
        'pandas',
        'seaborn',
        'scipy',
        'sklearn',
        'tqdm'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required modules imported successfully!")
        return True

def test_local_modules():
    """Test if our local modules can be imported"""
    print("\n🧪 Testing local module imports...")
    
    local_modules = [
        'pose_analyzer',
        'angle_calculator', 
        'shot_classifier',
        'analytics',
        'heatmap',
        'exporter',
        'utils'
    ]
    
    failed_imports = []
    
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import local modules: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All local modules imported successfully!")
        return True

def test_yolo_model():
    """Test if YOLOv8 model can be loaded"""
    print("\n🧪 Testing YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load the pose model (this will download if not present)
        print("📥 Loading YOLOv8 pose model...")
        model = YOLO('yolov8n-pose.pt')
        print("✅ YOLOv8 pose model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading YOLOv8 model: {e}")
        print("The model will be downloaded automatically on first use.")
        return False

def test_basic_functionality():
    """Test basic functionality of our modules"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test angle calculator
        from angle_calculator import JointAngleCalculator
        calculator = JointAngleCalculator()
        print("✅ Angle calculator initialized")
        
        # Test shot classifier
        from shot_classifier import ShotClassifier
        classifier = ShotClassifier()
        print("✅ Shot classifier initialized")
        
        # Test analytics
        from analytics import ShotAnalytics
        analytics = ShotAnalytics()
        print("✅ Analytics initialized")
        
        # Test exporter
        from exporter import ResultsExporter
        exporter = ResultsExporter()
        print("✅ Exporter initialized")
        
        print("\n✅ All modules initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing functionality: {e}")
        return False

def main():
    """Run all tests"""
    print("🏀 Basketball Shot Form Analyzer - Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test local modules
    if not test_local_modules():
        all_passed = False
    
    # Test YOLO model
    if not test_yolo_model():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The Basketball Shot Form Analyzer is ready to use.")
        print("\nTo start the application, run:")
        print("  python main.py")
        print("\nOr for interactive mode:")
        print("  python main.py --interactive")
    else:
        print("❌ Some tests failed. Please check the errors above and fix them.")
        print("\nCommon solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Make sure you're in the correct directory")
        print("3. Check that all files are present")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 