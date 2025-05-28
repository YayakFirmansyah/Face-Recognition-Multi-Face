#!/usr/bin/env python3
"""
Face Recognition System - Easy Run Script
Provides simple commands to train and run face recognition
"""

import os
import sys
import subprocess
import argparse


def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'main.py',
        'main2.py', 
        'preprocessing.py',
        'facenet_embeddings.py',
        'svm_classifier.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True


def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def train_model(dataset_path, output_dir='models'):
    """Train face recognition model"""
    print(f"🎯 Training model with dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory not found: {dataset_path}")
        print("\nCreate a dataset folder with this structure:")
        print("dataset/")
        print("├── person1/")
        print("│   ├── image1.jpg")
        print("│   ├── image2.jpg")
        print("│   └── ...")
        print("├── person2/")
        print("│   ├── image1.jpg")
        print("│   └── ...")
        print("└── ...")
        return False
    
    try:
        cmd = [sys.executable, 'main.py', 'train', dataset_path, '--output-dir', output_dir]
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False


def run_camera(model_dir='models', camera_index=0, confidence_threshold=0.7):
    """Run real-time face recognition"""
    model_path = os.path.join(model_dir, 'svm_face_model.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print("❌ Trained model not found!")
        print(f"Expected files:")
        print(f"  - {model_path}")
        print(f"  - {encoder_path}")
        print("\nTrain a model first using: python run_face_recognition.py train")
        return False
    
    print("🎥 Starting real-time face recognition...")
    print("Press 'q' to quit when camera window is active")
    
    try:
        cmd = [
            sys.executable, 'main2.py',
            '--model-path', model_path,
            '--encoder-path', encoder_path,
            '--camera-index', str(camera_index),
            '--confidence-threshold', str(confidence_threshold)
        ]
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Camera recognition failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Camera stopped by user")
        return True


def test_prediction(image_paths, model_dir='models'):
    """Test prediction on image files"""
    model_path = os.path.join(model_dir, 'svm_face_model.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print("❌ Trained model not found!")
        return False
    
    print(f"🔍 Testing prediction on {len(image_paths)} images...")
    
    try:
        cmd = [sys.executable, 'main.py', 'predict', model_path, encoder_path] + image_paths
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Prediction failed: {e}")
        return False


def run_demo():
    """Run demo with quick settings"""
    print("🚀 Running demo...")
    
    try:
        cmd = [sys.executable, 'main.py', 'demo']
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        return False


def show_status():
    """Show system status"""
    print("📊 Face Recognition System Status")
    print("=" * 40)
    
    # Check files
    print("\n📁 Required Files:")
    required_files = ['main.py', 'main2.py', 'preprocessing.py', 'facenet_embeddings.py', 'svm_classifier.py']
    for file in required_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"  {status} {file}")
    
    # Check models
    print("\n🤖 Trained Models:")
    model_files = ['models/svm_face_model.pkl', 'models/label_encoder.pkl']
    for file in model_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"  {status} {file}")
    
    # Check dataset
    print("\n📂 Dataset:")
    if os.path.exists('dataset'):
        try:
            people = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
            print(f"  ✅ Found {len(people)} people in dataset/")
            for person in people[:5]:  # Show first 5
                person_path = os.path.join('dataset', person)
                images = [f for f in os.listdir(person_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"    - {person}: {len(images)} images")
            if len(people) > 5:
                print(f"    ... and {len(people) - 5} more")
        except Exception as e:
            print(f"  ⚠️  Error reading dataset: {e}")
    else:
        print("  ❌ No dataset/ folder found")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Face Recognition System - Easy Runner')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Install command
    install_parser = subparsers.add_parser('install', help='Install required packages')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train face recognition model')
    train_parser.add_argument('dataset_path', nargs='?', default='dataset',
                             help='Path to dataset directory (default: dataset)')
    train_parser.add_argument('--output-dir', default='models',
                             help='Output directory for models (default: models)')
    
    # Camera command
    camera_parser = subparsers.add_parser('camera', help='Run real-time face recognition')
    camera_parser.add_argument('--model-dir', default='models',
                              help='Directory containing trained models (default: models)')
    camera_parser.add_argument('--camera-index', type=int, default=0,
                              help='Camera index to use (default: 0)')
    camera_parser.add_argument('--confidence', type=float, default=0.7,
                              help='Recognition confidence threshold (default: 0.7)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Test prediction on images')
    predict_parser.add_argument('images', nargs='+', help='Path to image files')
    predict_parser.add_argument('--model-dir', default='models',
                               help='Directory containing trained models (default: models)')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo with sample data')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    # Check requirements first
    if not check_requirements():
        print("\n💡 Tip: Make sure all required files are in the same directory")
        sys.exit(1)
    
    if args.command == 'install':
        success = install_requirements()
        sys.exit(0 if success else 1)
    
    elif args.command == 'train':
        success = train_model(args.dataset_path, args.output_dir)
        if success:
            print("\n🎉 Training completed! Now you can run:")
            print("python run_face_recognition.py camera")
        sys.exit(0 if success else 1)
    
    elif args.command == 'camera':
        success = run_camera(args.model_dir, args.camera_index, args.confidence)
        sys.exit(0 if success else 1)
    
    elif args.command == 'predict':
        success = test_prediction(args.images, args.model_dir)
        sys.exit(0 if success else 1)
    
    elif args.command == 'demo':
        success = run_demo()
        if success:
            print("\n🎉 Demo completed! You can now run:")
            print("python run_face_recognition.py camera")
        sys.exit(0 if success else 1)
    
    elif args.command == 'status':
        show_status()
        sys.exit(0)
    
    else:
        # Show help and quick start guide
        print("🎯 Face Recognition System - Quick Start Guide")
        print("=" * 50)
        print("\n📋 Available Commands:")
        print("  install  - Install required packages")
        print("  train    - Train face recognition model")
        print("  camera   - Run real-time face recognition")
        print("  predict  - Test prediction on images")
        print("  demo     - Run demo with sample data")
        print("  status   - Show system status")
        
        print("\n🚀 Quick Start:")
        print("1. Install requirements:")
        print("   python run_face_recognition.py install")
        
        print("\n2. Prepare your dataset:")
        print("   Create a 'dataset' folder with subdirectories for each person")
        print("   dataset/")
        print("   ├── john/")
        print("   │   ├── photo1.jpg")
        print("   │   └── photo2.jpg")
        print("   └── jane/")
        print("       ├── photo1.jpg")
        print("       └── photo2.jpg")
        
        print("\n3. Train the model:")
        print("   python run_face_recognition.py train")
        
        print("\n4. Run real-time recognition:")
        print("   python run_face_recognition.py camera")
        
        print("\n📖 For detailed help on each command:")
        print("   python run_face_recognition.py <command> --help")
        
        # Show current status
        print("\n" + "=" * 50)
        show_status()


if __name__ == "__main__":
    main()