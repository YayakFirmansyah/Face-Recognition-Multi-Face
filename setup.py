#!/usr/bin/env python3
"""
Setup script for Face Recognition System
Automatically sets up the environment and checks compatibility
"""

import os
import sys
import subprocess
import platform
import pkg_resources
from pathlib import Path


def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🎭 Face Recognition System Setup")
    print("=" * 60)
    print()


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Minimum required: Python 3.7+")
        print("   Recommended: Python 3.8+")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True


def check_system_requirements():
    """Check system requirements"""
    print("\n💻 Checking system requirements...")
    
    # Check operating system
    os_name = platform.system()
    print(f"   OS: {os_name} {platform.release()}")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment")
        print("   Recommendation: Use virtual environment to avoid conflicts")
    
    return True


def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    try:
        # Upgrade pip first
        print("   Upgrading pip...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                            stdout=subprocess.DEVNULL)
        
        # Install requirements
        print("   Installing packages from requirements.txt...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        
        print("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        print("\n💡 Try manual installation:")
        print("   pip install -r requirements.txt")
        return False


def check_installed_packages():
    """Check if all required packages are installed"""
    print("\n🔍 Checking installed packages...")
    
    required_packages = [
        'opencv-python',
        'mtcnn', 
        'keras-facenet',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pandas',
        'numpy',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Try to import the package
            if package == 'opencv-python':
                import cv2
                version = cv2.__version__
            elif package == 'keras-facenet':
                import keras_facenet
                version = keras_facenet.__version__
            elif package == 'Pillow':
                import PIL
                version = PIL.__version__
            else:
                module = __import__(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
            
            print(f"   ✅ {package}: {version}")
            
        except ImportError:
            print(f"   ❌ {package}: Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("\n✅ All required packages are installed!")
    return True


def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        'dataset',
        'models',
        'temp',
        'logs'
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ {dir_name}/")
    
    # Create sample dataset structure
    sample_dirs = [
        'dataset/sample_person1',
        'dataset/sample_person2'
    ]
    
    for dir_name in sample_dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Create README for dataset
    dataset_readme = """# Dataset Directory

Place your face images here with the following structure:

dataset/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── person2/
│   ├── photo1.jpg
│   └── photo2.jpg
└── ...

Tips:
- Use 3-5 images per person for best results
- Ensure good lighting and clear face visibility
- Supported formats: JPG, PNG, JPEG
- Minimum resolution: 160x160 pixels
"""
    
    with open('dataset/README.md', 'w') as f:
        f.write(dataset_readme)
    
    print("   ✅ Sample dataset structure created")


def check_camera():
    """Check if camera is available"""
    print("\n📷 Checking camera availability...")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("   ⚠️  Camera 0 not available")
            
            # Try other camera indices
            for i in range(1, 4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"   ✅ Camera {i} is available")
                    cap.release()
                    return True
                cap.release()
            
            print("   ❌ No cameras found")
            print("   💡 You can still use the system for image prediction")
            return False
        
        else:
            # Test camera
            ret, frame = cap.read()
            if ret:
                print("   ✅ Camera 0 is working properly")
                cap.release()
                return True
            else:
                print("   ❌ Camera 0 found but cannot capture frames")
                cap.release()
                return False
                
    except Exception as e:
        print(f"   ❌ Error checking camera: {e}")
        return False


def create_sample_config():
    """Create sample configuration file"""
    print("\n⚙️  Creating sample configuration...")
    
    config_content = '''# Face Recognition System Configuration

# Training Configuration
TRAINING_CONFIG = {
    "preprocessing": {
        "target_size": (160, 160),
        "confidence_threshold": 0.9
    },
    "augmentation": {
        "use_augmentation": True,
        "augmentation_factor": 2
    },
    "training": {
        "test_size": 0.2,
        "validation_size": 0.1,
        "search_type": "comprehensive",  # Options: quick, comprehensive, fine_tune
        "cv_folds": 5
    },
    "evaluation": {
        "confidence_threshold": 0.7,
        "plot_results": True
    }
}

# Camera Configuration
CAMERA_CONFIG = {
    "camera_index": 0,
    "confidence_threshold": 0.7,
    "frame_skip": 2,  # Process every N frames
    "use_mtcnn": True,  # Use MTCNN instead of Haar cascade
    "resolution": (640, 480)
}

# Model Paths
MODEL_PATHS = {
    "model_dir": "models",
    "svm_model": "models/svm_face_model.pkl",
    "label_encoder": "models/label_encoder.pkl",
    "embeddings": "models/face_embeddings.npz"
}
'''
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("   ✅ config.py created")


def run_quick_test():
    """Run a quick test to ensure everything works"""
    print("\n🧪 Running quick system test...")
    
    try:
        # Test imports
        print("   Testing imports...")
        import cv2
        import numpy as np
        import tensorflow as tf
        from mtcnn.mtcnn import MTCNN
        from keras_facenet import FaceNet
        from sklearn.svm import SVC
        
        print("   ✅ All imports successful")
        
        # Test FaceNet loading
        print("   Testing FaceNet model...")
        try:
            facenet = FaceNet()
            print("   ✅ FaceNet model loaded successfully")
        except Exception as e:
            print(f"   ❌ FaceNet loading failed: {e}")
            return False
        
        # Test MTCNN
        print("   Testing MTCNN detector...")
        try:
            detector = MTCNN()
            print("   ✅ MTCNN detector initialized")
        except Exception as e:
            print(f"   ❌ MTCNN initialization failed: {e}")
            return False
        
        print("\n✅ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ System test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Prepare your dataset:")
    print("   - Add face images to dataset/ folder")
    print("   - Each person should have their own subfolder")
    print("   - Use 3-5 clear face images per person")
    
    print("\n2. Train the model:")
    print("   python run_face_recognition.py train")
    
    print("\n3. Run real-time recognition:")
    print("   python run_face_recognition.py camera")
    
    print("\n📖 Quick Commands:")
    print("   python run_face_recognition.py status    # Check system status")
    print("   python run_face_recognition.py demo      # Run demo")
    print("   python run_face_recognition.py --help    # Show all commands")
    
    print("\n📁 Important Files:")
    print("   - README.md: Complete documentation")
    print("   - config.py: Configuration settings")
    print("   - dataset/: Place your face images here")
    print("   - models/: Trained models will be saved here")


def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Install requirements
    print("\n" + "-" * 60)
    install_choice = input("📦 Install required packages? (y/n): ").lower()
    if install_choice in ['y', 'yes']:
        if not install_requirements():
            sys.exit(1)
    
    # Check installed packages
    if not check_installed_packages():
        print("\n💡 Some packages are missing. Please install them manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Create sample configuration
    create_sample_config()
    
    # Check camera
    check_camera()
    
    # Run quick test
    if not run_quick_test():
        print("\n⚠️  Some components failed testing, but you can still proceed")
        print("   The system might work for basic functionality")
    
    # Print next steps
    print_next_steps()
    
    print("\n🚀 Ready to go! Happy face recognition! 🎭")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)