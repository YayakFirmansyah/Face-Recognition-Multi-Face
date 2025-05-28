# main.py - Main Face Recognition System
import os
import pickle
import numpy as np
from face_detector import FaceDetector
from feature_extractor import FeatureExtractor
from face_classifier import FaceClassifier
from dataset_loader import DatasetLoader
from realtime_recognizer import RealtimeRecognizer

class FaceRecognitionSystem:
    def __init__(self):
        """Initialize all components"""
        print("🚀 Initializing Face Recognition System...")
        
        # Initialize components
        self.detector = FaceDetector()
        self.extractor = FeatureExtractor()
        self.classifier = FaceClassifier()
        self.loader = DatasetLoader(self.detector, self.extractor)
        self.recognizer = RealtimeRecognizer(self.detector, self.extractor, self.classifier)
        
        # Data storage
        self.embeddings = None
        self.labels = None
        self.names = None
        
        print("✅ System initialized successfully!")
    
    def load_dataset(self, dataset_path):
        """Load dataset from folder"""
        print(f"\n📂 Loading dataset from: {dataset_path}")
        
        # Validate structure first
        is_valid, result = self.loader.validate_structure(dataset_path)
        
        if not is_valid:
            print(f"❌ Invalid dataset structure: {result}")
            return False
        
        print(f"✅ Dataset structure valid:")
        print(f"  {result['total_persons']} persons, {result['total_images']} images")
        
        # Load data
        self.embeddings, self.labels, self.names, stats = self.loader.load_from_folder(dataset_path)
        
        if self.embeddings is None:
            return False
        
        # Show dataset info
        self.loader.show_dataset_info(self.labels)
        return True
    
    def train_model(self, use_optimization=True):
        """Train SVM classifier"""
        if self.embeddings is None:
            print("❌ No dataset loaded! Load dataset first.")
            return False
        
        print(f"\n🧠 Training SVM classifier...")
        
        # Convert to numpy arrays
        X = np.array(self.embeddings)
        y = np.array(self.labels)
        
        # Train with optimization
        accuracy = self.classifier.train(
            X, y, 
            use_augmentation=use_optimization,
            use_balancing=use_optimization,
            cv_folds=5
        )
        
        print(f"✅ Training completed! Accuracy: {accuracy:.3f}")
        return True
    
    def evaluate_model(self):
        """Evaluate trained model"""
        if self.classifier.model is None:
            print("❌ No trained model! Train model first.")
            return
        
        print(f"\n📊 Evaluating model...")
        X = np.array(self.embeddings)
        y = np.array(self.labels)
        
        accuracy = self.classifier.evaluate(X, y)
        return accuracy
    
    def start_realtime(self):
        """Start real-time recognition"""
        if self.classifier.model is None:
            print("❌ No trained model! Train model first.")
            return
        
        self.recognizer.start_recognition()
    
    def save_model(self, filename):
        """Save trained model"""
        if self.classifier.model is None:
            print("❌ No trained model to save!")
            return
        
        model_data = {
            'classifier': self.classifier,
            'embeddings': self.embeddings,
            'labels': self.labels,
            'names': self.names
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {filename}")
    
    def load_model(self, filename):
        """Load trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.embeddings = model_data['embeddings']
            self.labels = model_data['labels']
            self.names = model_data['names']
            
            print(f"📂 Model loaded from: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Main application"""
    print("=" * 50)
    print("🎯 FACE RECOGNITION SYSTEM")
    print("🔧 MTCNN + FaceNet + SVM")
    print("=" * 50)
    
    # Initialize system
    system = FaceRecognitionSystem()
    
    while True:
        print(f"\n📋 MENU:")
        print("1. 📂 Load Dataset")
        print("2. 🧠 Train Model")
        print("3. 📊 Evaluate Model")
        print("4. 🎥 Start Real-time Recognition")
        print("5. 💾 Save Model")
        print("6. 📂 Load Model")
        print("7. 🚪 Exit")
        
        choice = input("\n🔘 Choose option (1-7): ").strip()
        
        if choice == '1':
            dataset_path = input("📁 Enter dataset path: ").strip()
            system.load_dataset(dataset_path)
            
        elif choice == '2':
            use_opt = input("⚡ Use optimization? (y/n, default=y): ").lower() != 'n'
            system.train_model(use_optimization=use_opt)
            
        elif choice == '3':
            system.evaluate_model()
            
        elif choice == '4':
            system.start_realtime()
            
        elif choice == '5':
            filename = input("💾 Enter filename (e.g., model.pkl): ").strip()
            if not filename.endswith('.pkl'):
                filename += '.pkl'
            system.save_model(filename)
            
        elif choice == '6':
            filename = input("📂 Enter model filename: ").strip()
            system.load_model(filename)
            
        elif choice == '7':
            print("👋 Thank you for using Face Recognition System!")
            break
            
        else:
            print("❌ Invalid option! Please choose 1-7.")

def quick_demo():
    """Quick demonstration"""
    print("🚀 QUICK DEMO MODE")
    
    # Get dataset path
    dataset_path = input("📁 Enter your dataset path: ").strip()
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset path not found!")
        return
    
    # Initialize and run
    system = FaceRecognitionSystem()
    
    print("\n🔄 Running complete pipeline...")
    
    # Load → Train → Evaluate → Save → Real-time
    if system.load_dataset(dataset_path):
        if system.train_model(use_optimization=True):
            system.evaluate_model()
            system.save_model("trained_model.pkl")
            
            print("\n🎥 Starting real-time recognition in 3 seconds...")
            import time
            time.sleep(3)
            system.start_realtime()

if __name__ == "__main__":
    print("🎯 Select Mode:")
    print("1. 📋 Interactive Menu")
    print("2. 🚀 Quick Demo")
    
    mode = input("🔘 Choose mode (1/2): ").strip()
    
    if mode == '2':
        quick_demo()
    else:
        main()