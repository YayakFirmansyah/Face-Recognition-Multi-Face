# -*- coding: utf-8 -*-
"""
Real-time Face Recognition System
Integrates with trained models from main.py for live camera face recognition
"""

import cv2 as cv
import numpy as np
import os
import argparse
import sys
import time
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import custom modules
try:
    from preprocessing import FacePreprocessor
    from facenet_embeddings import FaceNetEmbedder
    from svm_classifier import SVMFaceClassifier
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure all required files are in the same directory:")
    print("- preprocessing.py")
    print("- facenet_embeddings.py") 
    print("- svm_classifier.py")
    sys.exit(1)


class RealTimeFaceRecognizer:
    """Real-time face recognition using webcam"""
    
    def __init__(self, model_path, encoder_path, confidence_threshold=0.7):
        """
        Initialize real-time face recognizer
        
        Args:
            model_path (str): Path to trained SVM model
            encoder_path (str): Path to label encoder
            confidence_threshold (float): Minimum confidence for recognition
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.confidence_threshold = confidence_threshold
        
        # Recognition components
        self.preprocessor = None
        self.embedder = None
        self.classifier = None
        
        # Camera settings
        self.camera = None
        self.frame_width = 640
        self.frame_height = 480
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Recognition cache (to avoid re-computing on every frame)
        self.recognition_cache = {}
        self.cache_timeout = 1.0  # Cache results for 1 second
        
        # UI settings
        self.colors = {
            'recognized': (0, 255, 0),      # Green for recognized faces
            'unrecognized': (0, 0, 255),   # Red for unrecognized faces
            'low_confidence': (0, 255, 255) # Yellow for low confidence
        }
        
        print("Initializing Real-time Face Recognizer...")
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all recognition components"""
        try:
            print("Loading face detection and recognition models...")
            
            # Initialize preprocessor with MTCNN
            self.preprocessor = FacePreprocessor(
                target_size=(160, 160),
                confidence_threshold=0.9
            )
            
            # Initialize FaceNet embedder
            self.embedder = FaceNetEmbedder()
            
            # Initialize and load trained classifier
            self.classifier = SVMFaceClassifier()
            self.classifier.load_model(self.model_path, self.encoder_path)
            
            print("✓ All models loaded successfully!")
            print(f"✓ Trained on {len(self.classifier.encoder.classes_)} people")
            print(f"✓ People in database: {', '.join(self.classifier.encoder.classes_)}")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            sys.exit(1)
    
    def initialize_camera(self, camera_index=0):
        """Initialize camera capture"""
        try:
            print(f"Initializing camera {camera_index}...")
            self.camera = cv.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Cannot open camera {camera_index}")
            
            # Set camera properties
            self.camera.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv.CAP_PROP_FPS, 30)
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret:
                raise ValueError("Cannot read from camera")
            
            print(f"✓ Camera initialized successfully")
            print(f"✓ Resolution: {self.frame_width}x{self.frame_height}")
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            print("Available options:")
            print("- Check if camera is connected")
            print("- Try different camera index (0, 1, 2, etc.)")
            print("- Make sure no other application is using the camera")
            sys.exit(1)
    
    def detect_faces_mtcnn(self, frame):
        """
        Detect faces using MTCNN (more accurate but slower)
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of detected face regions and info
        """
        try:
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.preprocessor.detector.detect_faces(rgb_frame)
            
            faces_info = []
            for detection in detections:
                if detection['confidence'] >= 0.9:  # High confidence faces only
                    x, y, w, h = detection['box']
                    x, y = max(0, x), max(0, y)  # Ensure positive coordinates
                    
                    # Add padding
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(frame.shape[1] - x, w + 2 * padding)
                    h = min(frame.shape[0] - y, h + 2 * padding)
                    
                    faces_info.append({
                        'box': (x, y, w, h),
                        'confidence': detection['confidence'],
                        'landmarks': detection.get('keypoints', {})
                    })
            
            return faces_info
            
        except Exception as e:
            print(f"Error in MTCNN detection: {e}")
            return []
    
    def detect_faces_haar(self, frame):
        """
        Detect faces using Haar Cascade (faster but less accurate)
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of detected face regions
        """
        try:
            # Load Haar cascade if not already loaded
            if not hasattr(self, 'haar_cascade'):
                cascade_path = "haarcascade_frontalface_alt2.xml"
                if not os.path.exists(cascade_path):
                    # Try default OpenCV path
                    cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
                
                self.haar_cascade = cv.CascadeClassifier(cascade_path)
                
                if self.haar_cascade.empty():
                    print("⚠️  Warning: Haar cascade not found, using MTCNN only")
                    return []
            
            # Convert to grayscale
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.haar_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(60, 60)
            )
            
            faces_info = []
            for (x, y, w, h) in faces:
                faces_info.append({
                    'box': (x, y, w, h),
                    'confidence': 0.95,  # Default confidence for Haar
                    'landmarks': {}
                })
            
            return faces_info
            
        except Exception as e:
            print(f"Error in Haar detection: {e}")
            return []
    
    def recognize_face(self, face_region):
        """
        Recognize a face region
        
        Args:
            face_region (numpy.ndarray): Face image region
            
        Returns:
            dict: Recognition results
        """
        try:
            # Resize face to expected input size
            face_resized = cv.resize(face_region, (160, 160))
            
            # Normalize to [0, 1]
            if face_resized.max() > 1.0:
                face_normalized = face_resized.astype('float32') / 255.0
            else:
                face_normalized = face_resized.astype('float32')
            
            # Generate embedding
            embedding = self.embedder.get_single_embedding(face_normalized)
            
            if embedding is None:
                return {
                    'success': False,
                    'error': 'Failed to generate embedding'
                }
            
            # Make prediction
            predicted_label, confidence, is_confident = self.classifier.predict_single(
                embedding, self.confidence_threshold
            )
            
            return {
                'success': True,
                'predicted_person': predicted_label,
                'confidence': confidence,
                'is_confident': is_confident
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def draw_face_info(self, frame, face_info, recognition_result, face_id):
        """
        Draw face bounding box and recognition info on frame
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            face_info (dict): Face detection information
            recognition_result (dict): Recognition results
            face_id (int): Face identifier for this frame
        """
        x, y, w, h = face_info['box']
        
        # Determine color based on recognition result
        if recognition_result['success']:
            if recognition_result['is_confident']:
                color = self.colors['recognized']
                status = "✓"
            else:
                color = self.colors['low_confidence']
                status = "?"
            
            label = recognition_result['predicted_person']
            confidence = recognition_result['confidence']
            
        else:
            color = self.colors['unrecognized']
            label = "Unknown"
            confidence = 0.0
            status = "✗"
        
        # Draw bounding box
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare text
        main_text = f"{status} {label}"
        conf_text = f"Conf: {confidence:.2f}"
        
        # Calculate text sizes
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (main_w, main_h), _ = cv.getTextSize(main_text, font, font_scale, thickness)
        (conf_w, conf_h), _ = cv.getTextSize(conf_text, font, font_scale - 0.1, 1)
        
        # Draw text background
        bg_color = (0, 0, 0)  # Black background
        cv.rectangle(frame, 
                    (x, y - main_h - conf_h - 10),
                    (x + max(main_w, conf_w) + 10, y),
                    bg_color, -1)
        
        # Draw text
        cv.putText(frame, main_text,
                  (x + 5, y - conf_h - 5),
                  font, font_scale, color, thickness)
        
        cv.putText(frame, conf_text,
                  (x + 5, y - 5),
                  font, font_scale - 0.1, (255, 255, 255), 1)
        
        # Draw landmarks if available
        if 'landmarks' in face_info and face_info['landmarks']:
            for landmark_name, (lx, ly) in face_info['landmarks'].items():
                cv.circle(frame, (int(lx), int(ly)), 2, (255, 255, 0), -1)
    
    def draw_ui_overlay(self, frame):
        """Draw UI overlay with FPS and instructions"""
        height, width = frame.shape[:2]
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv.putText(frame, fps_text, (10, 25),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw people count
        people_text = f"Known people: {len(self.classifier.encoder.classes_)}"
        cv.putText(frame, people_text, (10, 50),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw instructions
        instructions = "Press 'q' to quit, 's' to save frame, 'h' for help"
        cv.putText(frame, instructions, (10, height - 10),
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def save_frame(self, frame):
        """Save current frame with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_recognition_frame_{timestamp}.jpg"
        cv.imwrite(filename, frame)
        print(f"📸 Frame saved as {filename}")
    
    def show_help(self):
        """Show help information"""
        help_text = """
        🎥 Real-time Face Recognition Controls:
        ========================================
        
        Key Controls:
        - 'q' or ESC: Quit application
        - 's': Save current frame
        - 'h': Show this help
        - 'r': Reset recognition cache
        - 'c': Toggle confidence display
        
        Recognition Info:
        - Green box: Recognized person (high confidence)
        - Yellow box: Recognized person (low confidence)
        - Red box: Unknown person
        
        Performance Tips:
        - Ensure good lighting
        - Face the camera directly
        - Keep face centered in frame
        - Avoid rapid movements
        """
        print(help_text)
    
    def run(self, camera_index=0, use_mtcnn=True, frame_skip=2):
        """
        Run real-time face recognition
        
        Args:
            camera_index (int): Camera index to use
            use_mtcnn (bool): Use MTCNN for face detection (more accurate but slower)
            frame_skip (int): Process every Nth frame for better performance
        """
        print("\n🎥 Starting Real-time Face Recognition")
        print("="*50)
        
        # Initialize camera
        self.initialize_camera(camera_index)
        
        # Show initial help
        self.show_help()
        
        frame_count = 0
        
        try:
            print("✓ Starting camera feed... Press 'q' to quit")
            
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv.flip(frame, 1)
                
                # Process every Nth frame for better performance
                if frame_count % frame_skip == 0:
                    
                    # Detect faces
                    if use_mtcnn:
                        faces_info = self.detect_faces_mtcnn(frame)
                    else:
                        faces_info = self.detect_faces_haar(frame)
                    
                    # Process each detected face
                    for i, face_info in enumerate(faces_info):
                        x, y, w, h = face_info['box']
                        
                        # Extract face region
                        face_region = frame[y:y+h, x:x+w]
                        
                        if face_region.size > 0:
                            # Recognize face
                            recognition_result = self.recognize_face(face_region)
                            
                            # Draw face info
                            self.draw_face_info(frame, face_info, recognition_result, i)
                
                # Draw UI overlay
                self.draw_ui_overlay(frame)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv.imshow("Real-time Face Recognition", frame)
                
                # Handle key presses
                key = cv.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("👋 Exiting...")
                    break
                elif key == ord('s'):
                    self.save_frame(frame)
                elif key == ord('h'):
                    self.show_help()
                elif key == ord('r'):
                    self.recognition_cache.clear()
                    print("🔄 Recognition cache cleared")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        except Exception as e:
            print(f"❌ Error during recognition: {e}")
        
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        cv.destroyAllWindows()
        print("✓ Resources cleaned up")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Real-time Face Recognition')
    
    parser.add_argument('--model-path', default='models/svm_face_model.pkl',
                       help='Path to trained SVM model')
    parser.add_argument('--encoder-path', default='models/label_encoder.pkl',
                       help='Path to label encoder')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera index to use (default: 0)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Recognition confidence threshold (default: 0.7)')
    parser.add_argument('--use-haar', action='store_true',
                       help='Use Haar cascade instead of MTCNN (faster but less accurate)')
    parser.add_argument('--frame-skip', type=int, default=2,
                       help='Process every Nth frame (default: 2, higher = faster)')
    
    args = parser.parse_args()
    
    # Check if model files exist
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print("\nTo train a model first, run:")
        print("python main.py train dataset")
        print("\nOr specify the correct model path with --model-path")
        sys.exit(1)
    
    if not os.path.exists(args.encoder_path):
        print(f"❌ Encoder file not found: {args.encoder_path}")
        print("\nMake sure you have both model and encoder files from training")
        sys.exit(1)
    
    try:
        # Initialize recognizer
        recognizer = RealTimeFaceRecognizer(
            model_path=args.model_path,
            encoder_path=args.encoder_path,
            confidence_threshold=args.confidence_threshold
        )
        
        # Run recognition
        recognizer.run(
            camera_index=args.camera_index,
            use_mtcnn=not args.use_haar,
            frame_skip=args.frame_skip
        )
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()