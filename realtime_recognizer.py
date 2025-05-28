# realtime_recognizer.py - Real-time Recognition Module
import cv2
import numpy as np

class RealtimeRecognizer:
    def __init__(self, detector, extractor, classifier):
        """Initialize with detector, extractor, and classifier"""
        self.detector = detector
        self.extractor = extractor
        self.classifier = classifier
        
        # Performance settings
        self.process_every_n_frames = 3
        self.frame_count = 0
    
    def start_recognition(self):
        """Start real-time face recognition"""
        print("🎥 Starting real-time face recognition...")
        print("📝 Controls: Press 'q' to quit, 's' to save screenshot")
        
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame for performance
            if self.frame_count % self.process_every_n_frames == 0:
                frame = self._process_frame(frame)
            
            # Display frame
            cv2.imshow('Face Recognition System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'screenshot_{self.frame_count}.jpg', frame)
                print(f"📸 Screenshot saved: screenshot_{self.frame_count}.jpg")
            
            self.frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Recognition stopped")
    
    def _process_frame(self, frame):
        """Process single frame for face recognition"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect multiple faces
        faces = self.detector.detect_multiple(frame_rgb, min_confidence=0.8)
        
        for face_info in faces:
            x, y, w, h = face_info['box']
            confidence = face_info['confidence']
            
            # Extract face region
            face_crop = frame_rgb[y:y+h, x:x+w]
            
            if face_crop.size > 0:
                # Resize for feature extraction
                face_resized = cv2.resize(face_crop, (160, 160))
                
                # Extract features and classify
                embedding = self.extractor.extract(face_resized)
                
                if embedding is not None:
                    name, pred_confidence = self.classifier.predict(embedding)
                    
                    # Draw bounding box and label
                    self._draw_face_info(frame, x, y, w, h, name, pred_confidence)
        
        return frame
    
    def _draw_face_info(self, frame, x, y, w, h, name, confidence):
        """Draw face bounding box and information"""
        # Choose color based on recognition
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            label = f"Unknown ({confidence:.2f})"
        else:
            color = (0, 255, 0)  # Green for known
            label = f"{name} ({confidence:.2f})"
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y-30), (x + label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def recognize_image(self, image_path):
        """Recognize faces in a single image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Cannot load image: {image_path}")
                return
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_multiple(image_rgb)
            
            print(f"🔍 Found {len(faces)} faces in image")
            
            for i, face_info in enumerate(faces):
                x, y, w, h = face_info['box']
                face_crop = image_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, (160, 160))
                
                embedding = self.extractor.extract(face_resized)
                if embedding is not None:
                    name, confidence = self.classifier.predict(embedding)
                    print(f"  Face {i+1}: {name} (confidence: {confidence:.3f})")
                    
                    # Draw on image
                    self._draw_face_info(image, x, y, w, h, name, confidence)
            
            # Display result
            cv2.imshow('Recognition Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")