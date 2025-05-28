# face_detector.py - MTCNN Face Detection Module
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        """Initialize MTCNN detector"""
        self.detector = MTCNN()
    
    def detect_and_align(self, image):
        """
        Detect face and return aligned face crop
        Returns: face_crop (160x160), confidence
        """
        try:
            # Detect faces
            results = self.detector.detect_faces(image)
            
            if not results:
                return None, 0.0
            
            # Get best detection
            best_face = max(results, key=lambda x: x['confidence'])
            
            if best_face['confidence'] < 0.8:
                return None, best_face['confidence']
            
            # Extract and crop face
            x, y, w, h = best_face['box']
            
            # Add padding and ensure bounds
            padding = 30
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None, 0.0
            
            # Resize to standard size for FaceNet
            face_resized = cv2.resize(face_crop, (160, 160))
            
            # Enhance contrast
            face_enhanced = cv2.convertScaleAbs(face_resized, alpha=1.2, beta=10)
            
            return face_enhanced, best_face['confidence']
            
        except Exception as e:
            print(f"Detection error: {e}")
            return None, 0.0
    
    def detect_multiple(self, image, min_confidence=0.8):
        """Detect multiple faces in image"""
        try:
            results = self.detector.detect_faces(image)
            faces = []
            
            for face in results:
                if face['confidence'] >= min_confidence:
                    x, y, w, h = face['box']
                    faces.append({
                        'box': (x, y, w, h),
                        'confidence': face['confidence'],
                        'keypoints': face['keypoints']
                    })
            
            return faces
            
        except Exception:
            return []