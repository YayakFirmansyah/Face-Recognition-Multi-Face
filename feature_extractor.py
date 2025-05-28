# feature_extractor.py - FaceNet Feature Extraction Module
import numpy as np
import cv2
from keras_facenet import FaceNet

class FeatureExtractor:
    def __init__(self):
        """Initialize FaceNet model and MTCNN detector"""
        self.facenet = FaceNet()
        self.detector = self.facenet.mtcnn  # MTCNN detector

    def detect_faces(self, image, visualize=False):
        """
        Detect faces and return bounding boxes and cropped faces.
        Optionally visualize bounding boxes.
        """
        results = self.detector.detect(image)
        faces = []
        boxes = []
        if results is not None:
            for res in results:
                x1, y1, x2, y2 = map(int, res['box'])
                face = image[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face_resized = cv2.resize(face, (160, 160))
                faces.append(face_resized)
                boxes.append((x1, y1, x2, y2))
                if visualize:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            if visualize:
                cv2.imshow("Detected Faces", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return faces, boxes

    def extract(self, face_image):
        """
        Extract face embedding (128 or 512 dim, depending on model)
        Input: face_image (160x160)
        Output: normalized embedding vector
        """
        try:
            face_normalized = face_image.astype('float32') / 255.0
            face_normalized = (face_normalized - 0.5) * 2.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            embedding = self.facenet.embeddings(face_expanded)
            embedding_normalized = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding_normalized[0]
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def extract_batch(self, face_images):
        """Extract features from multiple faces"""
        embeddings = []
        for face in face_images:
            embedding = self.extract(face)
            if embedding is not None:
                embeddings.append(embedding)
        return np.array(embeddings) if embeddings else None

# Pengetesan
if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/yayak/atas.jpg"
    image = cv2.imread(img_path)
    fe = FeatureExtractor()
    faces, boxes = fe.detect_faces(image.copy(), visualize=True)
    if faces:
        print(f"Detected {len(faces)} face(s).")
        embeddings = fe.extract_batch(faces)
        print(f"Embedding shape: {embeddings.shape}")
        print(f"First embedding (truncated): {embeddings[0][:10]} ...")
    else:
        print("No faces detected.")