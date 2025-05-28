# -*- coding: utf-8 -*-
"""
Face Preprocessing Module
Handles face detection, extraction, and preprocessing
"""

import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class FacePreprocessor:
    def __init__(self, target_size=(160, 160), confidence_threshold=0.9):
        """
        Initialize Face Preprocessor
        
        Args:
            target_size (tuple): Target size for face images (width, height)
            confidence_threshold (float): Minimum confidence for face detection
        """
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.detector = MTCNN()
        
    def detect_and_extract_face(self, image_path):
        """
        Detect and extract face from image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            numpy.ndarray or None: Preprocessed face image or None if no face detected
        """
        try:
            # Read image
            img = cv.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return None
                
            # Convert BGR to RGB
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.detector.detect_faces(img)
            
            if not results:
                print(f"No face detected in: {image_path}")
                return None
            
            # Select face with highest confidence
            best_face = max(results, key=lambda x: x['confidence'])
            
            if best_face['confidence'] < self.confidence_threshold:
                print(f"Low confidence face in: {image_path} ({best_face['confidence']:.2f})")
                return None
                
            # Extract face coordinates
            x, y, w, h = best_face['box']
            x, y = abs(x), abs(y)
            
            # Add padding for better context
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            # Extract face region
            face = img[y:y+h, x:x+w]
            
            # Resize to target size
            face_resized = cv.resize(face, self.target_size)
            
            # Normalize pixel values to [0, 1]
            face_normalized = face_resized.astype('float32') / 255.0
            
            return face_normalized
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def detect_face_with_landmarks(self, image_path):
        """
        Detect face with additional landmark information
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Dictionary containing face image, landmarks, and metadata
        """
        try:
            img = cv.imread(image_path)
            if img is None:
                return None
                
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = self.detector.detect_faces(img)
            
            if not results:
                return None
            
            best_face = max(results, key=lambda x: x['confidence'])
            
            if best_face['confidence'] < self.confidence_threshold:
                return None
            
            # Extract face
            face_img = self.detect_and_extract_face(image_path)
            if face_img is None:
                return None
            
            return {
                'face_image': face_img,
                'landmarks': best_face.get('keypoints', {}),
                'confidence': best_face['confidence'],
                'bounding_box': best_face['box']
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def load_faces_from_directory(self, directory_path):
        """
        Load all faces from a directory
        
        Args:
            directory_path (str): Path to directory containing face images
            
        Returns:
            list: List of preprocessed face images
        """
        faces = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        if not os.path.exists(directory_path):
            print(f"Directory does not exist: {directory_path}")
            return faces
        
        print(f"Loading faces from: {directory_path}")
        
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                filepath = os.path.join(directory_path, filename)
                face = self.detect_and_extract_face(filepath)
                
                if face is not None:
                    faces.append(face)
                    
        print(f"Successfully loaded {len(faces)} faces from {directory_path}")
        return faces
    
    def load_dataset_from_structure(self, dataset_directory):
        """
        Load dataset from directory structure (person_name/images)
        
        Args:
            dataset_directory (str): Root directory containing person subdirectories
            
        Returns:
            tuple: (faces_array, labels_array)
        """
        all_faces = []
        all_labels = []
        
        if not os.path.exists(dataset_directory):
            print(f"Dataset directory does not exist: {dataset_directory}")
            return np.array([]), np.array([])
        
        print(f"Loading dataset from: {dataset_directory}")
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(dataset_directory) 
                      if os.path.isdir(os.path.join(dataset_directory, d))]
        
        if not person_dirs:
            print("No person directories found in dataset!")
            return np.array([]), np.array([])
        
        for person_name in person_dirs:
            person_path = os.path.join(dataset_directory, person_name)
            faces = self.load_faces_from_directory(person_path)
            
            if faces:
                labels = [person_name] * len(faces)
                all_faces.extend(faces)
                all_labels.extend(labels)
                print(f"Loaded {len(faces)} images for {person_name}")
            else:
                print(f"No valid faces found for {person_name}")
        
        print(f"Total dataset: {len(all_faces)} images for {len(person_dirs)} people")
        
        return np.array(all_faces), np.array(all_labels)


class DataAugmentor:
    def __init__(self):
        """Initialize Data Augmentor with predefined augmentation parameters"""
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
    
    def augment_dataset(self, faces, labels, augmentation_factor=3):
        """
        Augment dataset to increase size and variety
        
        Args:
            faces (numpy.ndarray): Original face images
            labels (numpy.ndarray): Corresponding labels
            augmentation_factor (int): Number of augmented images per original
            
        Returns:
            tuple: (augmented_faces, augmented_labels)
        """
        if len(faces) == 0:
            return faces, labels
            
        print(f"Augmenting dataset with factor {augmentation_factor}...")
        
        augmented_faces = []
        augmented_labels = []
        
        # Add original images first
        augmented_faces.extend(faces)
        augmented_labels.extend(labels)
        
        # Generate augmented images
        for i, (face, label) in enumerate(zip(faces, labels)):
            # Ensure face is in correct format for augmentation
            if face.max() <= 1.0:
                face_for_aug = face
            else:
                face_for_aug = face / 255.0
            
            # Expand dimensions for ImageDataGenerator
            face_expanded = np.expand_dims(face_for_aug, 0)
            
            # Generate augmented versions
            aug_iter = self.datagen.flow(face_expanded, batch_size=1)
            
            for _ in range(augmentation_factor):
                try:
                    aug_img = next(aug_iter)[0]
                    # Ensure values are in [0, 1] range
                    aug_img = np.clip(aug_img, 0, 1)
                    
                    augmented_faces.append(aug_img)
                    augmented_labels.append(label)
                    
                except StopIteration:
                    break
        
        result_faces = np.array(augmented_faces)
        result_labels = np.array(augmented_labels)
        
        print(f"Augmentation complete: {len(faces)} -> {len(result_faces)} images")
        
        return result_faces, result_labels
    
    def create_custom_augmentation(self, rotation_range=15, brightness_range=[0.8, 1.2], 
                                 zoom_range=0.1, horizontal_flip=True):
        """
        Create custom augmentation pipeline
        
        Args:
            rotation_range (int): Range for random rotations
            brightness_range (list): Range for brightness variation
            zoom_range (float): Range for random zoom
            horizontal_flip (bool): Whether to apply horizontal flip
            
        Returns:
            ImageDataGenerator: Custom augmentation generator
        """
        return ImageDataGenerator(
            rotation_range=rotation_range,
            brightness_range=brightness_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode='nearest'
        )


class FaceVisualizer:
    @staticmethod
    def plot_faces_grid(faces, labels, max_images=9, figsize=(15, 10)):
        """
        Plot faces in a grid layout
        
        Args:
            faces (numpy.ndarray): Array of face images
            labels (numpy.ndarray): Corresponding labels
            max_images (int): Maximum number of images to display
            figsize (tuple): Figure size for matplotlib
        """
        if len(faces) == 0:
            print("No faces to display!")
            return
        
        n_images = min(max_images, len(faces))
        cols = 3
        rows = (n_images + cols - 1) // cols
        
        plt.figure(figsize=figsize)
        
        for i in range(n_images):
            plt.subplot(rows, cols, i + 1)
            
            # Handle both normalized [0,1] and [0,255] images
            img = faces[i]
            if img.max() <= 1.0:
                plt.imshow(img)
            else:
                plt.imshow(img.astype(np.uint8))
            
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_face_detection_result(image_path, detector):
        """
        Visualize face detection results on an image
        
        Args:
            image_path (str): Path to image file
            detector (MTCNN): MTCNN detector instance
        """
        try:
            img = cv.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return
            
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = detector.detect_faces(img_rgb)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
            
            # Draw bounding boxes and landmarks
            for result in results:
                x, y, w, h = result['box']
                confidence = result['confidence']
                
                # Draw bounding box
                rect = plt.Rectangle((x, y), w, h, fill=False, 
                                   color='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                # Add confidence text
                plt.text(x, y-10, f'{confidence:.2f}', 
                        color='red', fontsize=12, fontweight='bold')
                
                # Draw landmarks if available
                if 'keypoints' in result:
                    keypoints = result['keypoints']
                    for key, point in keypoints.items():
                        plt.plot(point[0], point[1], 'ro', markersize=4)
            
            plt.title(f'Face Detection Results - {os.path.basename(image_path)}')
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing detection results: {str(e)}")


# Utility functions
def validate_dataset_structure(dataset_path):
    """
    Validate dataset directory structure
    
    Args:
        dataset_path (str): Path to dataset directory
        
    Returns:
        dict: Validation results and statistics
    """
    if not os.path.exists(dataset_path):
        return {
            'valid': False,
            'error': f"Dataset path does not exist: {dataset_path}"
        }
    
    person_dirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not person_dirs:
        return {
            'valid': False,
            'error': "No person directories found in dataset"
        }
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    stats = {}
    total_images = 0
    
    for person in person_dirs:
        person_path = os.path.join(dataset_path, person)
        images = [f for f in os.listdir(person_path) 
                 if any(f.lower().endswith(ext) for ext in valid_extensions)]
        stats[person] = len(images)
        total_images += len(images)
    
    return {
        'valid': True,
        'num_people': len(person_dirs),
        'total_images': total_images,
        'person_stats': stats,
        'avg_images_per_person': total_images / len(person_dirs) if person_dirs else 0
    }


if __name__ == "__main__":
    # Example usage
    print("Face Preprocessing Module")
    print("=" * 40)
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(target_size=(160, 160), confidence_threshold=0.9)
    
    # Example: Validate dataset structure
    dataset_path = "dataset"  # Change this to your dataset path
    validation_result = validate_dataset_structure(dataset_path)
    
    if validation_result['valid']:
        print(f"Dataset validation: PASSED")
        print(f"Number of people: {validation_result['num_people']}")
        print(f"Total images: {validation_result['total_images']}")
        print(f"Average images per person: {validation_result['avg_images_per_person']:.1f}")
        
        # Load dataset
        faces, labels = preprocessor.load_dataset_from_structure(dataset_path)
        
        if len(faces) > 0:
            # Initialize visualizer and augmentor
            visualizer = FaceVisualizer()
            augmentor = DataAugmentor()
            
            # Show original faces
            print("\nDisplaying sample faces...")
            visualizer.plot_faces_grid(faces[:9], labels[:9])
            
            # Augment data
            aug_faces, aug_labels = augmentor.augment_dataset(faces, labels, 
                                                            augmentation_factor=2)
            print(f"Dataset size after augmentation: {len(aug_faces)}")
            
        else:
            print("No valid faces found in dataset!")
    else:
        print(f"Dataset validation: FAILED")
        print(f"Error: {validation_result['error']}")