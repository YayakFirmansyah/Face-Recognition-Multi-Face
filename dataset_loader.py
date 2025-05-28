# dataset_loader.py - Dataset Loading Module
import os
import cv2
from collections import Counter

class DatasetLoader:
    def __init__(self, detector, extractor):
        """Initialize with detector and feature extractor"""
        self.detector = detector
        self.extractor = extractor
        
        # Supported image formats
        self.image_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.jfif', '.webp')
    
    def load_from_folder(self, dataset_path):
        """
        Load dataset from folder structure: dataset/person_name/images
        Returns: embeddings, labels, names, stats
        """
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return None, None, None, None
        
        embeddings = []
        labels = []
        names = []
        stats = {'persons': 0, 'total_images': 0, 'processed_images': 0}
        
        print(f"📁 Loading dataset from: {dataset_path}")
        
        # Process each person folder
        for person_folder in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_folder)
            
            if not os.path.isdir(person_path):
                continue
            
            person_embeddings = self._process_person_folder(person_path, person_folder)
            
            if person_embeddings:
                embeddings.extend(person_embeddings)
                labels.extend([person_folder] * len(person_embeddings))
                names.extend([person_folder] * len(person_embeddings))
                
                stats['persons'] += 1
                stats['processed_images'] += len(person_embeddings)
                
                print(f"  ✅ {person_folder}: {len(person_embeddings)} faces")
            else:
                print(f"  ❌ {person_folder}: No valid faces")
        
        # Final statistics
        stats['success_rate'] = (stats['processed_images'] / max(stats['total_images'], 1)) * 100
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Persons: {stats['persons']}")
        print(f"  Processed: {stats['processed_images']}/{stats['total_images']} images")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        
        if not embeddings:
            print("❌ No valid data loaded!")
            return None, None, None, stats
        
        return embeddings, labels, names, stats
    
    def _process_person_folder(self, person_path, person_name):
        """Process all images in a person's folder"""
        embeddings = []
        image_count = 0
        
        for filename in os.listdir(person_path):
            if not filename.lower().endswith(self.image_formats):
                continue
            
            image_count += 1
            image_path = os.path.join(person_path, filename)
            
            try:
                # Load and process image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face
                face, confidence = self.detector.detect_and_align(image_rgb)
                
                if face is not None:
                    # Extract features
                    embedding = self.extractor.extract(face)
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                
            except Exception as e:
                print(f"    Error processing {filename}: {e}")
        
        # Update total image count
        from dataset_loader import DatasetLoader
        if hasattr(DatasetLoader, '_total_images'):
            DatasetLoader._total_images += image_count
        else:
            DatasetLoader._total_images = image_count
        
        return embeddings
    
    def validate_structure(self, dataset_path):
        """Validate dataset folder structure"""
        if not os.path.exists(dataset_path):
            return False, "Path does not exist"
        
        persons = []
        total_images = 0
        
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            
            if os.path.isdir(item_path):
                images = [f for f in os.listdir(item_path) 
                         if f.lower().endswith(self.image_formats)]
                
                if images:
                    persons.append((item, len(images)))
                    total_images += len(images)
        
        if not persons:
            return False, "No valid person folders found"
        
        return True, {
            'persons': persons,
            'total_persons': len(persons),
            'total_images': total_images
        }
    
    def show_dataset_info(self, labels):
        """Display dataset information"""
        if not labels:
            print("❌ No dataset loaded")
            return
        
        label_counts = Counter(labels)
        total_images = len(labels)
        total_persons = len(label_counts)
        avg_per_person = total_images / total_persons
        
        print(f"\n📈 Dataset Information:")
        print(f"  Total images: {total_images}")
        print(f"  Total persons: {total_persons}")
        print(f"  Average per person: {avg_per_person:.1f}")
        print(f"\n  Distribution:")
        
        for name, count in sorted(label_counts.items()):
            print(f"    {name}: {count} images")