# -*- coding: utf-8 -*-
"""
Face Recognition System - Main Application
Orchestrates the complete face recognition pipeline using modular components
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Import our custom modules
from preprocessing import FacePreprocessor, DataAugmentor, FaceVisualizer, validate_dataset_structure
from facenet_embeddings import FaceNetEmbedder, EmbeddingAnalyzer, EmbeddingQualityChecker
from svm_classifier import SVMFaceClassifier, EnsembleFaceClassifier, ModelComparison

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FaceRecognitionPipeline:
    """Complete face recognition pipeline orchestrator"""
    
    def __init__(self, dataset_path, output_dir='models', config=None):
        """
        Initialize the face recognition pipeline
        
        Args:
            dataset_path (str): Path to dataset directory
            output_dir (str): Directory to save models and results
            config (dict): Configuration parameters
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.config = config or self.get_default_config()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = None
        self.embedder = None
        self.classifier = None
        self.augmentor = None
        
        # Data storage
        self.faces = None
        self.labels = None
        self.embeddings = None
        self.training_results = {}
        
        print(f"Initialized Face Recognition Pipeline")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_dir}")
    
    def get_default_config(self):
        """Get default configuration parameters"""
        return {
            'preprocessing': {
                'target_size': (160, 160),
                'confidence_threshold': 0.9
            },
            'augmentation': {
                'use_augmentation': True,
                'augmentation_factor': 2
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.1,
                'search_type': 'comprehensive',
                'cv_folds': 5
            },
            'evaluation': {
                'confidence_threshold': 0.7,
                'plot_results': True
            }
        }
    
    def validate_setup(self):
        """Validate dataset and setup"""
        print("Validating setup...")
        
        # Validate dataset structure
        validation_result = validate_dataset_structure(self.dataset_path)
        
        if not validation_result['valid']:
            raise ValueError(f"Dataset validation failed: {validation_result['error']}")
        
        print("✓ Dataset validation passed")
        print(f"  - People: {validation_result['num_people']}")
        print(f"  - Total images: {validation_result['total_images']}")
        print(f"  - Avg images/person: {validation_result['avg_images_per_person']:.1f}")
        
        # Check minimum requirements
        if validation_result['num_people'] < 2:
            raise ValueError("Need at least 2 people for face recognition")
        
        if validation_result['avg_images_per_person'] < 3:
            print("⚠️  Warning: Less than 3 images per person on average")
            print("   Consider adding more images for better performance")
        
        return validation_result
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        print("Initializing components...")
        
        # Preprocessor
        self.preprocessor = FacePreprocessor(
            target_size=self.config['preprocessing']['target_size'],
            confidence_threshold=self.config['preprocessing']['confidence_threshold']
        )
        
        # Augmentor
        self.augmentor = DataAugmentor()
        
        # Embedder
        self.embedder = FaceNetEmbedder()
        
        # Classifier
        self.classifier = SVMFaceClassifier()
        
        print("✓ All components initialized")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("\n" + "="*50)
        print("STEP 1: Data Loading and Preprocessing")
        print("="*50)
        
        # Load faces from dataset
        faces, labels = self.preprocessor.load_dataset_from_structure(self.dataset_path)
        
        if len(faces) == 0:
            raise ValueError("No valid faces found in dataset!")
        
        print(f"Loaded {len(faces)} faces for {len(np.unique(labels))} people")
        
        # Data augmentation if enabled
        if self.config['augmentation']['use_augmentation']:
            print("Applying data augmentation...")
            faces, labels = self.augmentor.augment_dataset(
                faces, labels, 
                augmentation_factor=self.config['augmentation']['augmentation_factor']
            )
            print(f"After augmentation: {len(faces)} faces")
        
        self.faces = faces
        self.labels = labels
        
        # Show sample images
        if self.config['evaluation']['plot_results']:
            visualizer = FaceVisualizer()
            visualizer.plot_faces_grid(faces[:9], labels[:9])
        
        return len(faces), len(np.unique(labels))
    
    def generate_embeddings(self):
        """Generate face embeddings using FaceNet"""
        print("\n" + "="*50)
        print("STEP 2: Embedding Generation")
        print("="*50)
        
        # Generate embeddings
        embeddings, valid_labels, failed_indices = self.embedder.generate_embeddings_for_dataset(
            self.faces, self.labels, batch_size=32
        )
        
        if len(embeddings) == 0:
            raise ValueError("No embeddings generated!")
        
        print(f"Generated {len(embeddings)} embeddings")
        if failed_indices:
            print(f"Failed to process {len(failed_indices)} faces")
        
        self.embeddings = embeddings
        self.labels = valid_labels  # Update labels to match valid embeddings
        
        # Save embeddings
        embeddings_path = os.path.join(self.output_dir, 'face_embeddings.npz')
        self.embedder.save_embeddings(embeddings, valid_labels, embeddings_path)
        
        # Quality check
        quality_checker = EmbeddingQualityChecker()
        quality_results = quality_checker.check_embedding_quality(embeddings, valid_labels)
        
        print("\nEmbedding Quality Assessment:")
        if quality_results['issues']:
            for issue in quality_results['issues']:
                print(f"  ⚠️  {issue}")
        else:
            print("  ✓ No quality issues detected")
        
        # Get improvement suggestions
        suggestions = quality_checker.suggest_improvements(quality_results)
        if suggestions and suggestions[0] != "Embedding quality looks good!":
            print("\nSuggestions for improvement:")
            for suggestion in suggestions:
                print(f"  💡 {suggestion}")
        
        return len(embeddings)
    
    def train_classifier(self):
        """Train the SVM classifier"""
        print("\n" + "="*50)
        print("STEP 3: Classifier Training")
        print("="*50)
        
        # Prepare data splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.classifier.prepare_data(
            self.embeddings, self.labels,
            test_size=self.config['training']['test_size'],
            validation_size=self.config['training']['validation_size']
        )
        
        # Train with grid search
        training_results = self.classifier.train_with_grid_search(
            X_train, y_train,
            search_type=self.config['training']['search_type'],
            cv_folds=self.config['training']['cv_folds']
        )
        
        print(f"✓ Training completed")
        print(f"Best CV Score: {training_results['best_score']:.4f}")
        
        # Evaluate on test set
        eval_results = self.classifier.evaluate_model(X_test, y_test)
        
        print(f"Test Accuracy: {eval_results['accuracy']:.4f}")
        print(f"Test Precision: {eval_results['precision']:.4f}")
        print(f"Test Recall: {eval_results['recall']:.4f}")
        print(f"Test F1-Score: {eval_results['f1_score']:.4f}")
        
        # Store results
        self.training_results = {
            'training': training_results,
            'evaluation': eval_results,
            'data_splits': {
                'train_size': len(X_train),
                'val_size': len(X_val) if X_val is not None else 0,
                'test_size': len(X_test)
            }
        }
        
        # Plot results if enabled
        if self.config['evaluation']['plot_results']:
            self.classifier.plot_confusion_matrix(eval_results['confusion_matrix'])
            self.classifier.plot_hyperparameter_results()
            
            # Confidence analysis
            confidence_results = self.classifier.analyze_prediction_confidence(
                X_test, y_test, 
                threshold=self.config['evaluation']['confidence_threshold']
            )
        
        return eval_results['accuracy']
    
    def save_models(self):
        """Save all trained models and components"""
        print("\n" + "="*50)
        print("STEP 4: Saving Models")
        print("="*50)
        
        # Save classifier
        model_path = os.path.join(self.output_dir, 'svm_face_model.pkl')
        encoder_path = os.path.join(self.output_dir, 'label_encoder.pkl')
        self.classifier.save_model(model_path, encoder_path)
        
        # Save training results
        import pickle
        results_path = os.path.join(self.output_dir, 'training_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.training_results, f)
        
        print(f"✓ Models saved to {self.output_dir}")
        return model_path, encoder_path
    
    def run_complete_pipeline(self):
        """Run the complete face recognition pipeline"""
        start_time = datetime.now()
        
        print("Starting Face Recognition Pipeline")
        print("="*60)
        
        try:
            # Validate setup
            validation_result = self.validate_setup()
            
            # Initialize components
            self.initialize_components()
            
            # Run pipeline steps
            num_faces, num_people = self.load_and_preprocess_data()
            num_embeddings = self.generate_embeddings()
            final_accuracy = self.train_classifier()
            model_path, encoder_path = self.save_models()
            
            # Final summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Duration: {duration}")
            print(f"Final Accuracy: {final_accuracy:.4f}")
            print(f"People in dataset: {num_people}")
            print(f"Total faces processed: {num_faces}")
            print(f"Embeddings generated: {num_embeddings}")
            print(f"Model saved to: {model_path}")
            
            return {
                'success': True,
                'accuracy': final_accuracy,
                'duration': duration,
                'model_path': model_path,
                'encoder_path': encoder_path,
                'stats': {
                    'num_people': num_people,
                    'num_faces': num_faces,
                    'num_embeddings': num_embeddings
                }
            }
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'duration': datetime.now() - start_time
            }


class FaceRecognitionPredictor:
    """Class for making predictions with trained models"""
    
    def __init__(self, model_path, encoder_path, embeddings_path=None):
        """
        Initialize predictor with trained models
        
        Args:
            model_path (str): Path to trained SVM model
            encoder_path (str): Path to label encoder
            embeddings_path (str): Path to embeddings file (optional)
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        
        # Initialize components
        self.preprocessor = FacePreprocessor()
        self.embedder = FaceNetEmbedder()
        self.classifier = SVMFaceClassifier()
        
        # Load trained models
        self.load_models()
        
        print("✓ Predictor initialized and models loaded")
    
    def load_models(self):
        """Load all trained models"""
        self.classifier.load_model(self.model_path, self.encoder_path)
    
    def predict_image(self, image_path, confidence_threshold=0.7):
        """
        Predict face in image
        
        Args:
            image_path (str): Path to image file
            confidence_threshold (float): Minimum confidence for prediction
            
        Returns:
            dict: Prediction results
        """
        try:
            # Extract face from image
            face = self.preprocessor.detect_and_extract_face(image_path)
            if face is None:
                return {
                    'success': False,
                    'error': 'No face detected in image'
                }
            
            # Generate embedding
            embedding = self.embedder.get_single_embedding(face)
            if embedding is None:
                return {
                    'success': False,
                    'error': 'Failed to generate embedding'
                }
            
            # Make prediction
            predicted_label, confidence, is_confident = self.classifier.predict_single(
                embedding, confidence_threshold
            )
            
            return {
                'success': True,
                'predicted_person': predicted_label,
                'confidence': confidence,
                'is_confident': is_confident,
                'confidence_threshold': confidence_threshold
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_batch(self, image_paths, confidence_threshold=0.7):
        """
        Predict multiple images
        
        Args:
            image_paths (list): List of image paths
            confidence_threshold (float): Minimum confidence for prediction
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        print(f"Predicting {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            result = self.predict_image(image_path, confidence_threshold)
            result['image_path'] = image_path
            results.append(result)
        
        return results


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Face Recognition System')
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train face recognition model')
    train_parser.add_argument('dataset_path', help='Path to dataset directory')
    train_parser.add_argument('--output-dir', default='models', help='Output directory for models')
    train_parser.add_argument('--search-type', choices=['quick', 'comprehensive', 'fine_tune'], 
                             default='comprehensive', help='Hyperparameter search type')
    train_parser.add_argument('--no-augmentation', action='store_true', 
                             help='Disable data augmentation')
    train_parser.add_argument('--augmentation-factor', type=int, default=2,
                             help='Data augmentation factor')
    train_parser.add_argument('--confidence-threshold', type=float, default=0.9,
                             help='Face detection confidence threshold')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                             help='Test set size (0.0-1.0)')
    train_parser.add_argument('--no-plots', action='store_true',
                             help='Disable result plotting')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict faces in images')
    predict_parser.add_argument('model_path', help='Path to trained model')
    predict_parser.add_argument('encoder_path', help='Path to label encoder')
    predict_parser.add_argument('image_paths', nargs='+', help='Paths to images to predict')
    predict_parser.add_argument('--confidence-threshold', type=float, default=0.7,
                               help='Prediction confidence threshold')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('model_path', help='Path to trained model')
    eval_parser.add_argument('encoder_path', help='Path to label encoder')
    eval_parser.add_argument('test_dataset', help='Path to test dataset')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo with sample data')
    demo_parser.add_argument('--dataset-path', default='dataset', 
                            help='Path to dataset directory')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_training(args)
    elif args.command == 'predict':
        run_prediction(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'demo':
        run_demo(args)
    else:
        parser.print_help()


def run_training(args):
    """Run training pipeline"""
    print("Face Recognition Training")
    print("="*40)
    
    # Create configuration
    config = {
        'preprocessing': {
            'target_size': (160, 160),
            'confidence_threshold': args.confidence_threshold
        },
        'augmentation': {
            'use_augmentation': not args.no_augmentation,
            'augmentation_factor': args.augmentation_factor
        },
        'training': {
            'test_size': args.test_size,
            'validation_size': 0.1,
            'search_type': args.search_type,
            'cv_folds': 5
        },
        'evaluation': {
            'confidence_threshold': 0.7,
            'plot_results': not args.no_plots
        }
    }
    
    # Initialize and run pipeline
    pipeline = FaceRecognitionPipeline(args.dataset_path, args.output_dir, config)
    result = pipeline.run_complete_pipeline()
    
    if result['success']:
        print(f"\n🎉 Training completed successfully!")
        print(f"Final accuracy: {result['accuracy']:.1%}")
        print(f"Duration: {result['duration']}")
        print(f"Model saved to: {result['model_path']}")
    else:
        print(f"\n❌ Training failed: {result['error']}")
        sys.exit(1)


def run_prediction(args):
    """Run prediction on images"""
    print("Face Recognition Prediction")
    print("="*40)
    
    # Initialize predictor
    try:
        predictor = FaceRecognitionPredictor(args.model_path, args.encoder_path)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        sys.exit(1)
    
    # Make predictions
    results = predictor.predict_batch(args.image_paths, args.confidence_threshold)
    
    # Display results
    print("\nPrediction Results:")
    print("-" * 60)
    
    for result in results:
        image_name = os.path.basename(result['image_path'])
        
        if result['success']:
            person = result['predicted_person']
            confidence = result['confidence']
            confident = "✓" if result['is_confident'] else "⚠️"
            
            print(f"{image_name:30} | {person:15} | {confidence:.3f} {confident}")
        else:
            print(f"{image_name:30} | ERROR: {result['error']}")


def run_evaluation(args):
    """Run model evaluation"""
    print("Model Evaluation")
    print("="*40)
    
    # This would implement comprehensive model evaluation
    # Including metrics on test dataset, confusion matrices, etc.
    print("Evaluation functionality to be implemented")
    # TODO: Implement evaluation functionality


def run_demo(args):
    """Run demo with sample data"""
    print("Face Recognition Demo")
    print("="*40)
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset path does not exist: {args.dataset_path}")
        print("\nTo run the demo, create a dataset folder with this structure:")
        print("dataset/")
        print("├── person1/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("├── person2/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("└── ...")
        return
    
    # Run with demo configuration (faster settings)
    config = {
        'preprocessing': {
            'target_size': (160, 160),
            'confidence_threshold': 0.9
        },
        'augmentation': {
            'use_augmentation': True,
            'augmentation_factor': 1  # Less augmentation for demo
        },
        'training': {
            'test_size': 0.2,
            'validation_size': 0.1,
            'search_type': 'quick',  # Faster search for demo
            'cv_folds': 3
        },
        'evaluation': {
            'confidence_threshold': 0.7,
            'plot_results': True
        }
    }
    
    # Initialize and run pipeline
    pipeline = FaceRecognitionPipeline(args.dataset_path, 'demo_models', config)
    result = pipeline.run_complete_pipeline()
    
    if result['success']:
        print(f"\n🎉 Demo completed successfully!")
        print(f"Accuracy: {result['accuracy']:.1%}")
        
        # Test prediction on a sample image
        sample_images = []
        for person_dir in os.listdir(args.dataset_path):
            person_path = os.path.join(args.dataset_path, person_dir)
            if os.path.isdir(person_path):
                images = [f for f in os.listdir(person_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    sample_images.append(os.path.join(person_path, images[0]))
                if len(sample_images) >= 3:  # Limit to 3 samples
                    break
        
        if sample_images:
            print(f"\nTesting predictions on {len(sample_images)} sample images:")
            predictor = FaceRecognitionPredictor(
                result['model_path'], 
                result['encoder_path']
            )
            
            for img_path in sample_images:
                pred_result = predictor.predict_image(img_path)
                if pred_result['success']:
                    print(f"  {os.path.basename(img_path)}: {pred_result['predicted_person']} "
                          f"({pred_result['confidence']:.3f})")
    else:
        print(f"\n❌ Demo failed: {result['error']}")


def create_sample_script():
    """Create a sample usage script"""
    sample_script = '''#!/usr/bin/env python3
"""
Sample usage script for Face Recognition System
"""

from main import FaceRecognitionPipeline, FaceRecognitionPredictor

def train_model():
    """Train a face recognition model"""
    
    # Configuration
    config = {
        'preprocessing': {
            'target_size': (160, 160),
            'confidence_threshold': 0.9
        },
        'augmentation': {
            'use_augmentation': True,
            'augmentation_factor': 2
        },
        'training': {
            'test_size': 0.2,
            'validation_size': 0.1,
            'search_type': 'comprehensive',
            'cv_folds': 5
        },
        'evaluation': {
            'confidence_threshold': 0.7,
            'plot_results': True
        }
    }
    
    # Train model
    pipeline = FaceRecognitionPipeline('dataset', 'models', config)
    result = pipeline.run_complete_pipeline()
    
    if result['success']:
        print(f"Training successful! Accuracy: {result['accuracy']:.1%}")
        return result['model_path'], result['encoder_path']
    else:
        print(f"Training failed: {result['error']}")
        return None, None

def predict_faces():
    """Make predictions on new images"""
    
    model_path = 'models/svm_face_model.pkl'
    encoder_path = 'models/label_encoder.pkl'
    
    # Initialize predictor
    predictor = FaceRecognitionPredictor(model_path, encoder_path)
    
    # Predict single image
    result = predictor.predict_image('test_image.jpg')
    
    if result['success']:
        print(f"Predicted: {result['predicted_person']}")
        print(f"Confidence: {result['confidence']:.3f}")
    else:
        print(f"Prediction failed: {result['error']}")

if __name__ == "__main__":
    # Train model
    model_path, encoder_path = train_model()
    
    # Make predictions if training was successful
    if model_path:
        predict_faces()
'''
    
    with open('sample_usage.py', 'w') as f:
        f.write(sample_script)
    
    print("Created sample_usage.py - check this file for usage examples")


if __name__ == "__main__":
    # If no arguments provided, show help and create sample script
    if len(sys.argv) == 1:
        print("Face Recognition System")
        print("="*40)
        print("\nAvailable commands:")
        print("  train     - Train a new face recognition model")
        print("  predict   - Predict faces in images")
        print("  evaluate  - Evaluate trained model")
        print("  demo      - Run demo with sample data")
        print("\nFor detailed help on each command:")
        print("  python main.py <command> --help")
        print("\nExample usage:")
        print("  python main.py train dataset")
        print("  python main.py predict models/svm_face_model.pkl models/label_encoder.pkl test1.jpg test2.jpg")
        print("  python main.py demo")
        
        # Create sample script
        create_sample_script()
    else:
        main()