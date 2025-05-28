# face_classifier.py - SVM Classification Module
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import random

class FaceClassifier:
    def __init__(self):
        """Initialize SVM classifier"""
        self.model = None
        self.label_encoder = LabelEncoder()
        self.threshold = 0.5
        
        # Optimized SVM parameters
        self.param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma': [0.001, 0.01, 0.1, 1, 10],
            'kernel': ['rbf', 'linear', 'poly']
        }
    
    def augment_data(self, X, y):
        """Data augmentation with noise injection"""
        X_aug, y_aug = [], []
        
        for embedding, label in zip(X, y):
            # Original data
            X_aug.append(embedding)
            y_aug.append(label)
            
            # Add 3 variations with different noise levels
            for noise_level in [0.01, 0.02, 0.03]:
                noise = np.random.normal(0, noise_level, embedding.shape)
                aug_embedding = embedding + noise
                aug_embedding = aug_embedding / np.linalg.norm(aug_embedding)
                
                X_aug.append(aug_embedding)
                y_aug.append(label)
        
        return np.array(X_aug), np.array(y_aug)
    
    def balance_data(self, X, y):
        """Balance dataset using oversampling"""
        # Group data by label
        label_data = {}
        for embedding, label in zip(X, y):
            if label not in label_data:
                label_data[label] = []
            label_data[label].append(embedding)
        
        # Find max class size
        max_size = max(len(embeddings) for embeddings in label_data.values())
        
        X_balanced, y_balanced = [], []
        
        for label, embeddings in label_data.items():
            # Add original data
            for embedding in embeddings:
                X_balanced.append(embedding)
                y_balanced.append(label)
            
            # Oversample to match max_size
            while len([l for l in y_balanced if l == label]) < max_size:
                base_embedding = random.choice(embeddings)
                noise = np.random.normal(0, 0.02, base_embedding.shape)
                new_embedding = base_embedding + noise
                new_embedding = new_embedding / np.linalg.norm(new_embedding)
                
                X_balanced.append(new_embedding)
                y_balanced.append(label)
        
        return np.array(X_balanced), np.array(y_balanced)
    
    def train(self, X, y, use_augmentation=True, use_balancing=True, cv_folds=5):
        """
        Train SVM classifier with optimization
        """
        print(f"Training with {len(X)} samples, {len(set(y))} classes")
        
        # Data preprocessing
        if use_augmentation:
            X, y = self.augment_data(X, y)
            print(f"After augmentation: {len(X)} samples")
        
        if use_balancing:
            X, y = self.balance_data(X, y)
            print(f"After balancing: {len(X)} samples")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Grid search with cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            self.param_grid,
            cv=kfold,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y_encoded)
        
        self.model = grid_search.best_estimator_
        
        # Training results
        train_accuracy = grid_search.best_score_
        print(f"Best CV accuracy: {train_accuracy:.3f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        return train_accuracy
    
    def predict(self, embedding):
        """Predict face identity"""
        if self.model is None:
            return "No Model", 0.0
        
        embedding_reshaped = embedding.reshape(1, -1)
        proba = self.model.predict_proba(embedding_reshaped)[0]
        
        predicted_class = np.argmax(proba)
        confidence = proba[predicted_class]
        
        if confidence < self.threshold:
            return "Unknown", confidence
        
        predicted_name = self.label_encoder.inverse_transform([predicted_class])[0]
        return predicted_name, confidence
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        if self.model is None:
            print("No trained model!")
            return
        
        y_encoded = self.label_encoder.transform(y)
        y_pred = self.model.predict(X)
        
        accuracy = accuracy_score(y_encoded, y_pred)
        print(f"Test Accuracy: {accuracy:.3f}")
        
        # Class distribution
        class_counts = Counter(y)
        print(f"Class distribution: {dict(class_counts)}")
        
        return accuracy