# -*- coding: utf-8 -*-
"""
SVM Classification Module
Handles SVM model training, evaluation, and prediction for face recognition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, 
    cross_val_score, validation_curve
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')


class SVMFaceClassifier:
    def __init__(self, random_state=42):
        """
        Initialize SVM Face Classifier
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = {}
        self.best_params = None
        
    def prepare_data(self, embeddings, labels, test_size=0.2, validation_size=0.1):
        """
        Prepare data for training with train/validation/test splits
        
        Args:
            embeddings (numpy.ndarray): Face embeddings
            labels (numpy.ndarray): Corresponding labels
            test_size (float): Proportion for test set
            validation_size (float): Proportion for validation set (from remaining data)
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("Preparing data for training...")
        
        # Encode labels
        y_encoded = self.encoder.fit_transform(labels)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            embeddings, y_encoded, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=y_encoded
        )
        
        # Second split: separate train and validation sets
        if validation_size > 0:
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.random_state,
                stratify=y_temp
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test)}")
        print(f"Number of classes: {len(self.encoder.classes_)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_hyperparameter_grid(self, search_type='comprehensive'):
        """
        Get hyperparameter grid for different search types
        
        Args:
            search_type (str): Type of search ('quick', 'comprehensive', 'fine_tune')
            
        Returns:
            dict: Parameter grid for GridSearchCV
        """
        if search_type == 'quick':
            return {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        
        elif search_type == 'comprehensive':
            return {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        
        elif search_type == 'fine_tune':
            return {
                'C': [0.5, 1, 2, 5, 10, 20, 50],
                'gamma': ['scale', 0.005, 0.01, 0.05, 0.1, 0.5],
                'kernel': ['rbf']
            }
        
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
    
    def train_with_grid_search(self, X_train, y_train, search_type='comprehensive', 
                              cv_folds=5, scoring='accuracy', n_jobs=-1):
        """
        Train SVM with hyperparameter grid search
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            search_type (str): Type of hyperparameter search
            cv_folds (int): Number of cross-validation folds
            scoring (str): Scoring metric for grid search
            n_jobs (int): Number of parallel jobs
            
        Returns:
            dict: Training results
        """
        print(f"Starting {search_type} grid search with {cv_folds}-fold CV...")
        
        # Get parameter grid
        param_grid = self.get_hyperparameter_grid(search_type)
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=self.random_state))
        ])
        
        # Adjust parameter names for pipeline
        param_grid_pipeline = {f'svm__{key}': value for key, value in param_grid.items()}
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid_pipeline,
            cv=cv,
            scoring=scoring,
            verbose=1,
            n_jobs=n_jobs,
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Store best model
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_trained = True
        
        # Store training history
        results = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_,
            'search_type': search_type,
            'cv_folds': cv_folds
        }
        
        self.training_history = results
        
        print(f"Grid search completed!")
        print(f"Best cross-validation score: {results['best_score']:.4f}")
        print(f"Best parameters: {results['best_params']}")
        
        return results
    
    def evaluate_model(self, X_test, y_test, class_names=None):
        """
        Comprehensive model evaluation
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            class_names (list): Names of classes for reporting
            
        Returns:
            dict: Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating model performance...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Classification report
        if class_names is None:
            class_names = self.encoder.classes_
        
        class_report = classification_report(
            y_test, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, confusion_matrix, class_names=None, 
                            figsize=(10, 8), normalize=False):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix (numpy.ndarray): Confusion matrix
            class_names (list): Class names for labels
            figsize (tuple): Figure size
            normalize (bool): Whether to normalize the matrix
        """
        if class_names is None:
            class_names = self.encoder.classes_
        
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm = confusion_matrix
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def plot_hyperparameter_results(self, figsize=(15, 5)):
        """
        Plot hyperparameter search results
        
        Args:
            figsize (tuple): Figure size
        """
        if not self.training_history:
            print("No training history available")
            return
        
        cv_results = self.training_history['cv_results']
        results_df = pd.DataFrame(cv_results)
        
        # Extract parameter values for plotting
        param_columns = [col for col in results_df.columns if col.startswith('param_')]
        
        if len(param_columns) < 2:
            print("Not enough parameters to create meaningful plots")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Score vs C parameter
        if 'param_svm__C' in results_df.columns:
            c_values = results_df['param_svm__C'].values
            scores = results_df['mean_test_score'].values
            
            axes[0].semilogx(c_values, scores, 'bo-')
            axes[0].set_xlabel('C Parameter')
            axes[0].set_ylabel('CV Score')
            axes[0].set_title('CV Score vs C Parameter')
            axes[0].grid(True)
        
        # Plot 2: Score distribution
        axes[1].hist(results_df['mean_test_score'], bins=20, alpha=0.7, edgecolor='black')
        axes[1].axvline(self.training_history['best_score'], color='red', 
                       linestyle='--', label=f"Best: {self.training_history['best_score']:.4f}")
        axes[1].set_xlabel('CV Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of CV Scores')
        axes[1].legend()
        
        # Plot 3: Top parameters performance
        top_10_idx = results_df['mean_test_score'].nlargest(10).index
        top_10_scores = results_df.loc[top_10_idx, 'mean_test_score']
        
        axes[2].barh(range(len(top_10_scores)), top_10_scores)
        axes[2].set_xlabel('CV Score')
        axes[2].set_ylabel('Parameter Combination Rank')
        axes[2].set_title('Top 10 Parameter Combinations')
        axes[2].grid(True, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_prediction_confidence(self, X_test, y_test, threshold=0.7):
        """
        Analyze prediction confidence distribution
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            threshold (float): Confidence threshold for analysis
            
        Returns:
            dict: Confidence analysis results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analysis")
        
        y_pred_proba = self.model.predict_proba(X_test)
        max_probabilities = np.max(y_pred_proba, axis=1)
        y_pred = self.model.predict(X_test)
        
        # Analyze confidence distribution
        high_confidence_mask = max_probabilities >= threshold
        high_conf_accuracy = accuracy_score(y_test[high_confidence_mask], 
                                          y_pred[high_confidence_mask])
        
        low_confidence_mask = max_probabilities < threshold
        if np.sum(low_confidence_mask) > 0:
            low_conf_accuracy = accuracy_score(y_test[low_confidence_mask], 
                                             y_pred[low_confidence_mask])
        else:
            low_conf_accuracy = 0.0
        
        results = {
            'mean_confidence': np.mean(max_probabilities),
            'std_confidence': np.std(max_probabilities),
            'high_confidence_ratio': np.mean(high_confidence_mask),
            'high_confidence_accuracy': high_conf_accuracy,
            'low_confidence_accuracy': low_conf_accuracy,
            'confidence_scores': max_probabilities
        }
        
        # Plot confidence distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(max_probabilities, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', 
                   label=f'Threshold: {threshold}')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        correct_predictions = (y_pred == y_test)
        plt.scatter(max_probabilities[correct_predictions], 
                   np.ones(np.sum(correct_predictions)), 
                   alpha=0.6, label='Correct', color='green')
        plt.scatter(max_probabilities[~correct_predictions], 
                   np.zeros(np.sum(~correct_predictions)), 
                   alpha=0.6, label='Incorrect', color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Prediction Correctness')
        plt.title('Confidence vs Correctness')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def predict_single(self, embedding, confidence_threshold=0.7):
        """
        Predict single face embedding
        
        Args:
            embedding (numpy.ndarray): Single face embedding
            confidence_threshold (float): Minimum confidence for positive prediction
            
        Returns:
            tuple: (predicted_label, confidence_score, is_confident)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Reshape if needed
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(embedding)[0]
        probabilities = self.model.predict_proba(embedding)[0]
        confidence = np.max(probabilities)
        
        # Decode prediction
        predicted_label = self.encoder.inverse_transform([prediction])[0]
        is_confident = confidence >= confidence_threshold
        
        return predicted_label, confidence, is_confident
    
    def save_model(self, model_path='svm_face_model.pkl', encoder_path='label_encoder.pkl'):
        """
        Save trained model and encoder
        
        Args:
            model_path (str): Path to save model
            encoder_path (str): Path to save encoder
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save encoder
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.encoder, f)
        
        # Save training history
        history_path = model_path.replace('.pkl', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Encoder saved to: {encoder_path}")
        print(f"Training history saved to: {history_path}")
    
    def load_model(self, model_path='svm_face_model.pkl', encoder_path='label_encoder.pkl'):
        """
        Load pre-trained model and encoder
        
        Args:
            model_path (str): Path to model file
            encoder_path (str): Path to encoder file
        """
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load encoder
            with open(encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)
            
            # Load training history if available
            history_path = model_path.replace('.pkl', '_history.pkl')
            if os.path.exists(history_path):
                with open(history_path, 'rb') as f:
                    self.training_history = pickle.load(f)
            
            self.is_trained = True
            print(f"Model loaded from: {model_path}")
            print(f"Encoder loaded from: {encoder_path}")
            print(f"Model classes: {self.encoder.classes_}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e


class EnsembleFaceClassifier:
    """Ensemble classifier combining multiple SVM models for better performance"""
    
    def __init__(self, random_state=42):
        """
        Initialize Ensemble Face Classifier
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.ensemble_model = None
        self.encoder = LabelEncoder()
        self.individual_models = {}
        self.is_trained = False
    
    def create_ensemble(self, X_train, y_train):
        """
        Create ensemble of different SVM configurations
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            
        Returns:
            VotingClassifier: Trained ensemble model
        """
        print("Creating ensemble of SVM classifiers...")
        
        # Encode labels
        y_encoded = self.encoder.fit_transform(y_train)
        
        # Define different SVM configurations
        svm_configs = {
            'svm_rbf_1': SVC(kernel='rbf', C=1, gamma='scale', 
                           probability=True, random_state=self.random_state),
            'svm_rbf_10': SVC(kernel='rbf', C=10, gamma='scale', 
                            probability=True, random_state=self.random_state),
            'svm_linear': SVC(kernel='linear', C=1, 
                            probability=True, random_state=self.random_state),
            'svm_poly': SVC(kernel='poly', degree=3, C=1, 
                          probability=True, random_state=self.random_state)
        }
        
        # Create pipelines with scaling
        estimators = []
        for name, svm in svm_configs.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', svm)
            ])
            estimators.append((name, pipeline))
            self.individual_models[name] = pipeline
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probabilities for voting
        )
        
        # Train ensemble
        self.ensemble_model.fit(X_train, y_encoded)
        self.is_trained = True
        
        print(f"Ensemble trained with {len(estimators)} models")
        return self.ensemble_model
    
    def evaluate_individual_models(self, X_test, y_test):
        """
        Evaluate individual models in the ensemble
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            
        Returns:
            dict: Individual model performances
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first")
        
        results = {}
        y_encoded = self.encoder.transform(y_test)
        
        for name, model in self.individual_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_encoded, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            print(f"{name}: {accuracy:.4f}")
        
        # Ensemble performance
        ensemble_pred = self.ensemble_model.predict(X_test)
        ensemble_accuracy = accuracy_score(y_encoded, ensemble_pred)
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'predictions': ensemble_pred
        }
        print(f"Ensemble: {ensemble_accuracy:.4f}")
        
        return results


class ModelComparison:
    """Class for comparing different model configurations"""
    
    @staticmethod
    def compare_kernels(X_train, X_test, y_train, y_test, encoder):
        """
        Compare different SVM kernels
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            encoder: Label encoder
            
        Returns:
            dict: Comparison results
        """
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        results = {}
        
        print("Comparing SVM kernels...")
        
        for kernel in kernels:
            print(f"Testing {kernel} kernel...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel=kernel, probability=True, random_state=42))
            ])
            
            # Train and evaluate
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            results[kernel] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'model': pipeline
            }
            
            print(f"{kernel}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return results
    
    @staticmethod
    def plot_kernel_comparison(comparison_results, figsize=(12, 8)):
        """
        Plot kernel comparison results
        
        Args:
            comparison_results (dict): Results from compare_kernels
            figsize (tuple): Figure size
        """
        kernels = list(comparison_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [comparison_results[kernel][metric] for kernel in kernels]
            
            bars = axes[i].bar(kernels, values, alpha=0.7)
            axes[i].set_title(f'{metric.capitalize()} by Kernel')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def learning_curve_analysis(model, X, y, cv=5, figsize=(10, 6)):
        """
        Analyze learning curves to understand model performance vs training size
        
        Args:
            model: Sklearn model
            X, y: Features and labels
            cv: Cross-validation folds
            figsize: Figure size
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv, 
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return train_sizes_abs, train_scores, val_scores


class AdvancedSVMAnalysis:
    """Advanced analysis tools for SVM models"""
    
    @staticmethod
    def analyze_support_vectors(svm_model, X_train, y_train, class_names=None):
        """
        Analyze support vectors in trained SVM model
        
        Args:
            svm_model: Trained SVM model
            X_train: Training features
            y_train: Training labels
            class_names: Names of classes
            
        Returns:
            dict: Support vector analysis
        """
        # Get the actual SVM from pipeline if needed
        if hasattr(svm_model, 'named_steps'):
            svm = svm_model.named_steps['svm']
        else:
            svm = svm_model
        
        if not hasattr(svm, 'support_vectors_'):
            print("Model doesn't have support vectors (not an SVM)")
            return None
        
        n_support = svm.n_support_
        support_vectors = svm.support_vectors_
        
        analysis = {
            'total_support_vectors': len(support_vectors),
            'support_vectors_per_class': n_support,
            'support_vector_ratio': len(support_vectors) / len(X_train),
            'class_names': class_names if class_names else list(range(len(n_support)))
        }
        
        print("Support Vector Analysis:")
        print(f"Total Support Vectors: {analysis['total_support_vectors']}")
        print(f"Support Vector Ratio: {analysis['support_vector_ratio']:.4f}")
        print("Support Vectors per Class:")
        
        for i, (name, count) in enumerate(zip(analysis['class_names'], n_support)):
            print(f"  {name}: {count}")
        
        return analysis
    
    @staticmethod
    def decision_boundary_confidence(model, X_test, percentiles=[10, 25, 50, 75, 90]):
        """
        Analyze decision boundary confidence
        
        Args:
            model: Trained model
            X_test: Test features
            percentiles: Percentiles to analyze
            
        Returns:
            dict: Confidence analysis
        """
        y_proba = model.predict_proba(X_test)
        max_probabilities = np.max(y_proba, axis=1)
        
        confidence_percentiles = np.percentile(max_probabilities, percentiles)
        
        analysis = {
            'mean_confidence': np.mean(max_probabilities),
            'std_confidence': np.std(max_probabilities),
            'min_confidence': np.min(max_probabilities),
            'max_confidence': np.max(max_probabilities),
            'percentiles': dict(zip(percentiles, confidence_percentiles))
        }
        
        print("Decision Boundary Confidence Analysis:")
        print(f"Mean Confidence: {analysis['mean_confidence']:.4f}")
        print(f"Std Confidence: {analysis['std_confidence']:.4f}")
        print("Confidence Percentiles:")
        for p, value in analysis['percentiles'].items():
            print(f"  {p}th percentile: {value:.4f}")
        
        return analysis


# Utility functions
def calculate_classification_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Calculate optimal classification threshold
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        
    Returns:
        tuple: (optimal_threshold, best_score)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, average='weighted')
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, average='weighted')
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, average='weighted')
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return optimal_threshold, best_score


if __name__ == "__main__":
    # Example usage
    print("SVM Classification Module")
    print("=" * 40)
    
    # This would typically be called with actual embeddings and labels
    # Example workflow:
    
    # 1. Initialize classifier
    # classifier = SVMFaceClassifier(random_state=42)
    
    # 2. Prepare data
    # X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(embeddings, labels)
    
    # 3. Train with grid search
    # results = classifier.train_with_grid_search(X_train, y_train, search_type='comprehensive')
    
    # 4. Evaluate model
    # eval_results = classifier.evaluate_model(X_test, y_test)
    
    # 5. Analyze results
    # classifier.plot_confusion_matrix(eval_results['confusion_matrix'])
    # classifier.plot_hyperparameter_results()
    
    # 6. Save model
    # classifier.save_model()
    
    print("SVM Classification module loaded successfully!")
    print("Use this module to train and evaluate SVM models for face recognition.")