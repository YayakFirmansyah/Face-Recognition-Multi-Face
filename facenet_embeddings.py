# -*- coding: utf-8 -*-
"""
FaceNet Embedding Module
Handles face embedding generation using pre-trained FaceNet model
"""

import numpy as np
import tensorflow as tf
from keras_facenet import FaceNet
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


class FaceNetEmbedder:
    def __init__(self, model_name='facenet'):
        """
        Initialize FaceNet Embedder
        
        Args:
            model_name (str): Name of the model to use ('facenet' or custom path)
        """
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        self.load_model()
    
    def load_model(self):
        """Load pre-trained FaceNet model"""
        try:
            print("Loading FaceNet model...")
            self.model = FaceNet()
            print("FaceNet model loaded successfully!")
            print(f"Model input shape: {self.model.model.input_shape}")
            print(f"Model output shape: {self.model.model.output_shape}")
            
        except Exception as e:
            print(f"Error loading FaceNet model: {str(e)}")
            raise e
    
    def preprocess_face_for_facenet(self, face_image):
        """
        Preprocess face image for FaceNet input
        
        Args:
            face_image (numpy.ndarray): Face image array
            
        Returns:
            numpy.ndarray: Preprocessed face image
        """
        # Ensure face is in correct format
        if face_image.shape != (160, 160, 3):
            print(f"Warning: Face shape {face_image.shape} != (160, 160, 3)")
        
        # Convert to float32 and scale to [0, 255] if needed
        if face_image.max() <= 1.0:
            face_processed = (face_image * 255.0).astype('float32')
        else:
            face_processed = face_image.astype('float32')
        
        # Add batch dimension
        face_batch = np.expand_dims(face_processed, axis=0)
        
        return face_batch
    
    def get_single_embedding(self, face_image):
        """
        Generate embedding for a single face image
        
        Args:
            face_image (numpy.ndarray): Preprocessed face image
            
        Returns:
            numpy.ndarray: Face embedding vector (512-dimensional)
        """
        try:
            # Preprocess face
            face_batch = self.preprocess_face_for_facenet(face_image)
            
            # Generate embedding
            embedding = self.model.embeddings(face_batch)
            
            # Return single embedding vector
            return embedding[0]
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def get_batch_embeddings(self, face_images, batch_size=32):
        """
        Generate embeddings for multiple face images in batches
        
        Args:
            face_images (numpy.ndarray): Array of face images
            batch_size (int): Batch size for processing
            
        Returns:
            numpy.ndarray: Array of embedding vectors
        """
        if len(face_images) == 0:
            return np.array([])
        
        print(f"Generating embeddings for {len(face_images)} faces...")
        
        embeddings = []
        failed_indices = []
        
        # Process in batches
        for i in range(0, len(face_images), batch_size):
            batch_end = min(i + batch_size, len(face_images))
            batch_faces = face_images[i:batch_end]
            
            try:
                # Preprocess batch
                batch_processed = []
                batch_indices = []
                
                for j, face in enumerate(batch_faces):
                    processed_face = self.preprocess_face_for_facenet(face)
                    if processed_face is not None:
                        batch_processed.append(processed_face[0])  # Remove batch dim
                        batch_indices.append(i + j)
                
                if batch_processed:
                    # Stack into batch
                    batch_array = np.stack(batch_processed, axis=0)
                    
                    # Generate embeddings
                    batch_embeddings = self.model.embeddings(batch_array)
                    
                    embeddings.extend(batch_embeddings)
                    
                    print(f"Processed batch {i//batch_size + 1}/{(len(face_images) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                failed_indices.extend(range(i, batch_end))
                continue
        
        if failed_indices:
            print(f"Failed to process {len(failed_indices)} images")
        
        embeddings_array = np.array(embeddings)
        print(f"Generated {len(embeddings_array)} embeddings successfully")
        
        return embeddings_array
    
    def generate_embeddings_for_dataset(self, faces, labels, batch_size=32):
        """
        Generate embeddings for entire dataset
        
        Args:
            faces (numpy.ndarray): Array of face images
            labels (numpy.ndarray): Corresponding labels
            batch_size (int): Batch size for processing
            
        Returns:
            tuple: (embeddings_array, valid_labels_array, failed_indices)
        """
        if len(faces) == 0:
            print("No faces provided for embedding generation")
            return np.array([]), np.array([]), []
        
        print(f"Generating embeddings for dataset: {len(faces)} faces")
        
        embeddings = []
        valid_labels = []
        failed_indices = []
        
        # Process in batches
        for i in range(0, len(faces), batch_size):
            batch_end = min(i + batch_size, len(faces))
            batch_faces = faces[i:batch_end]
            batch_labels = labels[i:batch_end]
            
            try:
                # Preprocess batch
                batch_processed = []
                batch_valid_labels = []
                
                for j, (face, label) in enumerate(zip(batch_faces, batch_labels)):
                    processed_face = self.preprocess_face_for_facenet(face)
                    if processed_face is not None:
                        batch_processed.append(processed_face[0])
                        batch_valid_labels.append(label)
                    else:
                        failed_indices.append(i + j)
                
                if batch_processed:
                    # Generate embeddings for batch
                    batch_array = np.stack(batch_processed, axis=0)
                    batch_embeddings = self.model.embeddings(batch_array)
                    
                    embeddings.extend(batch_embeddings)
                    valid_labels.extend(batch_valid_labels)
                
                print(f"Processed batch {i//batch_size + 1}/{(len(faces) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                failed_indices.extend(range(i, batch_end))
                continue
        
        embeddings_array = np.array(embeddings)
        valid_labels_array = np.array(valid_labels)
        
        print(f"Successfully generated {len(embeddings_array)} embeddings")
        if failed_indices:
            print(f"Failed to process {len(failed_indices)} faces")
        
        return embeddings_array, valid_labels_array, failed_indices
    
    def save_embeddings(self, embeddings, labels, filepath='face_embeddings.npz'):
        """
        Save embeddings and labels to file
        
        Args:
            embeddings (numpy.ndarray): Embedding vectors
            labels (numpy.ndarray): Corresponding labels
            filepath (str): Path to save file
        """
        try:
            np.savez_compressed(filepath, 
                              embeddings=embeddings, 
                              labels=labels,
                              model_name=self.model_name)
            print(f"Embeddings saved to {filepath}")
            print(f"Saved {len(embeddings)} embeddings with {len(np.unique(labels))} unique labels")
            
        except Exception as e:
            print(f"Error saving embeddings: {str(e)}")
    
    def load_embeddings(self, filepath='face_embeddings.npz'):
        """
        Load embeddings and labels from file
        
        Args:
            filepath (str): Path to embeddings file
            
        Returns:
            tuple: (embeddings, labels) or (None, None) if failed
        """
        try:
            if not os.path.exists(filepath):
                print(f"Embeddings file not found: {filepath}")
                return None, None
            
            data = np.load(filepath)
            embeddings = data['embeddings']
            labels = data['labels']
            
            print(f"Loaded embeddings from {filepath}")
            print(f"Loaded {len(embeddings)} embeddings with {len(np.unique(labels))} unique labels")
            
            return embeddings, labels
            
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return None, None


class EmbeddingAnalyzer:
    """Class for analyzing and visualizing embeddings"""
    
    @staticmethod
    def calculate_similarity_matrix(embeddings):
        """
        Calculate cosine similarity matrix for embeddings
        
        Args:
            embeddings (numpy.ndarray): Array of embedding vectors
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        return cosine_similarity(embeddings)
    
    @staticmethod
    def find_most_similar_faces(embeddings, labels, target_index, top_k=5):
        """
        Find most similar faces to a target face
        
        Args:
            embeddings (numpy.ndarray): Array of embedding vectors
            labels (numpy.ndarray): Corresponding labels
            target_index (int): Index of target face
            top_k (int): Number of similar faces to return
            
        Returns:
            list: List of (index, similarity_score, label) tuples
        """
        if target_index >= len(embeddings):
            print(f"Target index {target_index} out of range")
            return []
        
        target_embedding = embeddings[target_index].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, embeddings)[0]
        
        # Get top-k most similar (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in similar_indices:
            results.append((idx, similarities[idx], labels[idx]))
        
        return results
    
    @staticmethod
    def analyze_embedding_distribution(embeddings, labels):
        """
        Analyze the distribution of embeddings
        
        Args:
            embeddings (numpy.ndarray): Array of embedding vectors
            labels (numpy.ndarray): Corresponding labels
            
        Returns:
            dict: Analysis results
        """
        unique_labels = np.unique(labels)
        
        analysis = {
            'num_embeddings': len(embeddings),
            'num_unique_labels': len(unique_labels),
            'embedding_dimension': embeddings.shape[1],
            'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
            'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
            'label_distribution': {}
        }
        
        # Analyze per-label statistics
        for label in unique_labels:
            label_mask = labels == label
            label_embeddings = embeddings[label_mask]
            
            analysis['label_distribution'][label] = {
                'count': len(label_embeddings),
                'mean_norm': np.mean(np.linalg.norm(label_embeddings, axis=1)),
                'intra_similarity': np.mean(cosine_similarity(label_embeddings))
            }
        
        return analysis
    
    @staticmethod
    def plot_embedding_visualization(embeddings, labels, method='tsne', figsize=(12, 8)):
        """
        Visualize embeddings in 2D using dimensionality reduction
        
        Args:
            embeddings (numpy.ndarray): Array of embedding vectors
            labels (numpy.ndarray): Corresponding labels
            method (str): Dimensionality reduction method ('tsne', 'pca', 'umap')
            figsize (tuple): Figure size
        """
        try:
            if method.lower() == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            elif method.lower() == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
            elif method.lower() == 'umap':
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            print(f"Applying {method.upper()} dimensionality reduction...")
            embeddings_2d = reducer.fit_transform(embeddings)
            
            # Plot
            plt.figure(figsize=figsize)
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          c=[colors[i]], label=label, alpha=0.7, s=50)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Face Embeddings Visualization ({method.upper()})')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError as e:
            print(f"Required library not installed for {method}: {str(e)}")
            print("Install with: pip install scikit-learn umap-learn")
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
    
    @staticmethod
    def plot_similarity_heatmap(embeddings, labels, max_samples=20, figsize=(12, 10)):
        """
        Plot similarity heatmap for embeddings
        
        Args:
            embeddings (numpy.ndarray): Array of embedding vectors
            labels (numpy.ndarray): Corresponding labels
            max_samples (int): Maximum number of samples to include in heatmap
            figsize (tuple): Figure size
        """
        import seaborn as sns
        
        # Limit number of samples for readability
        if len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            sample_embeddings = embeddings[indices]
            sample_labels = labels[indices]
        else:
            sample_embeddings = embeddings
            sample_labels = labels
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(sample_embeddings)
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(similarity_matrix, 
                   xticklabels=sample_labels,
                   yticklabels=sample_labels,
                   annot=False,
                   cmap='viridis',
                   square=True)
        
        plt.title('Face Embedding Similarity Matrix')
        plt.xlabel('Face Index')
        plt.ylabel('Face Index')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()


class EmbeddingQualityChecker:
    """Class for checking embedding quality and detecting issues"""
    
    @staticmethod
    def check_embedding_quality(embeddings, labels, similarity_threshold=0.3):
        """
        Check quality of embeddings and detect potential issues
        
        Args:
            embeddings (numpy.ndarray): Array of embedding vectors
            labels (numpy.ndarray): Corresponding labels
            similarity_threshold (float): Threshold for detecting similar embeddings
            
        Returns:
            dict: Quality assessment results
        """
        results = {
            'total_embeddings': len(embeddings),
            'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'unique_labels': len(np.unique(labels)),
            'issues': [],
            'statistics': {}
        }
        
        if len(embeddings) == 0:
            results['issues'].append("No embeddings provided")
            return results
        
        # Check for zero or near-zero embeddings
        zero_embeddings = np.sum(np.all(np.abs(embeddings) < 1e-6, axis=1))
        if zero_embeddings > 0:
            results['issues'].append(f"Found {zero_embeddings} zero or near-zero embeddings")
        
        # Check for duplicate embeddings
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
        
        high_similarity_pairs = np.where(similarity_matrix > 0.99)
        if len(high_similarity_pairs[0]) > 0:
            results['issues'].append(f"Found {len(high_similarity_pairs[0])//2} highly similar embedding pairs (>0.99)")
        
        # Calculate intra-class and inter-class similarities
        unique_labels = np.unique(labels)
        intra_class_similarities = []
        inter_class_similarities = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            
            if len(label_indices) > 1:
                # Intra-class similarities
                label_embeddings = embeddings[label_indices]
                label_sim_matrix = cosine_similarity(label_embeddings)
                np.fill_diagonal(label_sim_matrix, 0)
                intra_class_similarities.extend(label_sim_matrix[label_sim_matrix > 0])
            
            # Inter-class similarities
            other_indices = np.where(labels != label)[0]
            if len(other_indices) > 0:
                inter_similarities = cosine_similarity(
                    embeddings[label_indices], 
                    embeddings[other_indices]
                )
                inter_class_similarities.extend(inter_similarities.flatten())
        
        # Calculate statistics
        if intra_class_similarities:
            results['statistics']['mean_intra_class_similarity'] = np.mean(intra_class_similarities)
            results['statistics']['std_intra_class_similarity'] = np.std(intra_class_similarities)
        
        if inter_class_similarities:
            results['statistics']['mean_inter_class_similarity'] = np.mean(inter_class_similarities)
            results['statistics']['std_inter_class_similarity'] = np.std(inter_class_similarities)
        
        # Check if intra-class similarity is higher than inter-class (good sign)
        if (intra_class_similarities and inter_class_similarities and
            np.mean(intra_class_similarities) <= np.mean(inter_class_similarities)):
            results['issues'].append("Mean intra-class similarity is not higher than inter-class similarity")
        
        # Check embedding norms
        norms = np.linalg.norm(embeddings, axis=1)
        results['statistics']['mean_embedding_norm'] = np.mean(norms)
        results['statistics']['std_embedding_norm'] = np.std(norms)
        
        # Check for unusually low or high norms
        if np.min(norms) < 0.1:
            results['issues'].append("Some embeddings have unusually low norms")
        if np.max(norms) > 100:
            results['issues'].append("Some embeddings have unusually high norms")
        
        return results
    
    @staticmethod
    def suggest_improvements(quality_results):
        """
        Suggest improvements based on quality assessment
        
        Args:
            quality_results (dict): Results from check_embedding_quality
            
        Returns:
            list: List of improvement suggestions
        """
        suggestions = []
        
        if not quality_results['issues']:
            suggestions.append("Embedding quality looks good!")
            return suggestions
        
        for issue in quality_results['issues']:
            if "zero or near-zero embeddings" in issue:
                suggestions.append("Check input images for quality issues or preprocessing errors")
            
            elif "highly similar embedding pairs" in issue:
                suggestions.append("Check for duplicate images in dataset or review face detection accuracy")
            
            elif "intra-class similarity" in issue:
                suggestions.append("Consider collecting more diverse images per person or improving face detection")
            
            elif "unusually low norms" in issue:
                suggestions.append("Check image preprocessing - faces might not be properly normalized")
            
            elif "unusually high norms" in issue:
                suggestions.append("Review face preprocessing - there might be scaling issues")
        
        # General suggestions based on statistics
        stats = quality_results.get('statistics', {})
        
        if stats.get('mean_intra_class_similarity', 0) < 0.3:
            suggestions.append("Low intra-class similarity suggests need for better face alignment or more consistent preprocessing")
        
        if stats.get('mean_inter_class_similarity', 0) > 0.7:
            suggestions.append("High inter-class similarity suggests faces might be too similar or preprocessing issues")
        
        return suggestions


# Utility functions for embedding operations
def calculate_face_distance(embedding1, embedding2, metric='cosine'):
    """
    Calculate distance between two face embeddings
    
    Args:
        embedding1 (numpy.ndarray): First embedding
        embedding2 (numpy.ndarray): Second embedding
        metric (str): Distance metric ('cosine', 'euclidean', 'manhattan')
        
    Returns:
        float: Distance between embeddings
    """
    if metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return 1 - similarity
    
    elif metric == 'euclidean':
        return np.linalg.norm(embedding1 - embedding2)
    
    elif metric == 'manhattan':
        return np.sum(np.abs(embedding1 - embedding2))
    
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def find_optimal_threshold(embeddings, labels, metric='cosine'):
    """
    Find optimal threshold for face recognition using embeddings
    
    Args:
        embeddings (numpy.ndarray): Array of embedding vectors
        labels (numpy.ndarray): Corresponding labels
        metric (str): Distance metric to use
        
    Returns:
        tuple: (optimal_threshold, accuracy_at_threshold)
    """
    from sklearn.metrics import accuracy_score
    
    # Generate pairs for threshold testing
    same_person_distances = []
    different_person_distances = []
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            distance = calculate_face_distance(embeddings[i], embeddings[j], metric)
            
            if labels[i] == labels[j]:
                same_person_distances.append(distance)
            else:
                different_person_distances.append(distance)
    
    # Test different thresholds
    all_distances = same_person_distances + different_person_distances
    thresholds = np.linspace(min(all_distances), max(all_distances), 100)
    
    best_threshold = 0
    best_accuracy = 0
    
    for threshold in thresholds:
        # Predict: distance < threshold means same person
        same_predictions = [1 if d < threshold else 0 for d in same_person_distances]
        diff_predictions = [0 if d < threshold else 1 for d in different_person_distances]
        
        true_labels = [1] * len(same_person_distances) + [1] * len(different_person_distances)
        predictions = same_predictions + diff_predictions
        
        accuracy = accuracy_score(true_labels, predictions)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold, best_accuracy


if __name__ == "__main__":
    # Example usage
    print("FaceNet Embedding Module")
    print("=" * 40)
    
    # Initialize embedder
    embedder = FaceNetEmbedder()
    
    # Example: Load saved embeddings
    embeddings, labels = embedder.load_embeddings('face_embeddings.npz')
    
    if embeddings is not None:
        print(f"Loaded {len(embeddings)} embeddings")
        
        # Analyze embedding quality
        quality_checker = EmbeddingQualityChecker()
        quality_results = quality_checker.check_embedding_quality(embeddings, labels)
        
        print("\nEmbedding Quality Assessment:")
        print(f"Total embeddings: {quality_results['total_embeddings']}")
        print(f"Unique labels: {quality_results['unique_labels']}")
        print(f"Issues found: {len(quality_results['issues'])}")
        
        for issue in quality_results['issues']:
            print(f"  - {issue}")
        
        # Get improvement suggestions
        suggestions = quality_checker.suggest_improvements(quality_results)
        print("\nImprovement suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
        
        # Analyze embeddings
        analyzer = EmbeddingAnalyzer()
        analysis = analyzer.analyze_embedding_distribution(embeddings, labels)
        
        print(f"\nEmbedding Analysis:")
        print(f"Mean embedding norm: {analysis['mean_norm']:.4f}")
        print(f"Embedding dimension: {analysis['embedding_dimension']}")
        
        # Visualize embeddings (if requested)
        # analyzer.plot_embedding_visualization(embeddings, labels, method='pca')
        
    else:
        print("No embeddings found. Run preprocessing first to generate embeddings.")