"""
Embeddings Module
This module handles creating and managing word embeddings using TF-IDF.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


class EmbeddingModel:
    """
    A simple embedding model using TF-IDF (Term Frequency-Inverse Document Frequency).
    
    Think of TF-IDF as a way to convert words into numbers so computers can understand them.
    - TF: How often a word appears in a document
    - IDF: How unique/rare a word is across all documents
    """
    
    def __init__(self):
        """Initialize the embedding model"""
        self.vectorizer = None
        self.tfidf_matrix = None
        self.vocabulary = None
        self.documents = []
        self.word_vectors = {}
        
    def load_corpus(self, corpus_path='corpus.json'):
        """
        Load documents from a JSON file.
        
        Args:
            corpus_path: Path to the JSON file containing documents
        """
        try:
            with open(corpus_path, 'r') as f:
                data = json.load(f)
                self.documents = data.get('documents', [])
            
            if not self.documents:
                raise ValueError("No documents found in corpus")
            
            print(f"✓ Loaded {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"✗ Error loading corpus: {e}")
            return False
    
    def train(self):
        """
        Train the TF-IDF model on the corpus.
        
        This creates a matrix where each row is a document and each column is a word.
        The values show how important each word is in each document.
        """
        try:
            # Create a TF-IDF vectorizer
            # max_features: Keep only the top 100 most frequent words
            # min_df: Ignore words that appear in less than 1 document
            # max_df: Ignore words that appear in more than 90% of documents
            self.vectorizer = TfidfVectorizer(
                max_features=100,
                min_df=1,
                max_df=0.9,
                lowercase=True,
                stop_words='english'  # Remove common words like 'the', 'a', 'is'
            )
            
            # Fit the vectorizer and transform documents
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            
            # Get the vocabulary (all unique words)
            self.vocabulary = self.vectorizer.get_feature_names_out()
            
            # Create word vectors (average TF-IDF score for each word)
            self._compute_word_vectors()
            
            print(f"✓ Model trained successfully")
            print(f"✓ Vocabulary size: {len(self.vocabulary)} words")
            return True
        except Exception as e:
            print(f"✗ Error training model: {e}")
            return False
    
    def _compute_word_vectors(self):
        """Compute vectors for each word based on its TF-IDF scores"""
        # Get the dense matrix representation
        dense_matrix = self.tfidf_matrix.toarray()
        
        # For each word, store its TF-IDF vector
        for idx, word in enumerate(self.vocabulary):
            # Get the TF-IDF scores for this word across all documents
            self.word_vectors[word] = dense_matrix[:, idx]
    
    def get_embedding(self, word):
        """
        Get the embedding (TF-IDF vector) for a word.
        
        Args:
            word: The word to get embedding for
            
        Returns:
            dict: Contains the embedding vector and word
        """
        word = word.lower().strip()
        
        # Check if word is in vocabulary
        if word not in self.word_vectors:
            return {
                'error': f"Word '{word}' not found in vocabulary",
                'vocabulary_size': len(self.vocabulary),
                'sample_words': list(self.vocabulary[:10])
            }
        
        # Get the word vector
        vector = self.word_vectors[word].tolist()
        
        return {
            'word': word,
            'embedding': vector,
            'dimension': len(vector),
            'norm': float(np.linalg.norm(self.word_vectors[word]))
        }
    
    def get_nearest_neighbors(self, word, top_k=5):
        """
        Find the most similar words to a given word.
        
        Uses cosine similarity to measure how similar word vectors are.
        Cosine similarity ranges from -1 (opposite) to 1 (identical).
        
        Args:
            word: The target word
            top_k: How many similar words to return
            
        Returns:
            dict: Contains the nearest neighbors and their similarity scores
        """
        word = word.lower().strip()
        
        # Check if word is in vocabulary
        if word not in self.word_vectors:
            return {
                'error': f"Word '{word}' not found in vocabulary"
            }
        
        # Get the vector for the target word
        target_vector = self.word_vectors[word].reshape(1, -1)
        
        # Calculate similarity with all words
        similarities = {}
        for w, vec in self.word_vectors.items():
            if w == word:
                continue  # Skip the word itself
            
            # Compute cosine similarity
            sim = cosine_similarity(target_vector, vec.reshape(1, -1))[0][0]
            similarities[w] = float(sim)
        
        # Sort by similarity and get top-k
        sorted_neighbors = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return {
            'word': word,
            'nearest_neighbors': [
                {'word': w, 'similarity': float(sim)}
                for w, sim in sorted_neighbors
            ]
        }
    
    def get_vocabulary(self):
        """Get the list of all words in the vocabulary"""
        return {
            'vocabulary_size': len(self.vocabulary),
            'words': self.vocabulary.tolist()
        }


# Create a global instance that will be used by the API
embedding_model = EmbeddingModel()
