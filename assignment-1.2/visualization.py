"""
Visualization Module
This module handles dimensionality reduction and visualization of embeddings.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px


class EmbeddingVisualizer:
    """
    Visualize high-dimensional word embeddings in 2D or 3D space.
    
    When we have many dimensions (e.g., 100), we can't visualize them easily.
    So we use techniques to reduce to 2D/3D while keeping important patterns.
    """
    
    @staticmethod
    def reduce_dimensions(word_vectors_dict, method='pca', n_components=2):
        """
        Reduce high-dimensional vectors to 2D or 3D.
        
        Args:
            word_vectors_dict: Dictionary of {word: vector}
            method: 'pca' or 'tsne'
            n_components: 2 or 3 (for visualization)
            
        Returns:
            dict: Contains reduced vectors and labels
        """
        try:
            # Extract words and vectors
            words = list(word_vectors_dict.keys())
            vectors = np.array([word_vectors_dict[w] for w in words])
            
            if method.lower() == 'pca':
                # PCA: Linear dimension reduction - very fast
                # Keeps the directions of maximum variance
                reducer = PCA(n_components=n_components)
                reduced_vectors = reducer.fit_transform(vectors)
                
                print(f"✓ PCA explained variance: {sum(reducer.explained_variance_ratio_):.2%}")
                
            elif method.lower() == 'tsne':
                # t-SNE: Non-linear dimension reduction - slower but often better visually
                # Good at preserving local structure (similar points stay close)
                reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(words)-1))
                reduced_vectors = reducer.fit_transform(vectors)
                
                print(f"✓ t-SNE completed")
                
            else:
                raise ValueError("Method must be 'pca' or 'tsne'")
            
            return {
                'words': words,
                'vectors': reduced_vectors,
                'method': method,
                'n_components': n_components
            }
            
        except Exception as e:
            print(f"✗ Error reducing dimensions: {e}")
            return None
    
    @staticmethod
    def plot_2d_embeddings(reduction_result):
        """
        Create an interactive 2D scatter plot of word embeddings.
        
        Args:
            reduction_result: Output from reduce_dimensions()
            
        Returns:
            plotly Figure object
        """
        if reduction_result is None:
            return None
        
        words = reduction_result['words']
        vectors = reduction_result['vectors']
        method = reduction_result['method'].upper()
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=vectors[:, 0],
            y=vectors[:, 1],
            mode='markers+text',
            marker=dict(
                size=8,
                color=list(range(len(words))),  # Color by index for variety
                colorscale='Viridis',
                showscale=True
            ),
            text=words,
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'2D Word Embeddings Visualization ({method})',
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            hovermode='closest',
            height=600,
            width=800,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_3d_embeddings(reduction_result):
        """
        Create an interactive 3D scatter plot of word embeddings.
        
        Args:
            reduction_result: Output from reduce_dimensions()
            
        Returns:
            plotly Figure object
        """
        if reduction_result is None or reduction_result['n_components'] != 3:
            return None
        
        words = reduction_result['words']
        vectors = reduction_result['vectors']
        method = reduction_result['method'].upper()
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=vectors[:, 0],
            y=vectors[:, 1],
            z=vectors[:, 2],
            mode='markers+text',
            marker=dict(
                size=5,
                color=list(range(len(words))),
                colorscale='Viridis',
                showscale=True
            ),
            text=words,
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D Word Embeddings Visualization ({method})',
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            hovermode='closest',
            height=700,
            width=900
        )
        
        return fig


# Create a global instance
visualizer = EmbeddingVisualizer()
