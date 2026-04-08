"""
REST API Server for Word Embeddings
This server exposes endpoints to get embeddings and find similar words.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from embeddings import embedding_model
from visualization import visualizer
import json

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests (needed for Streamlit to call the API)

# Global variable to track if model is trained
model_trained = False


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint - verify the API is running.
    
    Returns:
        JSON with status
    """
    return jsonify({
        'status': 'ok',
        'model_trained': model_trained,
        'message': 'Embeddings API is running'
    })


@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train the embedding model on the corpus.
    
    This endpoint must be called first before using other endpoints.
    
    Returns:
        JSON with training status
    """
    global model_trained
    
    try:
        # Load corpus
        if not embedding_model.load_corpus('corpus.json'):
            return jsonify({'error': 'Failed to load corpus'}), 400
        
        # Train the model
        if not embedding_model.train():
            return jsonify({'error': 'Failed to train model'}), 400
        
        model_trained = True
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'vocabulary_size': len(embedding_model.vocabulary),
            'num_documents': len(embedding_model.documents)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/embedding', methods=['POST'])
def get_embedding():
    """
    Get the embedding for a specific word.
    
    Request body:
        {
            "word": "python"
        }
    
    Returns:
        JSON with embedding vector and metadata
    """
    if not model_trained:
        return jsonify({'error': 'Model not trained. Call /api/train first'}), 400
    
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        
        if not word:
            return jsonify({'error': 'Please provide a word'}), 400
        
        result = embedding_model.get_embedding(word)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nearest-neighbors', methods=['POST'])
def get_nearest_neighbors():
    """
    Find similar words to a given word.
    
    Request body:
        {
            "word": "python",
            "top_k": 5
        }
    
    Returns:
        JSON with list of similar words and similarity scores
    """
    if not model_trained:
        return jsonify({'error': 'Model not trained. Call /api/train first'}), 400
    
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        top_k = data.get('top_k', 5)
        
        if not word:
            return jsonify({'error': 'Please provide a word'}), 400
        
        result = embedding_model.get_nearest_neighbors(word, top_k)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/vocabulary', methods=['GET'])
def get_vocabulary():
    """
    Get the list of all words in the vocabulary.
    
    Returns:
        JSON with vocabulary size and sample words
    """
    if not model_trained:
        return jsonify({'error': 'Model not trained. Call /api/train first'}), 400
    
    try:
        result = embedding_model.get_vocabulary()
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize', methods=['POST'])
def visualize_embeddings():
    """
    Generate visualization of embeddings.
    
    Request body:
        {
            "method": "pca",  # or "tsne"
            "dimension": 2     # 2 or 3
        }
    
    Returns:
        JSON with visualization data
    """
    if not model_trained:
        return jsonify({'error': 'Model not trained. Call /api/train first'}), 400
    
    try:
        data = request.get_json()
        method = data.get('method', 'pca').lower()
        dimension = data.get('dimension', 2)
        
        if method not in ['pca', 'tsne']:
            return jsonify({'error': 'Method must be "pca" or "tsne"'}), 400
        
        if dimension not in [2, 3]:
            return jsonify({'error': 'Dimension must be 2 or 3'}), 400
        
        # Get reduction result
        reduction_result = visualizer.reduce_dimensions(
            embedding_model.word_vectors,
            method=method,
            n_components=dimension
        )
        
        if reduction_result is None:
            return jsonify({'error': 'Visualization failed'}), 500
        
        # Convert numpy arrays to lists for JSON serialization
        vectors = reduction_result['vectors'].tolist()
        
        return jsonify({
            'words': reduction_result['words'],
            'vectors': vectors,
            'method': method,
            'dimension': dimension
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Available endpoints: /health, /api/train, /api/embedding, /api/nearest-neighbors, /api/vocabulary, /api/visualize'
    }), 404


if __name__ == '__main__':
    print("=" * 60)
    print("Word Embeddings REST API")
    print("=" * 60)
    print("\nAvailable Endpoints:")
    print("  GET  /health                    - Check if API is running")
    print("  POST /api/train                 - Train model on corpus")
    print("  POST /api/embedding             - Get embedding for a word")
    print("  POST /api/nearest-neighbors     - Find similar words")
    print("  GET  /api/vocabulary            - Get all words in vocabulary")
    print("  POST /api/visualize             - Get visualization data")
    print("\nStarting server on http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, port=5000)
