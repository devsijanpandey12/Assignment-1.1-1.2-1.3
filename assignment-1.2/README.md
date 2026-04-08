# 🔤 Word Embeddings Explorer

A beginner-friendly project demonstrating word embeddings, dimensionality reduction, and interactive visualization using TF-IDF, REST API, and Streamlit.

## 🎯 What This Project Does

1. **Computes TF-IDF Embeddings** on a custom corpus of documents
2. **Finds Similarity** between words using cosine similarity
3. **Visualizes Embeddings** in 2D/3D using PCA and t-SNE
4. **Provides a REST API** for computing embeddings programmatically
5. **Offers a Web UI** (Streamlit) for interactive exploration

## 📁 Project Structure

```
NLP_Assignment_2/
├── requirements.txt          # Python dependencies
├── corpus.json              # Sample documents for training
├── embeddings.py            # TF-IDF embedding logic
├── visualization.py         # PCA and t-SNE visualization
├── api_server.py            # Flask REST API server
├── streamlit_app.py         # Streamlit web interface
└── README.md               # This file
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**What's being installed:**
- `scikit-learn`: Machine learning library for TF-IDF and PCA
- `streamlit`: Web app framework
- `flask`: REST API framework
- `plotly`: Interactive visualizations
- `numpy`, `pandas`: Data manipulation

### Step 2: Start the API Server

Open a terminal and run:

```bash
python api_server.py
```

You should see:
```
============================================================
Word Embeddings REST API
============================================================

Available Endpoints:
  GET  /health                    - Check if API is running
  POST /api/train                 - Train model on corpus
  POST /api/embedding             - Get embedding for a word
  POST /api/nearest-neighbors     - Find similar words
  GET  /api/vocabulary            - Get all words in vocabulary
  POST /api/visualize             - Get visualization data

Starting server on http://localhost:5000
============================================================
```

**⚠️ Important:** Keep this terminal open! The API server must be running for the Streamlit app to work.

### Step 3: Start the Streamlit App

Open a **new terminal** and run:

```bash
streamlit run streamlit_app.py
```

Your browser should automatically open at `http://localhost:8501`

If not, manually go to: http://localhost:8501

## 📖 How to Use

### 1. **Train the Model**
   - Click the "🚀 Train Model" button in the sidebar
   - Wait for training to complete
   - You'll see "✓ Model ready"

### 2. **Explore Words** (Tab 1)
   - Enter a word (e.g., "python", "learning", "data")
   - See its embedding and vector norm
   - View the top similar words
   - Check the similarity scores

### 3. **Visualize Embeddings** (Tab 2)
   - Choose between PCA (fast) or t-SNE (beautiful)
   - Choose 2D or 3D visualization
   - Click "🎨 Generate Visualization"
   - Explore the interactive plot

### 4. **Learn** (Tab 3)
   - Understand how TF-IDF works
   - Learn about cosine similarity
   - See how PCA and t-SNE differ
   - Try suggested searches

## 🔍 Understanding the Code

### `embeddings.py` - The Core Logic

```python
model = EmbeddingModel()
model.load_corpus('corpus.json')      # Load documents
model.train()                          # Train on corpus
embedding = model.get_embedding('python')      # Get vector
neighbors = model.get_nearest_neighbors('python')  # Find similar
```

**Key Concept:** Each word is represented as a vector of numbers. 
Similar words have similar vectors.

### `api_server.py` - The Backend

Provides HTTP endpoints:
- `POST /api/train` - Train the model
- `POST /api/embedding` - Get embedding for a word
- `POST /api/nearest-neighbors` - Find similar words
- `POST /api/visualize` - Get 2D/3D coordinates

### `streamlit_app.py` - The Frontend

Calls the API and displays results beautifully:
- Search tab for word exploration
- Visualization tab for embedding plots
- Learn tab with explanations

## 💡 Example Queries

Try these words to see interesting relationships:

```
python        → Similar to: programming, language, data
learning      → Similar to: machine, models, training
intelligence  → Similar to: artificial, learning, networks
data          → Similar to: science, analysis, processing
neural        → Similar to: networks, deep, models
```

## 📊 Understanding the Visualizations

### TF-IDF Scores (Bar Chart)
Shows how important each dimension is for a word.

### Nearest Neighbors (Bar Chart)
Shows similarity scores (0 to 1) for neighboring words.

### PCA Embedding (Scatter Plot)
A fast, linear projection. Good for quick overview.
- Words close together = similar vocabulary
- Captures major variance directions

### t-SNE Embedding (Scatter Plot)
A smart, non-linear projection. Better visual clusters.
- Takes longer but looks nicer
- Preserves local neighborhood structure better

## 🛠️ Customizing the Project

### Change the Corpus
Edit `corpus.json` to add or remove documents:

```json
{
  "documents": [
    "Your document here",
    "Another document",
    "More text..."
  ]
}
```

Then retrain the model by clicking "🚀 Train Model" in the app.

### Adjust API Ports
In `api_server.py`, change:
```python
app.run(debug=True, port=5000)  # Change 5000 to any port
```

In `streamlit_app.py`, change:
```python
API_URL = "http://localhost:5000"  # Match the port above
```

### Modify Visualization Parameters
In `visualization.py`:
```python
reducer = PCA(n_components=2)  # Change to 3 for 3D
reducer = TSNE(n_components=2, perplexity=30)  # Decrease perplexity for faster results
```

## ❓ Troubleshooting

### "Cannot connect to API server"
- Make sure you ran `python api_server.py` in a terminal
- Check that no other app is using port 5000
- Try restarting both the API and Streamlit

### "Word not found in vocabulary"
- The word might not have appeared in the corpus
- Try simpler words like "python", "learning", "data"
- Add more documents to `corpus.json` and retrain

### App seems slow
- t-SNE is slower than PCA - this is normal
- For large vocabularies, use PCA instead
- Decrease perplexity in t-SNE for faster results

### Streamlit not opening
- Check terminal output for the URL (usually http://localhost:8501)
- Manually paste it into your browser

## 📚 Learning Resources

**What is TF-IDF?**
- https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Simple explanation: measure of how important a word is to a document

**What is Cosine Similarity?**
- Measures angle between two vectors
- High similarity (close to 1) means similar words
- Works in any number of dimensions

**What is PCA?**
- Principal Component Analysis
- Finds directions of maximum variance
- Fast and interpretable

**What is t-SNE?**
- t-Distributed Stochastic Neighbor Embedding
- Preserves local neighborhood structure
- Better for visualization

## 🎓 Next Steps to Learn More

1. **Try Different Embeddings:** Replace TF-IDF with Word2Vec or FastText
2. **Add More Data:** Use larger corpus from Wikipedia, news, etc.
3. **Build a Search Engine:** Use embeddings to find relevant documents
4. **Deploy to Cloud:** Put this on Heroku or AWS
5. **Real Similarity:** Explore pre-trained embeddings like GloVe or FastText

## 📝 Notes for Beginners

### What are embeddings?
Embeddings convert words into vectors (lists of numbers). Computers can't understand words directly, but they can do math on numbers.

### Why TF-IDF?
- Simple and interpretable
- No need for large training data
- Shows which words are important in documents

### Why PCA?
- Very fast
- Shows major patterns

### Why t-SNE?
- Better at showing clusters
- Words with similar meanings group together

### How does the app work?
1. You enter a word in Streamlit
2. Streamlit sends it to the API (Flask server)
3. API computes the embedding
4. Result comes back to Streamlit
5. Streamlit displays it beautifully

