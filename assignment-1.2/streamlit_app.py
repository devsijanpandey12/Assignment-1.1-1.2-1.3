"""
Streamlit Web App for Word Embeddings
A user-friendly interface to explore word embeddings and similarities.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Word Embeddings Explorer",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTitle {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# API URL
API_URL = "http://localhost:5000"

# ============================================================================
# SIDEBAR - Configuration
# ============================================================================

st.sidebar.title("⚙️ Configuration")

# Check API health
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    api_status = health_response.json()
    
    if health_response.status_code == 200:
        st.sidebar.success("✅ API Connected")
        model_trained = api_status.get('model_trained', False)
        
        if not model_trained:
            st.sidebar.warning("⚠️ Model not trained yet")
            
            if st.sidebar.button("🚀 Train Model", use_container_width=True):
                with st.spinner("Training model on corpus..."):
                    try:
                        response = requests.post(f"{API_URL}/api/train")
                        result = response.json()
                        
                        if response.status_code == 200:
                            st.sidebar.success(f"✓ Model trained!")
                            st.sidebar.info(f"Vocabulary: {result['vocabulary_size']} words")
                            st.balloons()
                            st.rerun()
                        else:
                            st.sidebar.error(f"Error: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.sidebar.error(f"Connection error: {e}")
        else:
            st.sidebar.success("✓ Model ready")
            
            # Vocabulary info
            try:
                response = requests.get(f"{API_URL}/api/vocabulary")
                if response.status_code == 200:
                    vocab = response.json()
                    st.sidebar.info(f"📚 Vocabulary: {vocab['vocabulary_size']} words")
            except:
                pass
    else:
        st.sidebar.error("❌ API Error")
        model_trained = False

except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ Cannot connect to API")
    st.sidebar.info("Make sure the API is running: `python api_server.py`")
    model_trained = False
except Exception as e:
    st.sidebar.error(f"❌ Error: {e}")
    model_trained = False


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🔤 Word Embeddings Explorer")
st.markdown("Explore word embeddings using TF-IDF and dimensionality reduction")

if not model_trained:
    st.warning("⚠️ Please train the model first using the button in the sidebar")
    st.stop()

# Create tabs for different features
tab1, tab2, tab3 = st.tabs([
    "🔍 Explore Words",
    "📊 Visualize Embeddings",
    "ℹ️ Learn"
])

# ============================================================================
# TAB 1: EXPLORE WORDS
# ============================================================================

with tab1:
    st.header("Search and Analyze Words")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Word input
        search_word = st.text_input(
            "Enter a word to explore:",
            placeholder="e.g., 'python', 'learning', 'data'",
            key="search_word"
        )
    
    with col2:
        top_k = st.slider(
            "Number of neighbors:",
            min_value=1,
            max_value=10,
            value=5
        )
    
    if search_word:
        search_word = search_word.lower().strip()
        
        # Create columns for results
        result_col1, result_col2 = st.columns([1, 1])
        
        # ---- GET EMBEDDING ----
        with result_col1:
            st.subheader("📈 Word Embedding")
            
            try:
                response = requests.post(
                    f"{API_URL}/api/embedding",
                    json={"word": search_word}
                )
                
                if response.status_code == 200:
                    embedding_result = response.json()
                    
                    if 'error' in embedding_result:
                        st.error(embedding_result['error'])
                        
                        # Show sample words
                        if 'sample_words' in embedding_result:
                            st.info(f"Sample words in vocabulary: {', '.join(embedding_result['sample_words'])}")
                    else:
                        st.success(f"✓ Found: **{embedding_result['word']}**")
                        
                        # Show embedding info
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Dimensions", embedding_result['dimension'])
                        with col_b:
                            st.metric("Vector Norm", f"{embedding_result['norm']:.4f}")
                        
                        # Show embedding vector
                        with st.expander("View Full Embedding Vector"):
                            embedding_vector = embedding_result['embedding']
                            
                            # Create a bar chart of embedding values
                            fig = go.Figure(data=[
                                go.Bar(
                                    y=embedding_vector[:20],  # Show first 20 dimensions
                                    name='TF-IDF Score'
                                )
                            ])
                            fig.update_layout(
                                title=f"TF-IDF Scores for '{search_word}' (first 20 dimensions)",
                                xaxis_title="Dimension",
                                yaxis_title="TF-IDF Score",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show as dataframe
                            embedding_df = pd.DataFrame({
                                'Dimension': range(len(embedding_vector)),
                                'Score': embedding_vector
                            })
                            st.dataframe(embedding_df, use_container_width=True)
                else:
                    st.error("Failed to get embedding")
            
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server")
            except Exception as e:
                st.error(f"Error: {e}")
        
        # ---- GET NEAREST NEIGHBORS ----
        with result_col2:
            st.subheader("🔗 Most Similar Words")
            
            try:
                response = requests.post(
                    f"{API_URL}/api/nearest-neighbors",
                    json={"word": search_word, "top_k": top_k}
                )
                
                if response.status_code == 200:
                    neighbors_result = response.json()
                    
                    if 'error' in neighbors_result:
                        st.error(neighbors_result['error'])
                    else:
                        neighbors = neighbors_result['nearest_neighbors']
                        
                        if not neighbors:
                            st.info("No similar words found")
                        else:
                            st.success(f"✓ Found {len(neighbors)} similar words")
                            
                            # Create a dataframe of neighbors
                            neighbors_df = pd.DataFrame(neighbors)
                            neighbors_df.columns = ['Similar Word', 'Similarity Score']
                            neighbors_df.index = range(1, len(neighbors_df) + 1)
                            neighbors_df.index.name = 'Rank'
                            
                            # Display as table
                            st.dataframe(
                                neighbors_df,
                                use_container_width=True,
                                height=300
                            )
                            
                            # Create a bar chart
                            fig = go.Figure(data=[
                                go.Bar(
                                    y=neighbors_df['Similar Word'],
                                    x=neighbors_df['Similarity Score'],
                                    orientation='h',
                                    marker_color='rgba(31, 119, 180, 0.8)'
                                )
                            ])
                            fig.update_layout(
                                title=f"Similarity Scores (Top {top_k})",
                                xaxis_title="Similarity Score",
                                yaxis_title="Words",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to get neighbors")
            
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server")
            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================================
# TAB 2: VISUALIZE EMBEDDINGS
# ============================================================================

with tab2:
    st.header("Visualize Word Embeddings")
    st.markdown("""
    Dimensionality reduction techniques are used to convert high-dimensional word vectors 
    into 2D or 3D space that we can visualize while preserving important relationships.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.radio(
            "Reduction Method:",
            ["PCA", "t-SNE"],
            help="PCA: Fast, linear reduction | t-SNE: Slower, better local structure"
        )
    
    with col2:
        dimensions = st.radio(
            "Visualization Dimension:",
            [2, 3],
            help="2D: Easier to view | 3D: More detail"
        )
    
    if st.button("🎨 Generate Visualization", use_container_width=True):
        with st.spinner(f"Generating {dimensions}D {method.upper()} visualization..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/visualize",
                    json={
                        "method": method.lower(),
                        "dimension": dimensions
                    },
                    timeout=60  # Give t-SNE time to run
                )
                
                if response.status_code == 200:
                    viz_data = response.json()
                    
                    words = viz_data['words']
                    vectors = np.array(viz_data['vectors'])
                    
                    if dimensions == 2:
                        # 2D Scatter Plot
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=vectors[:, 0],
                            y=vectors[:, 1],
                            mode='markers+text',
                            marker=dict(
                                size=10,
                                color=list(range(len(words))),
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Word Index")
                            ),
                            text=words,
                            textposition='top center',
                            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title=f'2D Word Embeddings Visualization ({method.upper()})',
                            xaxis_title=f'Component 1',
                            yaxis_title=f'Component 2',
                            hovermode='closest',
                            height=700,
                            width=1000,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # 3D
                        # 3D Scatter Plot
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
                                showscale=True,
                                colorbar=dict(title="Word Index")
                            ),
                            text=words,
                            textposition='top center',
                            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title=f'3D Word Embeddings Visualization ({method.upper()})',
                            scene=dict(
                                xaxis_title=f'Component 1',
                                yaxis_title=f'Component 2',
                                zaxis_title=f'Component 3'
                            ),
                            hovermode='closest',
                            height=800,
                            width=1200
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.info(f"✓ Visualized {len(words)} words using {method.upper()}")
                
                else:
                    st.error("Visualization failed")
            
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server")
            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================================
# TAB 3: LEARN
# ============================================================================

with tab3:
    st.header("📚 How This Works")
    
    st.subheader("1️⃣ TF-IDF Embeddings")
    st.markdown("""
    **TF-IDF** (Term Frequency-Inverse Document Frequency) is a numerical statistic that 
    reflects how important a word is to a document in a collection of documents.
    
    - **TF (Term Frequency)**: How many times a word appears in a document
    - **IDF (Inverse Document Frequency)**: How rare a word is across all documents
    - **TF-IDF**: Combines both to give each word an importance score
    
    Think of it like this: Common words like "the" have low TF-IDF scores because they 
    appear everywhere, while specific words have higher scores.
    """)
    
    st.subheader("2️⃣ Word Similarity")
    st.markdown("""
    We find similar words using **Cosine Similarity**, which measures the angle between 
    two word vectors.
    
    - Similarity = 1: Words are identical
    - Similarity = 0: Words are completely different
    - Similarity = -1: Words are opposite
    
    Similar words have similar TF-IDF patterns across documents.
    """)
    
    st.subheader("3️⃣ Dimensionality Reduction")
    st.markdown("""
    Word embeddings can have hundreds of dimensions, but we can only visualize 2D or 3D. 
    We use two techniques:
    
    **PCA (Principal Component Analysis)**
    - Very fast ⚡
    - Linear reduction (keeps straight lines straight)
    - Good for getting a quick overview
    
    **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
    - Slower 🐢
    - Non-linear reduction (can curve straight lines)
    - Better at preserving local structure
    - Great for seeing clusters and patterns
    """)
    
    st.subheader("4️⃣ REST API")
    st.markdown("""
    This app communicates with a Flask REST API that:
    - Trains the embedding model on a corpus
    - Computes embeddings for words
    - Finds similar words
    - Generates visualizations
    """)
    
    st.subheader("Available Endpoints")
    
    endpoints_data = {
        'Method': ['POST', 'POST', 'POST', 'GET', 'POST'],
        'Endpoint': [
            '/api/train',
            '/api/embedding',
            '/api/nearest-neighbors',
            '/api/vocabulary',
            '/api/visualize'
        ],
        'Description': [
            'Train model on corpus',
            'Get embedding for a word',
            'Find similar words',
            'Get vocabulary list',
            'Generate visualization'
        ]
    }
    
    st.dataframe(pd.DataFrame(endpoints_data), use_container_width=True)
    
    st.subheader("Try This!")
    st.markdown("""
    1. Go to the "Explore Words" tab
    2. Try searching for: `python`, `learning`, `data`, `intelligence`
    3. Check the nearest neighbors - do they make sense?
    4. Go to "Visualize Embeddings" and try both PCA and t-SNE
    5. Notice how PCA is much faster but t-SNE creates nicer clusters
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: small;'>
    Built with Streamlit | TF-IDF Embeddings | Flask API
    </div>
    """, unsafe_allow_html=True)
