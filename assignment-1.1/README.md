# NLP Preprocessing Basics

A comprehensive demonstration of core Natural Language Processing (NLP) preprocessing techniques using **FastAPI** (Backend) and **Streamlit** (Frontend).

## 🎯 Project Overview

This project showcases essential NLP preprocessing techniques:
- **Tokenization**: Breaking text into individual tokens (words/punctuation)
- **Lemmatization**: Reducing words to their base dictionary form
- **Stemming**: Reducing words to their root form using the Porter Stemmer
- **POS Tagging**: Assigning Part-of-Speech tags to tokens
- **Named Entity Recognition (NER)**: Identifying and extracting named entities (PERSON, GPE, DATE, etc.)

## 📁 Project Structure

```
assignment-1.1/
├── api.py                              # FastAPI backend with REST endpoints
├── app.py                              # Streamlit interactive frontend
├── nlp_logic.py                        # Core NLP processing functions
├── detailed_code_explanation.md        # Line-by-line code documentation
├── NLP_Project_Complete_Explanation.pdf # Comprehensive project guide
└── README.md                           # This file
```

## 📋 File Descriptions

### `nlp_logic.py` (Core NLP Logic)
Contains all NLP functions:
- `tokenize(text)`: Returns list of tokens
- `lemmatize(text)`: Returns tokens with their lemmas
- `stem(text)`: Returns tokens with their stems
- `pos_tagging(text)`: Returns tokens with POS tags and explanations
- `ner(text)`: Returns named entities with labels
- `compare_stem_lemma(words)`: Compares stemming vs lemmatization for a word list

**Key Features:**
- Automatically downloads the spaCy English model if not found
- Uses spaCy for advanced NLP tasks
- Uses NLTK for stemming (Porter Stemmer)

### `api.py` (REST API Backend)
FastAPI application that exposes NLP functions as HTTP endpoints:
- `GET /`: Welcome message
- `POST /tokenize`: Tokenize input text
- `POST /lemmatize`: Lemmatize text
- `POST /stem`: Stem text
- `POST /pos`: Perform POS tagging
- `POST /ner`: Extract named entities
- `GET /compare_default`: Get comparison of 10 predefined words

**Server Details:**
- Runs on `http://127.0.0.1:8000`
- Interactive API docs available at `http://127.0.0.1:8000/docs`

### `app.py` (Streamlit Frontend)
Interactive web interface for the NLP preprocessing system:
- **Interactive Demo**: Enter custom text and visualize all preprocessing techniques
- **Stemming vs Lemmatization Comparison**: Compare the differences between stemming and lemmatization

**Features:**
- Real-time text processing
- Formatted output in tables
- Navigation sidebar
- Clean, user-friendly interface

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Install required packages:**
```bash
pip install fastapi uvicorn streamlit requests pandas spacy nltk
```

2. **Download required NLP models:**
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

### Running the Application

#### Option 1: Run Both Backend and Frontend

1. **Start the FastAPI backend** (in one terminal):
```bash
python api.py
```
The API will be available at `http://127.0.0.1:8000`

2. **Start the Streamlit frontend** (in another terminal):
```bash
streamlit run app.py
```
The web interface will open in your browser (typically at `http://localhost:8501`)

#### Option 2: Using FastAPI Directly

Access interactive API documentation:
```
http://127.0.0.1:8000/docs
```

## 💡 Usage Examples

### Using the Web Interface
1. Enter text in the "Interactive Demo" section
2. Click "Process Text"
3. View results for tokenization, lemmatization, stemming, POS tagging, and NER

### Using the API with curl

**Tokenize:**
```bash
curl -X POST "http://127.0.0.1:8000/tokenize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple is looking at buying U.K. startup"}'
```

**Lemmatize:**
```bash
curl -X POST "http://127.0.0.1:8000/lemmatize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Running flies quickly"}'
```

**NER:**
```bash
curl -X POST "http://127.0.0.1:8000/ner" \
  -H "Content-Type: application/json" \
  -d '{"text": "John works at Apple in California"}'
```

## 📚 Documentation

For detailed line-by-line code explanations, see [detailed_code_explanation.md](detailed_code_explanation.md)

For comprehensive project documentation, see [NLP_Project_Complete_Explanation.pdf](NLP_Project_Complete_Explanation.pdf)

## 🔍 Key Concepts

### Tokenization
Breaking text into individual tokens (words, punctuation)
```
"Apple is great" → ["Apple", "is", "great"]
```

### Lemmatization
Reducing words to their base form using dictionary lookup
```
"running" → "run", "flies" → "fly"
```

### Stemming
Reducing words to their root form using rule-based algorithms
```
"running" → "run", "flies" → "fli"
```

### POS Tagging
Assigning grammatical roles (NOUN, VERB, ADJ, etc.)
```
"Apple" → NOUN, "is" → AUX, "great" → ADJ
```

### NER
Identifying named entities like people, places, organizations
```
"John works at Apple in New York"
→ PERSON: John, ORG: Apple, GPE: New York
```

## 🛠️ Technologies Used

- **FastAPI**: Modern, fast web framework for building APIs
- **Streamlit**: Simple framework for creating data web apps
- **spaCy**: Industrial-strength NLP library
- **NLTK**: Natural Language Toolkit for NLP tasks
- **Pandas**: Data manipulation and formatting
- **Uvicorn**: ASGI server implementation

