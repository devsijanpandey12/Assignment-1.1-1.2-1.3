# Line-by-Line Code Explanation: NLP Preprocessing Basics

This document provides a granular explanation of every script in the project.

---

## 1. [nlp_logic.py](file:///c:/Users/sijan/Desktop/NLP_Assignmnet_1/nlp_logic.py) (Core NLP Logic)

This file contains the functions that perform the actual NLP tasks.

| Line(s) | Code | Explanation |
| :--- | :--- | :--- |
| 1-3 | `import spacy`, `import nltk`, `from nltk.stem import PorterStemmer` | Modules needed for NLP operations. `spacy` for advanced tasks, `nltk` specifically for stemming. |
| 4 | `from typing import List, Dict, Any` | Used for type hinting, making the code more readable and robust. |
| 7-14 | `try: nlp = spacy.load(...) except OSError: ...` | **Robust Loading**: Attempts to load the English model. If not found, it automatically runs a shell command to download it, ensuring the script doesn't crash on first run. |
| 16 | `stemmer = PorterStemmer()` | Initializes the NLTK stemmer object once globally to save resources. |
| 18-20 | `def tokenize(text: str) -> List[str]:` | **Tokenization**: Takes a string, passes it through the `nlp` pipeline, and returns a list of individual words/punctuation (tokens). |
| 22-24 | `def lemmatize(text: str) -> List[Dict[str, str]]:` | **Lemmatization**: Returns a list of dictionaries. For each token, it includes the original text and its base dictionary form (`lemma_`). |
| 26-29 | `def stem(text: str) -> List[Dict[str, str]]:` | **Stemming**: Since stemming is more primitive, we first tokenize using NLTK (`word_tokenize`) and then apply the [stem()](file:///c:/Users/sijan/Desktop/NLP_Assignmnet_1/nlp_logic.py#26-30) method to each word. |
| 31-33 | `def pos_tagging(text: str) -> List[Dict[str, str]]:` | **POS Tagging**: Assigns Part-of-Speech tags. [pos_](file:///c:/Users/sijan/Desktop/NLP_Assignmnet_1/nlp_logic.py#31-34) is the coarse-grained tag (e.g., NOUN), `tag_` is fine-grained (e.g., NNS for plural noun). `spacy.explain()` gives the definition. |
| 35-37 | `def ner(text: str) -> List[Dict[str, str]]:` | **NER**: Extracts entities like Names or Dates using `doc.ents`. Returns the text and its label (e.g., PERSON, GPE). |
| 39-51 | `def compare_stem_lemma(...)` | **Comparison Logic**: Takes a list of words, computes both stem and lemma for each, and flags if they are different. This is specific to the assignment's comparison requirement. |
| 54-57 | `COMPARISON_EXAMPLES = [...]` | A static list of 10 words (running, flies, etc.) as required by the assignment. |

---

## 2. [api.py](file:///c:/Users/sijan/Desktop/NLP_Assignmnet_1/api.py) (REST API Backend)

This file exposes the logic via HTTP so the UI can communicate with it.

| Line(s) | Code | Explanation |
| :--- | :--- | :--- |
| 1 | `from fastapi import FastAPI, HTTPException` | Initializes the web framework. |
| 2 | `from pydantic import BaseModel` | Used to define data structures for the API (Data Validation). |
| 6 | `app = FastAPI(...)` | Creates the main application instance with a title and description. |
| 8-9 | `class TextRequest(BaseModel): text: str` | **Schema**: Defines that any POST request must include a JSON object with a string field named `text`. |
| 14-16 | `@app.get("/")` | The root endpoint. Returns a simple welcome message when you visit `http://localhost:8000/`. |
| 18-36 | `@app.post("/tokenize")`, etc. | **Endpoints**: Each function calls the corresponding logic from [nlp_logic.py](file:///c:/Users/sijan/Desktop/NLP_Assignmnet_1/nlp_logic.py). The `@app.post` decorator means they only accept POST requests. |
| 38-41 | `@app.get("/compare_default")` | A helper endpoint that specifically returns the 10-word comparison report. |
| 43-45 | `if __name__ == "__main__": uvicorn.run(...)` | Enables running the API directly using `python api.py`. **Uvicorn** is the server that hosts the FastAPI app. |

---

## 3. [app.py](file:///c:/Users/sijan/Desktop/NLP_Assignmnet_1/app.py) (Web Frontend)

This file builds the interactive user interface.

| Line(s) | Code | Explanation |
| :--- | :--- | :--- |
| 1-3 | `import streamlit as st`, `import requests`, `import pandas as pd` | `streamlit` for UI, `requests` to talk to the API, `pandas` to format data into nice tables. |
| 6 | `API_URL = "http://127.0.0.1:8000"` | Points the UI to where the backend API is running. |
| 8 | `st.set_page_config(...)` | Sets the browser tab title and favicon. |
| 17-19 | `page = st.sidebar.radio(...)` | Creates a lateral menu to switch between the "Demo" and "Comparison" pages. |
| 23-24 | `text_input = st.text_area(...)` | Creates a large text input box where users can type. |
| 26 | `if st.button("Process Text"):` | **Trigger**: The code inside this block only runs when the user clicks the button. |
| 30-38 | `res_tokens = requests.post(...)` | **API Integration**: The frontend sends the user's text to the backend API and waits for the JSON response. |
| 40-62 | `col1, col2 = st.columns(2)` | **Layout**: Splits the screen into two columns. Each column displays different results (Tokenization, Lemmatization, etc.) using `st.table()`. |
| 76-80 | `res_compare = requests.get(...).json()` | Fetches the pre-calculated 10-word comparison from the API. |
| 81 | `df_compare = pd.DataFrame(...)` | Converts the JSON data into a Table (DataFrame) for display. |
| 91-103 | `st.markdown("### Analysis...")` | Static text that explains the results for the user, fulfilling the "explain the differences" part of the assignment. |

---

## 4. `requirements.txt` (Dependencies)

While not a script, it lists everything needed:
- `spacy`: Modern NLP.
- `nltk`: Stemming and basic tokenization.
- `fastapi`: Web API framework.
- `uvicorn`: API server.
- `streamlit`: User Interface.
- `pandas`: Data manipulation (Tables).
