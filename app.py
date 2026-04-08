import streamlit as st
import requests
import pandas as pd

# API URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="NLP Preprocessing Basics", page_icon="📝", layout="wide")

st.title("📝 NLP Preprocessing Basics")
st.markdown("""
This app demonstrates core NLP preprocessing techniques: **Tokenization**, **Lemmatization**, **Stemming**, **POS Tagging**, and **NER**.
Built with **FastAPI** (Backend) and **Streamlit** (Frontend).
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Interactive Demo", "Stemming vs Lemmatization Comparison"])

if page == "Interactive Demo":
    st.header("Interactive NLP Demo")
    text_input = st.text_area("Enter text to process:", "Apple is looking at buying U.K. startup for $1 billion.")
    
    if st.button("Process Text"):
        if text_input.strip():
            try:
                # Tokenization
                res_tokens = requests.post(f"{API_URL}/tokenize", json={"text": text_input}).json()
                # Lemmatization
                res_lemmas = requests.post(f"{API_URL}/lemmatize", json={"text": text_input}).json()
                # Stemming
                res_stems = requests.post(f"{API_URL}/stem", json={"text": text_input}).json()
                # POS Tagging
                res_pos = requests.post(f"{API_URL}/pos", json={"text": text_input}).json()
                # NER
                res_ner = requests.post(f"{API_URL}/ner", json={"text": text_input}).json()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("1. Tokenization")
                    st.write(res_tokens["tokens"])

                    st.subheader("2. Lemmatization")
                    df_lemmas = pd.DataFrame(res_lemmas["lemmas"])
                    st.table(df_lemmas)

                with col2:
                    st.subheader("3. Stemming (Porter)")
                    df_stems = pd.DataFrame(res_stems["stems"])
                    st.table(df_stems)

                    st.subheader("4. POS Tagging")
                    df_pos = pd.DataFrame(res_pos["pos_tags"])
                    st.table(df_pos)

                st.subheader("5. Named Entity Recognition (NER)")
                if res_ner["entities"]:
                    df_ner = pd.DataFrame(res_ner["entities"])
                    st.table(df_ner)
                else:
                    st.info("No named entities found.")

            except Exception as e:
                st.error(f"Error connecting to API: {e}. Make sure the FastAPI server is running at {API_URL}")
        else:
            st.warning("Please enter some text.")

elif page == "Stemming vs Lemmatization Comparison":
    st.header("Stemming vs. Lemmatization Comparison")
    
    st.markdown("""
    ### Key Differences:
    - **Stemming**: A rule-based process that chops off the ends of words in the hope of reaching the root. It often produces non-dictionary words (e.g., 'flies' -> 'fli'). It is faster but less accurate.
    - **Lemmatization**: A linguistics-based process that uses a vocabulary and morphological analysis of words to return the base or dictionary form (lemma). It requires more computation and often depends on POS context.
    """)

    try:
        res_compare = requests.get(f"{API_URL}/compare_default").json()
        df_compare = pd.DataFrame(res_compare["comparison"])
        
        # Highlight differences
        def highlight_diff(s):
            return ['background-color: #ffcccc' if s.is_different else '' for _ in s]
        
        st.subheader("Examples Comparison (10 Words)")
        st.table(df_compare)

        st.markdown("""
        ### Analysis of the Examples:
        1. **running**: Both identify the root ('run'), but stemming might keep suffixes in some cases.
        2. **flies**: Stemming results in 'fli' (non-word), while lemmatization correctly finds 'fly'.
        3. **happily**: Stemming gives 'happili', lemmatization gives 'happily' (as it's an adverb).
        4. **studies**: Stemming -> 'studi', Lemmatization -> 'study'.
        5. **better**: Stemming -> 'better', Lemmatization -> 'well' (spaCy identifies 'better' as the comparative form of 'well/good').
        6. **feet**: Stemming -> 'feet', Lemmatization -> 'foot' (Lemmatization handles irregular plurals).
        7. **caring**: Stemming -> 'care', Lemmatization -> 'care'.
        8. **cars**: Both reach 'car'.
        9. **was**: Stemming -> 'wa' (usually), Lemmatization -> 'be' (correctly identifies the base verb).
        10. **universities**: Stemming -> 'univers', Lemmatization -> 'university'.
        """)

    except Exception as e:
        st.error(f"Error connecting to API: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Assignment 1.1 - NLP Preprocessing Basics")
