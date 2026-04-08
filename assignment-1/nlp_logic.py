import spacy
import nltk
from nltk.stem import PorterStemmer
from typing import List, Dict, Any

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model is not found, we can try to download it on the fly or log an error
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

stemmer = PorterStemmer()

def tokenize(text: str) -> List[str]:
    doc = nlp(text)
    return [token.text for token in doc]

def lemmatize(text: str) -> List[Dict[str, str]]:
    doc = nlp(text)
    return [{"text": token.text, "lemma": token.lemma_} for token in doc]

def stem(text: str) -> List[Dict[str, str]]:
    # Simple whitespace tokenization for stemming demo
    tokens = nltk.word_tokenize(text)
    return [{"text": token, "stem": stemmer.stem(token)} for token in tokens]

def pos_tagging(text: str) -> List[Dict[str, str]]:
    doc = nlp(text)
    return [{"text": token.text, "pos": token.pos_, "tag": token.tag_, "explanation": spacy.explain(token.tag_)} for token in doc]

def ner(text: str) -> List[Dict[str, str]]:
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_, "explanation": spacy.explain(ent.label_)} for ent in doc.ents]

def compare_stem_lemma(words: List[str]) -> List[Dict[str, str]]:
    results = []
    for word in words:
        # For lemmatization, we need context or at least a single-word doc
        doc = nlp(word)
        lemma = doc[0].lemma_ if len(doc) > 0 else word
        stem_val = stemmer.stem(word)
        results.append({
            "word": word,
            "stem": stem_val,
            "lemma": lemma,
            "is_different": stem_val != lemma
        })
    return results

# Default 10 examples for the comparison task
COMPARISON_EXAMPLES = [
    "running", "flies", "happily", "studies", "better", 
    "feet", "caring", "cars", "was", "universities"
]

def get_comparison_report():
    return compare_stem_lemma(COMPARISON_EXAMPLES)

if __name__ == "__main__":
    test_text = "Apple IS looking AT buying U.K. startup for $1 billion. He was running happily."
    print("Tokens:", tokenize(test_text))
    print("Lemmas:", lemmatize(test_text))
    print("Stems:", stem(test_text))
    print("POS Tags:", pos_tagging(test_text))
    print("NER:", ner(test_text))
    print("Comparison:", get_comparison_report())
