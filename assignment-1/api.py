from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import nlp_logic

app = FastAPI(title="NLP Preprocessing API", description="REST API for core NLP preprocessing tasks.")

class TextRequest(BaseModel):
    text: str

class CompareRequest(BaseModel):
    words: List[str] = nlp_logic.COMPARISON_EXAMPLES

@app.get("/")
async def root():
    return {"message": "Welcome to the NLP Preprocessing API. Use /docs to see endpoints."}

@app.post("/tokenize")
async def tokenize(request: TextRequest):
    return {"tokens": nlp_logic.tokenize(request.text)}

@app.post("/lemmatize")
async def lemmatize(request: TextRequest):
    return {"lemmas": nlp_logic.lemmatize(request.text)}

@app.post("/stem")
async def stem(request: TextRequest):
    return {"stems": nlp_logic.stem(request.text)}

@app.post("/pos")
async def pos(request: TextRequest):
    return {"pos_tags": nlp_logic.pos_tagging(request.text)}

@app.post("/ner")
async def ner(request: TextRequest):
    return {"entities": nlp_logic.ner(request.text)}

@app.post("/compare")
async def compare(request: CompareRequest):
    return {"comparison": nlp_logic.compare_stem_lemma(request.words)}

@app.get("/compare_default")
async def compare_default():
    return {"comparison": nlp_logic.get_comparison_report()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
