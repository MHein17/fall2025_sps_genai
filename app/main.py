from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel
from .bigram_model import BigramModel
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .helper_lib.model import get_model
import torch
import io
import base64
from PIL import Image

import os
app = FastAPI()

# Load spaCy model for embeddings
nlp = spacy.load("en_core_web_md")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class WordEmbeddingRequest(BaseModel):
    word: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

class LinearAlgebraRequest(BaseModel):
    word1: str
    word2: str
    word3: str
    word4: str

class SentenceSimilarityRequest(BaseModel):
    query: str
    sentences: List[str]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_word_embedding(request: WordEmbeddingRequest):
    """Get word embedding vector for a single word"""
    def calculate_embedding(input_word):
        word = nlp(input_word)
        return word.vector
    
    embedding = calculate_embedding(request.word)
    
    return {
        "word": request.word,
        "embedding": embedding.tolist(),
        "vector_size": len(embedding),
        "first_10_elements": embedding[:10].tolist()
    }

@app.post("/similarity")
def calculate_word_similarity(request: SimilarityRequest):
    """Calculate similarity between two words"""
    def calculate_similarity(word1, word2):
        return nlp(word1).similarity(nlp(word2))
    
    similarity = calculate_similarity(request.word1, request.word2)
    
    return {
        "word1": request.word1,
        "word2": request.word2,
        "similarity": float(similarity)
    }

@app.post("/linear_algebra")
def word_linear_algebra(request: LinearAlgebraRequest):
    """Perform linear algebra: word1 + word2 - word3, compare with word4"""
    la_word1_embedding = nlp(request.word1).vector
    la_word2_embedding = nlp(request.word2).vector
    la_word3_embedding = nlp(request.word3).vector
    la_word4_embedding = nlp(request.word4).vector
    
    # Calculate: word1 + word2 - word3
    la_word = la_word1_embedding + (la_word2_embedding - la_word3_embedding)
    
    # Calculate cosine similarity with word4
    cosine_sim = cosine_similarity([la_word], [la_word4_embedding])[0][0]
    
    return {
        "operation": f"{request.word1} + {request.word2} - {request.word3}",
        "compared_to": request.word4,
        "cosine_similarity": float(cosine_sim),
        "description": f"How similar is '{request.word1} + {request.word2} - {request.word3}' to '{request.word4}'"
    }

@app.post("/sentence_similarity")
def sentence_similarity(request: SentenceSimilarityRequest):
    """Compare a query sentence with multiple target sentences"""
    query_doc = nlp(request.query)
    results = []
    
    for i, sentence in enumerate(request.sentences):
        sentence_doc = nlp(sentence)
        similarity = query_doc.similarity(sentence_doc)
        results.append({
            "sentence": sentence,
            "similarity": float(similarity),
            "rank": i + 1
        })
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        "query": request.query,
        "results": results
    }



###### VAE MODEL


# Global variable for the VAE model
vae = None
device = None

def load_vae_model():
    """Load VAE model lazily on first request"""
    global vae, device
    if vae is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae = get_model("VAE").to(device)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'vae_model.pth')
        
        print(f"Loading VAE model from: {model_path}")
        vae.load_state_dict(torch.load(model_path, map_location=device))
        vae.eval()
        print("VAE model loaded successfully")

class VAEGenerateRequest(BaseModel):
    num_samples: int = 10

@app.post("/vae/generate")
def generate_vae_samples(request: VAEGenerateRequest):
    """Generate samples from the trained VAE"""
    load_vae_model()  # Load model on first request
    
    vae.eval()
    with torch.no_grad():
        z = torch.randn(request.num_samples, 2).to(device)
        samples = vae.decoder(z).cpu().numpy()
    
    images_base64 = []
    for i in range(request.num_samples):
        img_array = (samples[i].squeeze() * 255).astype('uint8')
        img = Image.fromarray(img_array, mode='L')
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_base64.append(img_str)
    
    return {
        "num_samples": request.num_samples,
        "images": images_base64
    }