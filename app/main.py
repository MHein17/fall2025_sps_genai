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



###### CLASSIFIER MODEL
# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Global variable for classifier
classifier = None
classifier_device = None

def load_classifier_model():
    """Load classifier model lazily on first request"""
    global classifier, classifier_device
    if classifier is None:
        classifier_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier = get_model("Assignment2CNN").to(classifier_device)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'classifier_model.pth')
        
        print(f"Loading classifier model from: {model_path}")
        classifier.load_state_dict(torch.load(model_path, map_location=classifier_device))
        classifier.eval()
        print("Classifier model loaded successfully")

class ImageClassifyRequest(BaseModel):
    image_base64: str

@app.post("/classify")
def classify_image(request: ImageClassifyRequest):
    """Classify a 64x64 RGB image"""
    load_classifier_model()
    
    # Decode base64 image
    img_data = base64.b64decode(request.image_base64)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize((64, 64))
    
    # Convert to tensor
    import numpy as np
    img_array = np.array(img).transpose(2, 0, 1) / 255.0
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).to(classifier_device)
    
    # Predict
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "predicted_class": CIFAR10_CLASSES[predicted_class],
        "class_index": predicted_class,
        "confidence": float(confidence),
        "all_probabilities": {
            CIFAR10_CLASSES[i]: float(probabilities[0][i]) 
            for i in range(10)
        }
    }



###### GAN MODEL - MNIST DIGIT GENERATION
from .mnist_gan import Generator

# Global variable for GAN generator
gan_generator = None
gan_device = None
Z_DIM = 100

def load_gan_model():
    """Load GAN generator model lazily on first request"""
    global gan_generator, gan_device
    if gan_generator is None:
        gan_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gan_generator = Generator(z_dim=Z_DIM).to(gan_device)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'generator.pth')
        
        print(f"Loading GAN generator from: {model_path}")
        gan_generator.load_state_dict(torch.load(model_path, map_location=gan_device))
        gan_generator.eval()
        print("GAN generator loaded successfully")

class DigitGenerationRequest(BaseModel):
    num_digits: int = 1
    seed: int = None  # Optional seed for reproducibility

@app.post("/generate_digit")
def generate_mnist_digit(request: DigitGenerationRequest):
    """
    Generate MNIST-style handwritten digit(s) using trained GAN
    
    Parameters:
    - num_digits: Number of digits to generate (default: 1, max: 16)
    - seed: Optional random seed for reproducibility
    
    Returns:
    - images: List of base64-encoded PNG images
    - num_generated: Number of images generated
    """
    load_gan_model()
    
    # Limit number of digits
    num_digits = min(max(1, request.num_digits), 16)
    
    # Set seed if provided
    if request.seed is not None:
        torch.manual_seed(request.seed)
    
    # Generate random noise
    noise = torch.randn(num_digits, Z_DIM).to(gan_device)
    
    # Generate images
    gan_generator.eval()
    with torch.no_grad():
        generated_images = gan_generator(noise)
    
    # Convert tensors to base64 images
    images_base64 = []
    for i in range(num_digits):
        # Get single image tensor
        img_tensor = generated_images[i]
        
        # Denormalize from [-1, 1] to [0, 1]
        img_tensor = (img_tensor + 1) / 2.0
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to PIL Image
        img_array = img_tensor.cpu().squeeze().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_array, mode='L')
        
        # Convert to base64
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        images_base64.append(img_base64)
    
    return {
        "images": images_base64,
        "num_generated": num_digits,
        "image_size": "28x28",
        "model": "MNIST GAN",
        "seed_used": request.seed
    }
