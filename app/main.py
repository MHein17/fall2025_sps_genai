from typing import Union, List
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from .bigram_model import BigramModel
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .helper_lib.model import get_model
import torch
import io
import base64
import zipfile
import time
from PIL import Image
from torchvision.utils import save_image

import os
app = FastAPI()

# Load spaCy model for embeddings
nlp = spacy.load("en_core_web_md")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond DantÃ¨s, who is falsely imprisoned and later seeks revenge.",
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
    seed: int = None

@app.post("/generate_digit")
def generate_mnist_digit(request: DigitGenerationRequest):
    """
    Generate MNIST-style handwritten digit(s) using trained GAN
    
    Parameters:
    - num_digits: Number of digits to generate (default: 1, max: 16)
    - seed: Optional random seed for reproducibility
    
    Returns:
    - Single PNG image if num_digits=1
    - ZIP file with multiple PNGs if num_digits>1
    """
    load_gan_model()
    
    # Limit number of digits
    num_digits = min(max(1, request.num_digits), 16)
    
    # Set seed if provided
    if request.seed is not None:
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)
    
    # Generate random noise
    noise = torch.randn(num_digits, Z_DIM).to(gan_device)
    
    # Generate images
    gan_generator.eval()
    with torch.no_grad():
        generated_images = gan_generator(noise)
    
    # Single image: return PNG directly
    if num_digits == 1:
        img_tensor = generated_images[0]
        img_tensor = (img_tensor + 1) / 2.0
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        buffer = io.BytesIO()
        save_image(img_tensor, buffer, format="PNG")
        buffer.seek(0)
        return Response(content=buffer.getvalue(), media_type="image/png")
    
    # Multiple images: return ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(num_digits):
            img_tensor = generated_images[i]
            img_tensor = (img_tensor + 1) / 2.0
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            b = io.BytesIO()
            save_image(img_tensor, b, format="PNG")
            b.seek(0)
            zf.writestr(f"digit_{i:02d}.png", b.read())
    
    zip_buffer.seek(0)
    filename = f"mnist_digits_{int(time.time())}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)



###### ENERGY-BASED MODEL - CIFAR-10 IMAGE GENERATION
from .energy_model import EnergyModel, generate_samples as energy_generate_samples

# Global variable for energy model
energy_model = None
energy_device = None

def load_energy_model():
    """Load Energy-Based Model lazily on first request"""
    global energy_model, energy_device
    if energy_model is None:
        energy_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        energy_model = EnergyModel().to(energy_device)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'energy_model.pth')
        
        print(f"Loading Energy-Based Model from: {model_path}")
        energy_model.load_state_dict(torch.load(model_path, map_location=energy_device))
        energy_model.eval()
        print("Energy-Based Model loaded successfully")

class EnergyGenerationRequest(BaseModel):
    num_images: int = 1
    steps: int = 256
    step_size: float = 10.0
    noise_std: float = 0.01
    seed: int = None

@app.post("/generate_energy")
def generate_energy_images(request: EnergyGenerationRequest):
    """
    Generate CIFAR-10-style images using trained Energy-Based Model
    
    Parameters:
    - num_images: Number of images to generate (default: 1, max: 16)
    - steps: Number of Langevin dynamics steps (default: 256)
    - step_size: Step size for gradient descent (default: 10.0)
    - noise_std: Standard deviation of noise (default: 0.01)
    - seed: Optional random seed for reproducibility
    
    Returns:
    - Single PNG image if num_images=1
    - ZIP file with multiple PNGs if num_images>1
    """
    load_energy_model()
    
    # Limit number of images
    num_images = min(max(1, request.num_images), 16)
    
    # Set seed if provided
    if request.seed is not None:
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)
    
    # Generate random starting images in [-1, 1]
    x = torch.rand((num_images, 3, 32, 32), device=energy_device) * 2 - 1
    
    # Generate images using Langevin dynamics
    generated_images = energy_generate_samples(
        energy_model, x, 
        steps=request.steps, 
        step_size=request.step_size, 
        noise_std=request.noise_std
    )
    
    # Single image: return PNG directly
    if num_images == 1:
        img_tensor = generated_images[0]
        img_tensor = (img_tensor + 1) / 2.0
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        buffer = io.BytesIO()
        save_image(img_tensor, buffer, format="PNG")
        buffer.seek(0)
        return Response(content=buffer.getvalue(), media_type="image/png")
    
    # Multiple images: return ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(num_images):
            img_tensor = generated_images[i]
            img_tensor = (img_tensor + 1) / 2.0
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            b = io.BytesIO()
            save_image(img_tensor, b, format="PNG")
            b.seek(0)
            zf.writestr(f"energy_{i:02d}.png", b.read())
    
    zip_buffer.seek(0)
    filename = f"energy_images_{int(time.time())}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)



###### DIFFUSION MODEL - CIFAR-10 IMAGE GENERATION
from .diffusion_model import DiffusionModel, UNet, offset_cosine_diffusion_schedule

# Global variable for diffusion model
diffusion_model = None
diffusion_device = None

def load_diffusion_model():
    """Load Diffusion Model lazily on first request"""
    global diffusion_model, diffusion_device
    if diffusion_model is None:
        diffusion_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create UNet and Diffusion Model
        unet = UNet(image_size=32, num_channels=3, embedding_dim=32)
        diffusion_model = DiffusionModel(unet, offset_cosine_diffusion_schedule)
        diffusion_model.to(diffusion_device)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'diffusion_model.pth')
        
        print(f"Loading Diffusion Model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=diffusion_device)
        
        # Load model state
        diffusion_model.network.load_state_dict(checkpoint['model_state_dict'])
        diffusion_model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
        # Fix: Move normalizer tensors to correct device
        diffusion_model.normalizer_mean = checkpoint['normalizer_mean'].to(diffusion_device)
        diffusion_model.normalizer_std = checkpoint['normalizer_std'].to(diffusion_device)
        
        diffusion_model.eval()
        print("Diffusion Model loaded successfully")

class DiffusionGenerationRequest(BaseModel):
    num_images: int = 1
    diffusion_steps: int = 50
    seed: int = None

@app.post("/generate_diffusion")
def generate_diffusion_images(request: DiffusionGenerationRequest):
    """
    Generate CIFAR-10-style images using trained Diffusion Model
    
    Parameters:
    - num_images: Number of images to generate (default: 1, max: 16)
    - diffusion_steps: Number of reverse diffusion steps (default: 50)
    - seed: Optional random seed for reproducibility
    
    Returns:
    - Single PNG image if num_images=1
    - ZIP file with multiple PNGs if num_images>1
    """
    load_diffusion_model()
    
    # Limit number of images
    num_images = min(max(1, request.num_images), 16)
    
    # Set seed if provided
    if request.seed is not None:
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)
    
    # Generate images
    diffusion_model.eval()
    generated_images = diffusion_model.generate(
        num_images=num_images,
        diffusion_steps=request.diffusion_steps,
        image_size=32
    )
    
    # Single image: return PNG directly
    if num_images == 1:
        img_tensor = generated_images[0]
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        buffer = io.BytesIO()
        save_image(img_tensor, buffer, format="PNG")
        buffer.seek(0)
        return Response(content=buffer.getvalue(), media_type="image/png")
    
    # Multiple images: return ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(num_images):
            img_tensor = generated_images[i]
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            b = io.BytesIO()
            save_image(img_tensor, b, format="PNG")
            b.seek(0)
            zf.writestr(f"diffusion_{i:02d}.png", b.read())
    
    zip_buffer.seek(0)
    filename = f"diffusion_images_{int(time.time())}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)
