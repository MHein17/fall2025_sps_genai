import torch
from torchvision import datasets, transforms
import numpy as np

def get_data_loader(data_dir, batch_size=32, train=True):
    # Preprocessing for Fashion-MNIST (pad to 32x32)
    def preprocess(img):
        img = np.pad(img, ((2, 2), (2, 2)), constant_values=0.0)
        return img
    
    transform = transforms.Compose([
        transforms.Lambda(preprocess), 
        transforms.ToTensor()
    ])
    
    # Load Fashion-MNIST dataset
    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    
    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )
    
    return loader