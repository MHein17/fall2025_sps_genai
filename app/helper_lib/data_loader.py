import torch
from torchvision import datasets, transforms
import numpy as np

def get_data_loader(data_dir, batch_size=32, train=True, dataset_name='CIFAR10'):
    """
    Get data loader for different datasets
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        train: Whether to load training or test set
        dataset_name: 'CIFAR10' (32x32) or 'CIFAR10_64' (64x64) or 'FashionMNIST'
    """
    if dataset_name == 'FashionMNIST':
        # Preprocessing for Fashion-MNIST (pad to 32x32)
        def preprocess(img):
            img = np.pad(img, ((2, 2), (2, 2)), constant_values=0.0)
            return img
        
        transform = transforms.Compose([
            transforms.Lambda(preprocess), 
            transforms.ToTensor()
        ])
        
        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
    
    elif dataset_name == 'CIFAR10':
        # Standard CIFAR-10 (32x32)
        transform = transforms.Compose([transforms.ToTensor()])
        
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
    
    elif dataset_name == 'CIFAR10_64':
        # CIFAR-10 resized to 64x64 for Assignment 2
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        dataset = datasets.CIFAR10(
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