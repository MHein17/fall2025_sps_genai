import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from energy_model import EnergyModel, EBM
from diffusion_model import DiffusionModel, UNet, offset_cosine_diffusion_schedule

def quick_train_energy(epochs=2, batch_size=128, device='cuda'):
    """Quick Energy Model training for demo (20 minutes)"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    nn_energy_model = EnergyModel().to(device)
    ebm = EBM(nn_energy_model, alpha=0.1, steps=60, step_size=10, noise=0.005, device=device)
    optimizer = torch.optim.Adam(nn_energy_model.parameters(), lr=0.0001, betas=(0.0, 0.999))
    
    for epoch in range(epochs):
        ebm.reset_metrics()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(device)
            metrics = ebm.train_step(real_imgs, optimizer)
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
        print(f"Epoch {epoch+1} - " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    
    os.makedirs('models', exist_ok=True)
    torch.save(nn_energy_model.state_dict(), 'models/energy_model.pth')
    print("Energy model saved to models/energy_model.pth")

def quick_train_diffusion(epochs=10, batch_size=64, device='cuda'):
    """Quick Diffusion Model training for demo (1 hour)"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Quick normalization stats calculation (sample only 20 batches)
    print("Calculating normalization statistics (quick)...")
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    for i, (imgs, _) in enumerate(train_loader):
        if i >= 20:  # Only use 20 batches for speed
            break
        batch_size_actual = imgs.size(0)
        imgs_flat = imgs.view(batch_size_actual, 3, -1)
        batch_mean = imgs_flat.mean(dim=(0, 2))
        batch_std = imgs_flat.std(dim=(0, 2))
        mean += batch_mean * batch_size_actual
        std += batch_std * batch_size_actual
        total_samples += batch_size_actual
    
    mean /= total_samples
    std /= total_samples
    print("Stats - Mean:", mean, "Std:", std)
    mean = mean.reshape(1, 3, 1, 1).to(device)
    std = std.reshape(1, 3, 1, 1).to(device)
    
    unet = UNet(image_size=32, num_channels=3, embedding_dim=32)
    diffusion_model = DiffusionModel(unet, offset_cosine_diffusion_schedule)
    diffusion_model.to(device)
    diffusion_model.set_normalizer(mean, std)
    
    optimizer = torch.optim.AdamW(diffusion_model.network.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    
    os.makedirs('models', exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        diffusion_model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)
            loss = diffusion_model.train_step(images, optimizer, loss_fn)
            train_losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': diffusion_model.network.state_dict(),
            'ema_model_state_dict': diffusion_model.ema_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'normalizer_mean': diffusion_model.normalizer_mean,
            'normalizer_std': diffusion_model.normalizer_std
        }
        
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            torch.save(checkpoint, 'models/diffusion_model.pth')
            print(f"Best model saved with loss: {avg_train_loss:.4f}")
    
    print("Diffusion model saved to models/diffusion_model.pth")

if __name__ == '__main__':
    print("Training Energy Model...")
    quick_train_energy(epochs=2, batch_size=128)
    
    print("\nTraining Diffusion Model...")
    quick_train_diffusion(epochs=10, batch_size=64)
    
    print("\nTraining complete. Models saved in models/")

