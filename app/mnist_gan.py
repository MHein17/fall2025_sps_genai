"""
MNIST GAN Implementation for Assignment 3
Following the architecture specifications from the assignment document
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm


# Check which device is available
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")


class Generator(nn.Module):
    """
    Generator Network for MNIST
    Architecture as per Assignment 3 requirements:
    - Input: Noise vector of shape (BATCH_SIZE, 100)
    - Fully connected layer to 7 × 7 × 128, then reshape
    - ConvTranspose2D: 128 → 64, kernel size 4, stride 2, padding 1 ⇒ output size 14 × 14
      - Followed by BatchNorm2D and ReLU
    - ConvTranspose2D: 64 → 1, kernel size 4, stride 2, padding 1 ⇒ output size 28 × 28
      - Followed by Tanh activation
    """
    
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        # Fully connected layer to 7x7x128
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        
        # ConvTranspose2d layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(True)
        
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # x shape: (batch_size, z_dim)
        x = self.fc(x)
        # Reshape to (batch_size, 128, 7, 7)
        x = x.view(x.size(0), 128, 7, 7)
        
        # First ConvTranspose: 7x7 -> 14x14
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Second ConvTranspose: 14x14 -> 28x28
        x = self.deconv2(x)
        x = self.tanh(x)
        
        return x


class Discriminator(nn.Module):
    """
    Discriminator Network for MNIST
    Architecture as per Assignment 3 requirements:
    - Input: Image of shape (1, 28, 28)
    - Conv2D: 1 → 64, kernel size 4, stride 2, padding 1 ⇒ output size 14 × 14
      - Followed by LeakyReLU(0.2)
    - Conv2D: 64 → 128, kernel size 4, stride 2, padding 1 ⇒ output size 7 × 7
      - Followed by BatchNorm2D and LeakyReLU(0.2)
    - Flatten and apply Linear layer to get a single output (real/fake probability)
    """
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # First Conv2d layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        # Second Conv2d layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        # Flatten and Linear layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        # Note: No sigmoid here - will use BCEWithLogitsLoss which includes sigmoid
    
    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        
        # First conv: 28x28 -> 14x14
        x = self.conv1(x)
        x = self.act1(x)
        
        # Second conv: 14x14 -> 7x7
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # Flatten and linear (returns logits, not probabilities)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x


def load_mnist_data(batch_size=64, data_dir='./data'):
    """
    Load MNIST dataset with appropriate transforms
    Following Module 6 pattern but for MNIST
    """
    # Transform: Normalize to [-1, 1] for Tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Download and load MNIST dataset
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return dataloader, dataset


def train_gan(epochs=20, batch_size=64, z_dim=100, lr=0.0002, data_dir='./data', save_path='./models'):
    """
    Train the GAN model
    Following Module 6 training pattern but with standard GAN loss (Binary Cross Entropy)
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Load data
    print("Loading MNIST dataset...")
    dataloader, dataset = load_mnist_data(batch_size=batch_size, data_dir=data_dir)
    print(f"Dataset loaded: {len(dataset)} images")
    
    # Initialize models
    generator = Generator(z_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers (using Adam like standard GAN, not RMSprop like WGAN)
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss function (BCEWithLogitsLoss combines Sigmoid + BCE for numerical stability)
    criterion = nn.BCEWithLogitsLoss()
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, z_dim).to(device)
    
    # Training logs
    datalogs = []
    
    print("\nStarting training...")
    for epoch in range(epochs):
        train_loader_with_progress = tqdm(
            iterable=dataloader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        
        for batch_number, (real_images, _) in enumerate(train_loader_with_progress):
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            
            # Labels for real and fake images
            real_labels = torch.ones(batch_size_actual, 1).to(device)
            fake_labels = torch.zeros(batch_size_actual, 1).to(device)
            
            ## === Train Discriminator === ##
            discriminator.zero_grad()
            
            # Train on real images
            real_output = discriminator(real_images)
            loss_real = criterion(real_output, real_labels)
            
            # Train on fake images
            noise = torch.randn(batch_size_actual, z_dim).to(device)
            fake_images = generator(noise).detach()
            fake_output = discriminator(fake_images)
            loss_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            loss_disc = loss_real + loss_fake
            loss_disc.backward()
            opt_disc.step()
            
            ## === Train Generator === ##
            generator.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size_actual, z_dim).to(device)
            fake_images = generator(noise)
            
            # Try to fool discriminator
            fake_output = discriminator(fake_images)
            loss_gen = criterion(fake_output, real_labels)  # Want discriminator to think these are real
            
            loss_gen.backward()
            opt_gen.step()
            
            # Logging
            if batch_number % 100 == 0:
                train_loader_with_progress.set_postfix({
                    "D loss": f"{loss_disc.item():.4f}",
                    "G loss": f"{loss_gen.item():.4f}",
                })
                
                datalogs.append({
                    "epoch": epoch + batch_number / len(dataloader),
                    "D loss": loss_disc.item(),
                    "G loss": loss_gen.item(),
                })
        
        # Generate and save sample images after each epoch
        with torch.no_grad():
            generator.eval()
            fake = generator(fixed_noise).detach().cpu()
            generator.train()
        
        grid = make_grid(fake, normalize=True, nrow=8)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
        plt.title(f"Epoch {epoch+1}")
        plt.axis("off")
        plt.savefig(f"{save_path}/epoch_{epoch+1}.png")
        plt.close()
        
        print(f"Epoch {epoch+1}/{epochs} completed. Sample saved.")
    
    # Save trained models
    torch.save(generator.state_dict(), f"{save_path}/generator.pth")
    torch.save(discriminator.state_dict(), f"{save_path}/discriminator.pth")
    print(f"\nModels saved to {save_path}/")
    
    return generator, discriminator, datalogs


def generate_digit(generator, z_dim=100, num_images=1):
    """
    Generate digit(s) using the trained generator
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, z_dim).to(device)
        generated_images = generator(noise)
    
    return generated_images


def visualize_generated_digits(generator, z_dim=100, num_images=16):
    """
    Visualize generated digits
    """
    images = generate_digit(generator, z_dim, num_images)
    grid = make_grid(images.cpu(), normalize=True, nrow=4)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.title("Generated MNIST Digits")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Print model architectures
    print("\n" + "="*50)
    print("Generator Architecture:")
    print("="*50)
    gen = Generator(z_dim=100)
    print(gen)
    
    print("\n" + "="*50)
    print("Discriminator Architecture:")
    print("="*50)
    disc = Discriminator()
    print(disc)
    
    # Test with sample input
    print("\n" + "="*50)
    print("Testing forward pass:")
    print("="*50)
    sample_noise = torch.randn(1, 100)
    sample_output = gen(sample_noise)
    print(f"Generator output shape: {sample_output.shape}")
    
    sample_image = torch.randn(1, 1, 28, 28)
    sample_disc_output = disc(sample_image)
    print(f"Discriminator output shape: {sample_disc_output.shape}")
    
    # Train the model
    print("\n" + "="*50)
    print("Starting GAN Training")
    print("="*50)
    
    trained_gen, trained_disc, logs = train_gan(
        epochs=20,
        batch_size=128,
        z_dim=100,
        lr=0.0002,
        data_dir='./data',
        save_path='./models'
    )
    
    print("\nTraining completed!")
    
    # Visualize some generated digits
    print("\nGenerating sample digits...")
    visualize_generated_digits(trained_gen, z_dim=100, num_images=16)
