from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_vae_model
from helper_lib.model import get_model
from helper_lib.losses import vae_loss_function
import torch
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

train_loader = get_data_loader('./data', batch_size=64, train=True)

vae = get_model("VAE").to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

train_vae_model(vae, train_loader, vae_loss_function, optimizer, device=device, epochs=5)

# Save the trained model
torch.save(vae.state_dict(), 'vae_model.pth')
print("Model saved to vae_model.pth")