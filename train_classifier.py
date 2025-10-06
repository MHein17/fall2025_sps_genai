import sys
sys.path.insert(0, 'app')

from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.model import get_model
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load CIFAR-10 data resized to 64x64
print("Loading CIFAR-10 dataset (64x64)...")
train_loader = get_data_loader('./data', batch_size=64, train=True, dataset_name='CIFAR10_64')
test_loader = get_data_loader('./data', batch_size=64, train=False, dataset_name='CIFAR10_64')

# Create the Assignment 2 CNN model
model = get_model("Assignment2CNN").to(device)
print(model)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting classifier training...")
trained_model = train_model(
    model, 
    train_loader, 
    test_loader,  # Using test_loader as validation
    criterion, 
    optimizer, 
    device=device, 
    epochs=10,
    checkpoint_dir='checkpoints_classifier'
)

# Save the final model
torch.save(trained_model.state_dict(), 'app/classifier_model.pth')
print("Classifier model saved to app/classifier_model.pth")