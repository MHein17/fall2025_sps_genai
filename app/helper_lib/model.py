import torch.nn as nn
import torch.nn.functional as F
import torch

# Assignment 2 - Part 1: Specific CNN Architecture
class Assignment2CNN(nn.Module):
    def __init__(self):
        super(Assignment2CNN, self).__init__()
        # Input: 64x64x3 RGB image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # After 2 pooling layers: 64->32->16
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(model_name, **kwargs):
    """
    Get model by name: 'FCNN', 'CNN', 'EnhancedCNN', 'Assignment2CNN'
    """
    if model_name == 'FCNN':
        return FCNN()
    elif model_name == 'CNN':
        return SimpleCNN()
    elif model_name == 'EnhancedCNN':
        return EnhancedCNN()
    elif model_name == 'Assignment2CNN':
        return Assignment2CNN()
    else:
        raise ValueError(f"Unknown model: {model_name}")