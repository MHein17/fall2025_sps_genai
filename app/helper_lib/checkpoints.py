import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir='checkpoints'):
    """
    Save model checkpoint

    TODO: Implement checkpoint saving that includes:
    1. Model state dict
    2. Optimizer state dict  
    3. Epoch number
    4. Loss and accuracy metrics
    5. Create checkpoint directory if needed

    Hint: Reference the FCNN notebook checkpoint implementation
    """
    # TODO: Implement checkpoint saving
    pass

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint and restore training state

    TODO: Implement checkpoint loading that:
    1. Loads the checkpoint file
    2. Restores model and optimizer states
    3. Returns epoch, loss, and accuracy information

    Why save optimizer state? See FCNN notebook documentation!
    """
    # TODO: Implement checkpoint loading  
    pass