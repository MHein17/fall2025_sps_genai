import torch
import torch.nn.functional as F

def vae_loss_function(recon_x, x, mu, logvar):
    """
    VAE loss from Module_5_Practical_VAE.py
    """
    beta = 500
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * BCE + KLD