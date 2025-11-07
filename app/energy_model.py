import torch
import torch.nn as nn
import numpy as np
import random

# Swish activation function
def swish(x):
    return x * torch.sigmoid(x)

class EnergyModel(nn.Module):
    """
    Energy-Based Model for CIFAR-10 (32x32x3 RGB images)
    """
    def __init__(self):
        super(EnergyModel, self).__init__()
        # Input: 3 channels (RGB), 32x32 images
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)  # 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 4x4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 2x2

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)

def generate_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    """
    Generate samples using Langevin dynamics
    """
    nn_energy_model.eval()

    for _ in range(steps):
        # Add noise
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)

        # Compute energy and gradients
        energy = nn_energy_model(inp_imgs)

        # Calculate gradient with respect to input images
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        # Apply gradient clipping and gradient descent step
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size*grads).clamp(-1.0, 1.0)

    return inp_imgs.detach()

class Buffer:
    """
    Buffer to store generated samples for efficient sampling
    """
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        # Start with random images in the buffer (3 channels for RGB)
        self.examples = [torch.rand((1, 3, 32, 32), device=self.device) * 2 - 1 for _ in range(128)]

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(128, 0.05)

        # Generate new random images for around 5% of the inputs
        new_rand_imgs = torch.rand((n_new, 3, 32, 32),  device=self.device) * 2 - 1

        # Sample old images from the buffer for the rest
        old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)

        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)

        # Run Langevin dynamics
        new_imgs = generate_samples(self.model, inp_imgs, steps, step_size, noise)

        # Update buffer
        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]

        return new_imgs

class Metric:
    """
    Simple metric tracker
    """
    def __init__(self):
        self.reset()

    def update(self, val):
        self.total += val.item()
        self.count += 1

    def result(self):
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0

class EBM(nn.Module):
    """
    Energy-Based Model trainer
    """
    def __init__(self, model, alpha, steps, step_size, noise, device):
        super().__init__()
        self.device = device
        # Define the nn energy model 
        self.model = model
    
        self.buffer = Buffer(self.model, device=device)

        # Define the hyperparameters
        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise

        self.loss_metric = Metric()
        self.reg_loss_metric = Metric()
        self.cdiv_loss_metric = Metric()
        self.real_out_metric = Metric()
        self.fake_out_metric = Metric()

    def metrics(self):
        return {
            "loss": self.loss_metric.result(),
            "reg": self.reg_loss_metric.result(),
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result()
        }

    def reset_metrics(self):
        for m in [self.loss_metric, self.reg_loss_metric, self.cdiv_loss_metric,
                  self.real_out_metric, self.fake_out_metric]:
            m.reset()

    def train_step(self, real_imgs, optimizer):
        real_imgs = real_imgs + torch.randn_like(real_imgs) * self.noise
        real_imgs = torch.clamp(real_imgs, -1.0, 1.0)

        fake_imgs = self.buffer.sample_new_exmps(
            steps=self.steps, step_size=self.step_size, noise=self.noise)

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        inp_imgs = inp_imgs.clone().detach().to(self.device).requires_grad_(False)

        out_scores = self.model(inp_imgs)

        real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

        cdiv_loss = real_out.mean() - fake_out.mean() 
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss 

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

        optimizer.step()

        self.loss_metric.update(loss)
        self.reg_loss_metric.update(reg_loss)
        self.cdiv_loss_metric.update(cdiv_loss)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())

        return self.metrics()

    def test_step(self, real_imgs):
        batch_size = real_imgs.shape[0]
        fake_imgs = torch.rand((batch_size, 3, 32, 32), device=self.device) * 2 - 1
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)

        with torch.no_grad():
            out_scores = self.model(inp_imgs)
            real_out, fake_out = torch.split(out_scores, batch_size, dim=0)
            cdiv = real_out.mean() - fake_out.mean()

        self.cdiv_loss_metric.update(cdiv)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())

        return {
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result()
        }
