"""
Training Script for MNIST GAN - Assignment 3
Run this script to train the GAN model on MNIST dataset
"""

import torch
from mnist_gan import train_gan, Generator, visualize_generated_digits, device
import os


def main():
    print("="*70)
    print(" MNIST GAN Training - Assignment 3")
    print("="*70)
    print(f"\nUsing device: {device}")
    
    # Training parameters
    EPOCHS = 20
    BATCH_SIZE = 128
    Z_DIM = 100
    LEARNING_RATE = 0.0002
    DATA_DIR = './data'
    SAVE_DIR = './models'
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Noise Dimension: {Z_DIM}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Model Save Directory: {SAVE_DIR}")
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Train the GAN
    print("\n" + "="*70)
    print(" Starting Training...")
    print("="*70 + "\n")
    
    trained_generator, trained_discriminator, training_logs = train_gan(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        z_dim=Z_DIM,
        lr=LEARNING_RATE,
        data_dir=DATA_DIR,
        save_path=SAVE_DIR
    )
    
    print("\n" + "="*70)
    print(" Training Completed Successfully!")
    print("="*70)
    
    print(f"\nModels saved to: {SAVE_DIR}/")
    print(f"  - generator.pth")
    print(f"  - discriminator.pth")
    print(f"\nGenerated images saved to: {SAVE_DIR}/epoch_*.png")
    
    # Test the trained generator
    print("\n" + "="*70)
    print(" Testing Trained Generator...")
    print("="*70 + "\n")
    
    print("Generating 16 sample digits...")
    visualize_generated_digits(trained_generator, z_dim=Z_DIM, num_images=16)
    
    # Print training statistics
    if training_logs:
        print("\n" + "="*70)
        print(" Training Statistics")
        print("="*70)
        final_log = training_logs[-1]
        print(f"\nFinal Epoch: {final_log['epoch']:.2f}")
        print(f"Final Discriminator Loss: {final_log['D loss']:.4f}")
        print(f"Final Generator Loss: {final_log['G loss']:.4f}")
    
    print("\n" + "="*70)
    print(" All Done! You can now deploy the model to the API.")
    print("="*70)


if __name__ == "__main__":
    main()
