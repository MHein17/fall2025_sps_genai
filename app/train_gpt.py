import torch
import torch.optim as optim
from tqdm import tqdm
import os

def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir='checkpoints'):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, checkpoint_path)
    
    return checkpoint_path


def train_gpt(model, dataloader, optimizer, epochs, device):
    """
    Fine-tune GPT-2 model on SQuAD dataset
    """
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        data_loader_with_progress = tqdm(
            iterable=dataloader,
            ncols=80,
            desc=f"Epoch {epoch+1}/{epochs}"
        )
        for batch_number, batch in enumerate(data_loader_with_progress):
            # Extract data from batch dict
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()

            # Forward pass with attention mask
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()
            if (batch_number % 100 == 0) or (batch_number == len(dataloader) - 1):
                data_loader_with_progress.set_postfix(
                    {
                        "avg loss": f"{total_loss/(batch_number+1):.4f}",
                    }
                )
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Save checkpoint after each epoch
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=avg_loss,
            accuracy=0.0,  # QA doesn't have simple accuracy metric during training
            checkpoint_dir="app/checkpoints/gpt2-squad"
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    return avg_loss


if __name__ == "__main__":
    # Import dependencies
    from app.helper_lib.data_loader import get_data_loader
    from app.helper_lib.model import get_model
    
    # Training configuration
    batch_size = 128
    max_length = 80
    epochs = 5
    learning_rate = 1e-5
    
    # Check device availability
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")
    
    # Load data
    train_loader = get_data_loader("data/train", batch_size=batch_size, dataset_name="SQUAD", max_length=max_length)
    print(f"Number of training samples: {len(train_loader.dataset)}")
    
    # Load model
    model = get_model("GPT2")
    
    # Lower learning rate for finetuning (typical range: 1e-5 to 5e-5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Train the model
    train_gpt(model, train_loader, optimizer, epochs=epochs, device=device)
    
    print("Fine-tuning complete!")
    print("To test the finetuned model, run: python app/test_gpt_model.py")
