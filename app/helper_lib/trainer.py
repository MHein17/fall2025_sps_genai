import torch
from tqdm import tqdm
from helper_lib.checkpoints import save_checkpoint

def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=10, checkpoint_dir='checkpoints'):
    """
    Enhanced training loop with checkpoint functionality

    Works for FCNN, SimpleCNN, EnhancedCNN, and any other classification model
    """
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        train_loader_with_progress = tqdm(
            iterable=train_loader, 
            ncols=120, 
            desc=f'Epoch {epoch+1}/{epochs}'
        )
        
        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            
            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix({
                    'avg accuracy': f'{running_correct/running_total:.3f}',
                    'avg loss': f'{running_loss/(batch_number+1):.4f}'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * running_correct / running_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Save checkpoint every epoch
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch + 1, epoch_loss, epoch_accuracy, 
            checkpoint_dir=checkpoint_dir
        )
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_path = save_checkpoint(
                model, optimizer, epoch + 1, val_loss, val_accuracy,
                checkpoint_dir=f'{checkpoint_dir}/best'
            )
            print(f"New best model saved! Val Accuracy: {val_accuracy:.2f}%")
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_accuracy:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2f}%")
        print(f"Checkpoint saved: {checkpoint_path}")
    
    print("Finished Training")
    return model



def train_vae_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    Training loop for VAE models (unsupervised)
    """
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        data_loader_with_progress = tqdm(
            iterable=data_loader,
            ncols=120,
            desc=f'Epoch {epoch+1}/{epochs}'
        )
        
        for batch_number, (inputs, _) in enumerate(data_loader_with_progress):
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss = criterion(recon, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_number % 100 == 99:
                data_loader_with_progress.set_postfix({
                    'avg loss': f'{running_loss/(batch_number+1):.4f}'
                })
        
        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}")
    
    print("Finished Training")
    return model