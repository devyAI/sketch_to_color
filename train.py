import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import ColorizerNet
from load_data import get_data_loaders, convert_to_grayscale
import matplotlib.pyplot as plt
import numpy as np

def train_model(num_epochs=50, batch_size=32, learning_rate=0.001):
    # Initialize model, loss, and optimizer
    model = ColorizerNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, _ = data
            inputs = inputs.to(device)
            
            # Convert to grayscale
            grayscale_inputs = convert_to_grayscale(inputs)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(grayscale_inputs)
            
            # Compute loss
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Log loss to TensorBoard
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)
            
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Save model checkpoint
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch + 1}.pth')
    
    print('Finished Training')
    writer.close()
    
    return model

def visualize_results(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, _ = data
            grayscale_images = convert_to_grayscale(images)
            
            # Get predictions
            colored_images = model(grayscale_images)
            
            # Show original, grayscale, and predicted images
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            
            for i in range(5):
                # Original image
                original_img = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
                axes[0, i].imshow(original_img)
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Grayscale image
                grayscale_img = np.squeeze(grayscale_images[i].cpu().numpy())
                axes[1, i].imshow(grayscale_img, cmap='gray')
                axes[1, i].set_title('Grayscale')
                axes[1, i].axis('off')
                
                # Predicted colored image
                predicted_img = np.transpose(colored_images[i].cpu().numpy(), (1, 2, 0))
                axes[2, i].imshow(predicted_img)
                axes[2, i].set_title('Predicted')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.show()
            break  # Show only one batch

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoints directory if it doesn't exist
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train the model
    model = train_model()
    
    # Visualize results
    _, test_loader = get_data_loaders(batch_size=5)
    visualize_results(model, test_loader)
