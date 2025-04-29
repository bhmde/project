import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from tqdm import tqdm

class GameStateUtilityDataset(Dataset):
    def __init__(self, csv_file, use_current_player=True, delimiter='|', precompute=True):
        """
        Custom dataset for game state data with binary vector representation.
        
        Args:
            csv_file (str): Path to the CSV file
            use_current_player (bool): If True, label is utility of current player,
                                      otherwise utility of other player
            delimiter (str): CSV delimiter character
            precompute (bool): Whether to precompute all binary vectors at init
        """
        # Read CSV file with specified delimiter
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                # Convert all values to integers
                self.data.append([int(val) for val in row])
        
        self.use_current_player = use_current_player
        
        # Precompute binary vectors if enabled
        if precompute:
            print("Precomputing binary vectors for all states...")
            self.vectors = np.zeros((len(self.data), 64), dtype=np.float32)
            
            # Efficiently compute binary representation for all states
            for i, row in enumerate(self.data):
                state = row[0]  # First column is the state
                for bit_pos in range(64):
                    if state & (1 << bit_pos):
                        self.vectors[i, 63-bit_pos] = 1.0
                        
            self.vectors = torch.tensor(self.vectors, dtype=torch.float32)
        else:
            self.vectors = None
    
    def __len__(self):
        return len(self.data)
    
    def _state_to_vector(self, state):
        """
        Convert integer state to a 64-dimensional binary vector.
        
        Args:
            state (int): Integer state value
            
        Returns:
            torch.Tensor: 64-dimensional binary vector
        """
        # Create a tensor of zeros
        vec = torch.zeros(64, dtype=torch.float32)
        
        # Set bits in the vector
        for i in range(64):
            if state & (1 << i):
                vec[63-i] = 1.0
                
        return vec
    
    def __getitem__(self, idx):
        row = self.data[idx]
        state = row[0]          # State
        current_player = row[2]  # Current player
        utility_0 = row[3]      # Utility for player 0
        utility_1 = row[4]      # Utility for player 1
        
        # Get binary vector representation
        if self.vectors is not None:
            # Use precomputed vector
            x = self.vectors[idx]
        else:
            # Compute on-the-fly
            x = self._state_to_vector(state)
        
        # Get utility based on use_current_player flag
        if self.use_current_player:
            utility = utility_0 if current_player == 0 else utility_1
        else:
            utility = utility_1 if current_player == 0 else utility_0
        
        # Convert utility to class index (assuming values in {-1, 0, 1})
        # Map -1 -> 0, 0 -> 1, 1 -> 2 for use with CrossEntropyLoss
        utility_class = utility + 1
        
        y = torch.tensor(utility_class, dtype=torch.long)
        
        return x, y

class GameStateUtilityMLP(nn.Module):
    def __init__(self, input_size=64, hidden_sizes=[128, 64], num_classes=3, dropout=0.2):
        """
        MLP model for game state utility prediction.
        
        Args:
            input_size (int): Input feature dimension (64 for binary state vector)
            hidden_sizes (list): List of hidden layer sizes
            num_classes (int): Number of output classes (3 for {-1, 0, 1})
            dropout (float): Dropout probability
        """
        super(GameStateUtilityMLP, self).__init__()
        
        # Build the layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer for multi-class classification
        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)
    
    def forward(self, x):
        features = self.feature_layers(x)
        return self.classifier(features)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Run one training epoch"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_samples += targets.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    return epoch_loss, epoch_accuracy

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # For confusion matrix
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            # Store for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    # Calculate confusion matrix (optional)
    # confusion = confusion_matrix(all_targets, all_predictions)
    
    return val_loss, val_accuracy

def plot_metrics_in_dir(train_losses, val_losses, train_accuracies, val_accuracies, output_dir):
    """Plot training and validation metrics (loss and accuracy) in the given directory"""
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the loss plot
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)  # Accuracy is between 0 and 1
    
    # Save the accuracy plot
    acc_plot_path = os.path.join(output_dir, 'accuracy_plot.png')
    plt.savefig(acc_plot_path)
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and not args.no_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = GameStateUtilityDataset(
        csv_file=args.data,
        use_current_player=args.use_current_player,
        delimiter=args.delimiter,
        precompute=True
    )
    
    # Split dataset into train and validation sets
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create model for multi-class classification
    model = GameStateUtilityMLP(
        input_size=64,  # Binary state vector size
        hidden_sizes=args.hidden_sizes,
        num_classes=3,  # -1, 0, 1 mapped to 0, 1, 2
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Loss function and optimizer for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Print model summary
    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S_training')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Calculate checkpoint epochs based on n_checkpoints
    checkpoint_epochs = []
    if args.n_checkpoints > 0 and args.epochs > 1:
        # Distribute checkpoints evenly throughout training
        if args.n_checkpoints >= args.epochs:
            # If more checkpoints than epochs, just save after every epoch
            checkpoint_epochs = list(range(1, args.epochs + 1))
        else:
            # Distribute checkpoints evenly
            step = args.epochs / (args.n_checkpoints + 1)
            checkpoint_epochs = [int(round(step * i)) for i in range(1, args.n_checkpoints + 1)]
            # Make sure we don't include the final epoch (that will be saved separately)
            if args.epochs in checkpoint_epochs:
                checkpoint_epochs.remove(args.epochs)
        
        print(f"Will save checkpoints at epochs: {checkpoint_epochs}")
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0  # Track best validation accuracy
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_num = epoch + 1  # For easier reading (epochs start at 1)
        
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print statistics
        print(f"Epoch {epoch_num}/{args.epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        # Save checkpoint if this epoch is in the checkpoint list
        if epoch_num in checkpoint_epochs:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch_num}.pth')
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch_num}: {checkpoint_path}")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'best_val_acc': best_val_acc,
                'args': vars(args)
            }, best_model_path)
            print(f"Best model saved at epoch {epoch_num} with validation accuracy: {val_acc:.4f}")
    
    # Create plots in the same output directory (don't create another timestamp folder)
    plot_metrics_in_dir(train_losses, val_losses, train_accuracies, val_accuracies, output_dir)
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'args': vars(args)
    }, final_model_path)
    
    print(f"Final model saved to {final_model_path}")
    
    ## # Add a confusion matrix as a bonus (optional)
    ## try:
    ##     from sklearn.metrics import confusion_matrix
    ##     import seaborn as sns
    ##     
    ##     # Predict on validation set
    ##     model.eval()
    ##     all_preds = []
    ##     all_targets = []
    ##     
    ##     with torch.no_grad():
    ##         for inputs, targets in val_loader:
    ##             inputs, targets = inputs.to(device), targets.to(device)
    ##             outputs = model(inputs)
    ##             _, preds = torch.max(outputs, 1)
    ##             
    ##             all_preds.extend(preds.cpu().numpy())
    ##             all_targets.extend(targets.cpu().numpy())
    ##     
    ##     # Create confusion matrix
    ##     cm = confusion_matrix(all_targets, all_preds)
    ##     
    ##     # Plot confusion matrix
    ##     plt.figure(figsize=(8, 6))
    ##     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    ##                 xticklabels=['-1', '0', '1'], 
    ##                 yticklabels=['-1', '0', '1'])
    ##     plt.xlabel('Predicted')
    ##     plt.ylabel('True')
    ##     plt.title('Confusion Matrix')
    ##     
    ##     # Save confusion matrix
    ##     cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    ##     plt.savefig(cm_path)
    ##     plt.close()  # Close plot to avoid display in non-interactive environments
    ##     
    ##     print(f"Confusion matrix saved to {cm_path}")
    ## except:
    ##     print("Skipping confusion matrix generation (requires sklearn, seaborn)")


    # Save command line arguments to a text file
    args_file_path = os.path.join(output_dir, 'training_args.txt')
    with open(args_file_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    print(f"Command line arguments saved to {args_file_path}")
    print("Training completed!")
    

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MLP for game state utility prediction')
    
    # Dataset parameters
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--delimiter', type=str, default='|', help='CSV delimiter character')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--use_current_player', action='store_true', 
                        help='Use utility of current player as target')
    
    # Model parameters
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 64], 
                        help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--n_checkpoints', type=int, default=3, 
                        help='Number of checkpoints to save throughout training (0 to disable)')
    
    # Other parameters
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_mps', action='store_true', help='Disable MPS (Apple Silicon)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Directory to save model and plots')
                        
    args = parser.parse_args()

    main(args)
