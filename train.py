import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime

# Import the CNP class from our local implementation
from cnp_model import CNP

# Default hyperparameters
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_HIDDEN_LAYERS = 3
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 200
DEFAULT_MIN_CONTEXT = 3
DEFAULT_MAX_CONTEXT = 20
DEFAULT_MIN_STD = 0.05

def get_device(device_str='auto'):
    """
    Get the appropriate device based on the device string and availability.
    
    Args:
        device_str: String specifying the device. Options:
            - 'auto': Use MPS if available, then CUDA if available, else CPU
            - 'cuda': Use CUDA
            - 'mps': Use MPS (Apple M-series GPU)
            - 'cpu': Use CPU
    
    Returns:
        The torch device
    """
    if device_str == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    elif device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_str == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_data(data_path):
    """Load demonstrations from file."""
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data['demonstrations'])} demonstrations from {data_path}")
        return data['demonstrations'], data['heights']
    except (FileNotFoundError, EOFError) as e:
        print(f"Error loading data from {data_path}: {e}")
        print("Please run collect_data.py first to generate the demonstrations.")
        exit(1)

def create_dataset(demonstrations, heights):
    """
    Create a dataset from demonstrations.
    
    Each demonstration has the form (t, ey, ez, oy, oz, h) where:
    - t is time (query dimension)
    - h is the object height (condition dimension)
    - ey, ez, oy, oz are target dimensions
    
    We need to create a dataset where each entry has:
    - x_context: (n_context, d_x) = (n_context, 2) [t, h]
    - y_context: (n_context, d_y) = (n_context, 4) [ey, ez, oy, oz]
    - x_target: (n_target, d_x) = (n_target, 2) [t, h]
    - y_target: (n_target, d_y) = (n_target, 4) [ey, ez, oy, oz]
    """
    dataset = []
    
    for demo, height in zip(demonstrations, heights):
        # Extract components
        times = demo[:, 0]  # t
        ee_pos = demo[:, 1:3]  # ey, ez
        obj_pos = demo[:, 3:5]  # oy, oz
        
        # Create x (query dimensions): [t, h]
        x = np.column_stack([times, np.ones_like(times) * height])
        
        # Create y (target dimensions): [ey, ez, oy, oz]
        y = np.column_stack([ee_pos, obj_pos])
        
        dataset.append((x, y))
    
    return dataset

def train_model(model, train_data, val_data=None, 
                batch_size=DEFAULT_BATCH_SIZE, 
                num_epochs=DEFAULT_NUM_EPOCHS,
                learning_rate=DEFAULT_LEARNING_RATE,
                min_context=DEFAULT_MIN_CONTEXT, 
                max_context=DEFAULT_MAX_CONTEXT,
                device='cpu'):
    """Train the CNP model."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    
    # Move model to device
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Create mini-batches
        np.random.shuffle(train_data)
        
        # Process in batches
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch_data = train_data[i:i+batch_size]
            batch_loss = 0.0
            
            # Process each demonstration in the batch
            observations = []
            targets = []
            target_truths = []
            
            for x, y in batch_data:
                # Randomly select number of context points
                n_context = np.random.randint(min_context, min(max_context, len(x)))
                n_target = len(x)  # Use all points as targets
                
                # Randomly sample context points
                context_indices = np.random.choice(len(x), n_context, replace=False)
                
                # Create context and target tensors
                x_context = x[context_indices]
                y_context = y[context_indices]
                x_target = x  # All points are targets
                y_target = y
                
                # Combine x and y for observations
                observation = np.concatenate([x_context, y_context], axis=1)
                target = x_target
                target_truth = y_target
                
                # Convert to tensors
                observation = torch.tensor(observation, dtype=torch.float32).to(device)
                target = torch.tensor(target, dtype=torch.float32).to(device)
                target_truth = torch.tensor(target_truth, dtype=torch.float32).to(device)
                
                observations.append(observation)
                targets.append(target)
                target_truths.append(target_truth)
            
            # Stack along batch dimension
            max_obs_len = max(obs.shape[0] for obs in observations)
            max_target_len = max(tgt.shape[0] for tgt in targets)
            
            # Create tensors with padding and masks
            batch_observations = torch.zeros(len(batch_data), max_obs_len, observations[0].shape[1], device=device)
            batch_targets = torch.zeros(len(batch_data), max_target_len, targets[0].shape[1], device=device)
            batch_target_truths = torch.zeros(len(batch_data), max_target_len, target_truths[0].shape[1], device=device)
            obs_mask = torch.zeros(len(batch_data), max_obs_len, device=device)
            target_mask = torch.zeros(len(batch_data), max_target_len, device=device)
            
            for b in range(len(batch_data)):
                obs_len = observations[b].shape[0]
                target_len = targets[b].shape[0]
                
                batch_observations[b, :obs_len] = observations[b]
                batch_targets[b, :target_len] = targets[b]
                batch_target_truths[b, :target_len] = target_truths[b]
                obs_mask[b, :obs_len] = 1.0
                target_mask[b, :target_len] = 1.0
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass and loss calculation
            loss = model.nll_loss(batch_observations, batch_targets, batch_target_truths, 
                              observation_mask=obs_mask, target_mask=target_mask)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1
        
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        
        # Validation
        if val_data is not None:
            val_loss = evaluate(model, val_data, batch_size, min_context, max_context, device)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.4f}")
    
    return train_losses, val_losses

def evaluate(model, val_data, batch_size, min_context, max_context, device='cpu'):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch_data = val_data[i:i+batch_size]
            observations = []
            targets = []
            target_truths = []
            
            for x, y in batch_data:
                # Randomly select number of context points
                n_context = np.random.randint(min_context, min(max_context, len(x)))
                
                # Randomly sample context points
                context_indices = np.random.choice(len(x), n_context, replace=False)
                
                # Create context and target tensors
                x_context = x[context_indices]
                y_context = y[context_indices]
                x_target = x  # All points are targets
                y_target = y
                
                # Combine x and y for observations
                observation = np.concatenate([x_context, y_context], axis=1)
                target = x_target
                target_truth = y_target
                
                # Convert to tensors
                observation = torch.tensor(observation, dtype=torch.float32).to(device)
                target = torch.tensor(target, dtype=torch.float32).to(device)
                target_truth = torch.tensor(target_truth, dtype=torch.float32).to(device)
                
                observations.append(observation)
                targets.append(target)
                target_truths.append(target_truth)
            
            # Stack along batch dimension
            max_obs_len = max(obs.shape[0] for obs in observations)
            max_target_len = max(tgt.shape[0] for tgt in targets)
            
            # Create tensors with padding and masks
            batch_observations = torch.zeros(len(batch_data), max_obs_len, observations[0].shape[1], device=device)
            batch_targets = torch.zeros(len(batch_data), max_target_len, targets[0].shape[1], device=device)
            batch_target_truths = torch.zeros(len(batch_data), max_target_len, target_truths[0].shape[1], device=device)
            obs_mask = torch.zeros(len(batch_data), max_obs_len, device=device)
            target_mask = torch.zeros(len(batch_data), max_target_len, device=device)
            
            for b in range(len(batch_data)):
                obs_len = observations[b].shape[0]
                target_len = targets[b].shape[0]
                
                batch_observations[b, :obs_len] = observations[b]
                batch_targets[b, :target_len] = targets[b]
                batch_target_truths[b, :target_len] = target_truths[b]
                obs_mask[b, :obs_len] = 1.0
                target_mask[b, :target_len] = 1.0
            
            # Forward pass and loss calculation
            loss = model.nll_loss(batch_observations, batch_targets, batch_target_truths, 
                              observation_mask=obs_mask, target_mask=target_mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def plot_loss_curves(train_losses, val_losses=None, save_path=None):
    """Plot the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def save_model(model, args, train_loss, val_loss):
    """Save model checkpoint."""
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create path to model file
    model_path = os.path.join(args.model_dir, 'cnp_model.pt')
    
    # Save model checkpoint
    checkpoint = {
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'min_std': args.min_std
    }
    
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")
    return model_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a CNP model for robot trajectory demonstration')
    parser.add_argument('--data_path', type=str, default='data/demonstrations.pkl', help='Path to demonstration data')
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_HIDDEN_SIZE, help='Hidden size of the model')
    parser.add_argument('--num_hidden_layers', type=int, default=DEFAULT_NUM_HIDDEN_LAYERS, help='Number of hidden layers')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--min_context', type=int, default=DEFAULT_MIN_CONTEXT, help='Minimum number of context points')
    parser.add_argument('--max_context', type=int, default=DEFAULT_MAX_CONTEXT, help='Maximum number of context points')
    parser.add_argument('--min_std', type=float, default=DEFAULT_MIN_STD, help='Minimum standard deviation')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save the trained model')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device(args.device)
    
    # Print training configuration
    print(f"Training configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Number of hidden layers: {args.num_hidden_layers}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Min context points: {args.min_context}")
    print(f"  Max context points: {args.max_context}")
    print(f"  Min std: {args.min_std}")
    print(f"  Validation split: {args.val_split}")
    print(f"  Device: {device}")
    
    # Load data
    print("Loading data...")
    demonstrations, heights = load_data(args.data_path)
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(demonstrations, heights)
    
    # Split into train and validation sets
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_data = dataset[:train_size]
    val_data = dataset[train_size:] if val_size > 0 else None
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data) if val_data else 0}")
    
    # Create model
    print("Creating model...")
    # Input shape: (d_x, d_y) = (2, 4) for ([t, h], [ey, ez, oy, oz])
    model = CNP(in_shape=(2, 4), 
                hidden_size=args.hidden_size, 
                num_hidden_layers=args.num_hidden_layers,
                min_std=args.min_std)
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        min_context=args.min_context,
        max_context=args.max_context,
        device=device
    )
    
    # Create timestamp for loss curve file only
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model - always use the fixed name cnp_model.pt
    model_path = save_model(model, args, train_losses[-1], val_losses[-1] if val_losses else None)
    
    # Plot loss curves
    data_dir = os.path.dirname(args.data_path)
    plot_loss_curves(
        train_losses, 
        val_losses, 
        save_path=os.path.join(data_dir, f"loss_curves_{timestamp}.png")
    )

if __name__ == "__main__":
    main() 