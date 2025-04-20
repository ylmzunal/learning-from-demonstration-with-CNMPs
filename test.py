import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from tqdm import tqdm
import sys

# Import the CNP class from our local implementation
from cnp_model import CNP

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

def load_model(model_path, device='cpu'):
    """Load a trained CNP model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create a new model with the same parameters
    model = CNP(
        in_shape=(2, 4),
        hidden_size=checkpoint['hidden_size'],
        num_hidden_layers=checkpoint['num_hidden_layers'],
        min_std=checkpoint['min_std']
    )
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

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

def create_test_set(demonstrations, heights, num_tests=100):
    """Create a test set with randomly generated observations and queries."""
    test_set = []
    
    # Select random demonstrations for testing
    indices = np.random.choice(len(demonstrations), min(num_tests, len(demonstrations)), replace=False)
    
    for idx in indices:
        demo = demonstrations[idx]
        height = heights[idx]
        
        # Extract components
        times = demo[:, 0]  # t
        ee_pos = demo[:, 1:3]  # ey, ez
        obj_pos = demo[:, 3:5]  # oy, oz
        
        # Create x (query dimensions): [t, h]
        x = np.column_stack([times, np.ones_like(times) * height])
        
        # Create y (target dimensions): [ey, ez, oy, oz]
        y = np.column_stack([ee_pos, obj_pos])
        
        # Randomly select number of context points (between 1 and 20)
        n_context = np.random.randint(3, 20)
        
        # Randomly sample context points
        context_indices = np.random.choice(len(x), n_context, replace=False)
        
        # Create context tensors
        x_context = x[context_indices]
        y_context = y[context_indices]
        
        # Use all points as targets
        x_target = x
        y_target = y
        
        test_set.append((x_context, y_context, x_target, y_target, height))
    
    return test_set

def compute_metrics(model, test_set, device='cpu'):
    """Compute MSE for end-effector and object predictions."""
    ee_errors = []
    obj_errors = []
    
    for x_context, y_context, x_target, y_target, _ in tqdm(test_set, desc="Computing metrics"):
        # Create observation tensor by concatenating context inputs and outputs
        observation = np.concatenate([x_context, y_context], axis=1)
        observation = torch.tensor(observation, dtype=torch.float32).to(device)
        
        # Create target tensor - add batch dimension
        target = torch.tensor(x_target, dtype=torch.float32).to(device)
        target_truth = torch.tensor(y_target, dtype=torch.float32).to(device)
        
        # Add batch dimension
        observation = observation.unsqueeze(0)  # Shape becomes [1, n_context, d_x+d_y]
        target = target.unsqueeze(0)  # Shape becomes [1, n_target, d_x]
        target_truth = target_truth.unsqueeze(0)  # Shape becomes [1, n_target, d_y]
        
        # Get model predictions
        with torch.no_grad():
            mean, std = model(observation, target)
        
        # Calculate MSE for end-effector (first two dimensions)
        ee_error = ((mean[0, :, :2] - target_truth[0, :, :2]) ** 2).mean().item()
        
        # Calculate MSE for object (last two dimensions)
        obj_error = ((mean[0, :, 2:] - target_truth[0, :, 2:]) ** 2).mean().item()
        
        ee_errors.append(ee_error)
        obj_errors.append(obj_error)
    
    return np.array(ee_errors), np.array(obj_errors)

def plot_predictions(model, test_set, num_to_plot=3, device='cpu', save_path=None):
    """Plot predictions for a few test examples."""
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(len(test_set), min(num_to_plot, len(test_set)), replace=False)
    
    fig, axs = plt.subplots(num_to_plot, 2, figsize=(12, 4 * num_to_plot))
    if num_to_plot == 1:
        axs = np.array([axs])  # Make sure axs is indexable for single plot
    
    for i, idx in enumerate(indices):
        x_context, y_context, x_target, y_target, height = test_set[idx]
        
        # Create observation tensor
        observation = np.concatenate([x_context, y_context], axis=1)
        observation = torch.tensor(observation, dtype=torch.float32).to(device)
        
        # Create target tensor and add batch dimension
        target = torch.tensor(x_target, dtype=torch.float32).to(device)
        
        # Add batch dimension
        observation = observation.unsqueeze(0)  # Shape becomes [1, n_context, d_x+d_y]
        target = target.unsqueeze(0)  # Shape becomes [1, n_target, d_x]
        
        # Get model predictions
        with torch.no_grad():
            mean, std = model(observation, target)
        
        # Extract the first batch item and convert to numpy
        mean = mean[0].cpu().numpy()
        std = std[0].cpu().numpy()
        
        # Plot end-effector trajectories
        ax = axs[i, 0]
        
        # Plot ground truth
        ax.plot(y_target[:, 0], y_target[:, 1], 'b-', label='Ground Truth')
        
        # Plot context points
        ax.scatter(y_context[:, 0], y_context[:, 1], color='k', s=20, label='Context Points')
        
        # Plot predictions with confidence intervals
        ax.plot(mean[:, 0], mean[:, 1], 'r-', label='Prediction')
        
        # Plot confidence intervals
        ax.fill_between(
            mean[:, 0],
            mean[:, 1] - 2 * std[:, 1],
            mean[:, 1] + 2 * std[:, 1],
            color='r',
            alpha=0.3,
            label='95% CI'
        )
        
        ax.set_xlabel('e_y')
        ax.set_ylabel('e_z')
        ax.set_title(f'End-effector Trajectory (h={height:.3f})')
        ax.legend()
        
        # Plot object positions
        ax = axs[i, 1]
        
        # Plot ground truth
        ax.plot(y_target[:, 2], y_target[:, 3], 'b-', label='Ground Truth')
        
        # Plot context points
        ax.scatter(y_context[:, 2], y_context[:, 3], color='k', s=20, label='Context Points')
        
        # Plot predictions with confidence intervals
        ax.plot(mean[:, 2], mean[:, 3], 'r-', label='Prediction')
        
        # Plot confidence intervals
        ax.fill_between(
            mean[:, 2],
            mean[:, 3] - 2 * std[:, 3],
            mean[:, 3] + 2 * std[:, 3],
            color='r',
            alpha=0.3,
            label='95% CI'
        )
        
        ax.set_xlabel('o_y')
        ax.set_ylabel('o_z')
        ax.set_title(f'Object Trajectory (h={height:.3f})')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_error_bars(ee_errors, obj_errors, save_path=None):
    """Plot error bars for end-effector and object predictions."""
    categories = ['End-effector', 'Object']
    means = [np.mean(ee_errors), np.mean(obj_errors)]
    stds = [np.std(ee_errors), np.std(obj_errors)]
    
    plt.figure(figsize=(8, 6))
    plt.bar(categories, means, yerr=stds, capsize=10, color=['blue', 'orange'])
    plt.ylabel('Mean Squared Error')
    plt.title('Prediction Errors')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels for means and stds
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.01, f'Mean: {mean:.4f}\nStd: {std:.4f}', 
                 ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    # Print statistics
    print("\nError Statistics:")
    print(f"End-effector - Mean: {means[0]:.6f}, Std: {stds[0]:.6f}")
    print(f"Object - Mean: {means[1]:.6f}, Std: {stds[1]:.6f}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test a trained CNP model for robot trajectory demonstration')
    parser.add_argument('--model_path', type=str, default='model/cnp_model.pt', help='Path to the trained model')
    parser.add_argument('--data_path', type=str, default='data/demonstrations.pkl', help='Path to demonstration data')
    parser.add_argument('--num_tests', type=int, default=100, help='Number of test cases')
    parser.add_argument('--num_plots', type=int, default=3, help='Number of plots to generate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device(args.device)
    
    # Print test configuration
    print(f"Test configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Data path: {args.data_path}")
    print(f"  Number of tests: {args.num_tests}")
    print(f"  Number of plots: {args.num_plots}")
    print(f"  Device: {device}")
    print(f"  Output directory: {args.output_dir}")
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Model file {args.model_path} not found!")
        sys.exit(1)
        
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Create model instance
    model = CNP(in_shape=(2, 4), 
                hidden_size=checkpoint['hidden_size'],
                num_hidden_layers=checkpoint['num_hidden_layers'],
                min_std=checkpoint['min_std'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load data
    print("Loading data...")
    demonstrations, heights = load_data(args.data_path)
    
    # Create test set
    print("Creating test set...")
    test_set = create_test_set(demonstrations, heights, args.num_tests)
    
    # Compute metrics
    print("Computing metrics...")
    ee_errors, obj_errors = compute_metrics(model, test_set, device)
    
    # Plot error bars
    print("Plotting error bars...")
    plot_error_bars(ee_errors, obj_errors, save_path=os.path.join(args.output_dir, "error_bars.png"))
    
    # Plot predictions
    print("Plotting predictions...")
    plot_predictions(model, test_set, args.num_plots, device, save_path=os.path.join(args.output_dir, "predictions.png"))

if __name__ == '__main__':
    main() 