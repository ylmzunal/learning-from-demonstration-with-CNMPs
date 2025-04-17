import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from cnp import CNP

def generate_synthetic_data(n_trajectories=100, n_points=100, noise_level=0.02):
    """
    Generate synthetic data similar to the robot demonstration data
    
    Parameters:
    -----------
    n_trajectories : int
        Number of trajectories to generate
    n_points : int
        Number of points per trajectory
    noise_level : float
        Standard deviation of the noise to add to the trajectories
        
    Returns:
    --------
    data : list of numpy arrays
        Each array has shape (n_points, 6) where the columns are (t, e_y, e_z, o_y, o_z, h)
    """
    data = []
    
    for i in range(n_trajectories):
        # Generate a random height for the object
        h = np.random.uniform(0.03, 0.1)
        
        # Generate time points
        t = np.linspace(0, 1, n_points).reshape(-1, 1)
        
        # Generate end-effector trajectory (semicircle-like)
        e_y = 0.3 * np.cos(np.pi * t) + 0.5 * np.random.uniform(-0.1, 0.1)
        e_z = 0.3 * np.sin(np.pi * t) + 1.04 + 0.2 * np.random.uniform(0, 1)
        
        # Generate object trajectory (influenced by end-effector with some delay)
        # The object movement depends on the height
        o_y = 0.8 * e_y + 0.2 * np.roll(e_y, int(n_points/10)) + noise_level * np.random.randn(n_points, 1)
        o_z = np.ones_like(e_z) * (1.04 + h/2) + h * 0.5 * np.sin(2 * np.pi * t) * np.exp(-5 * (t - 0.5)**2) + noise_level * np.random.randn(n_points, 1)
        
        # Add noise to make it more realistic
        e_y += noise_level * np.random.randn(n_points, 1)
        e_z += noise_level * np.random.randn(n_points, 1)
        
        # Stack everything together
        h_array = np.ones_like(t) * h
        trajectory = np.concatenate([t, e_y, e_z, o_y, o_z, h_array], axis=1)
        
        data.append(trajectory)
    
    return data


class RobotCNMP:
    def __init__(self, hidden_size=64, num_hidden_layers=3, min_std=0.01):
        # CNP Model for predicting the robot and object positions
        # Input shape: t (query dimension) -> (e_y, e_z, o_y, o_z) (target dimensions)
        # Additionally, we use h (object height) as a condition
        self.model = CNP(in_shape=(2, 4), hidden_size=hidden_size, 
                         num_hidden_layers=num_hidden_layers, min_std=min_std)
        # Check for MPS (Apple Silicon) first, then CUDA, then fall back to CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
    def train(self, dataset, num_epochs=100, batch_size=16):
        """
        Train the CNMP model
        
        Parameters:
        -----------
        dataset : list of numpy arrays
            Each array has shape (n_steps, 6) where the columns are (t, e_y, e_z, o_y, o_z, h)
        num_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        """
        print(f"Training on {len(dataset)} trajectories for {num_epochs} epochs")
        
        losses = []
        
        for epoch in range(num_epochs):
            # Train for one epoch
            epoch_loss = self._train_epoch(dataset, batch_size)
            losses.append(epoch_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('training_loss.png')
        plt.close()
        
        return losses
    
    def _train_epoch(self, dataset, batch_size=16):
        """
        Train the model for one epoch
        
        Parameters:
        -----------
        dataset : list of numpy arrays
            Dataset to train on
        batch_size : int
            Batch size
            
        Returns:
        --------
        avg_loss : float
            Average loss for the epoch
        """
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle dataset for each epoch
        np.random.shuffle(dataset)
        
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i:i+batch_size]
            
            # Process batch for CNP training
            batch_loss = self._train_batch(batch_data)
            
            epoch_loss += batch_loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def _train_batch(self, batch_data):
        """Process and train on a batch of trajectories"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Number of context points to sample randomly
        n_context = np.random.randint(1, 10)
        
        batch_size = len(batch_data)
        total_loss = 0.0
        
        for traj in batch_data:
            # Each trajectory has shape (n_steps, 6) where columns are (t, e_y, e_z, o_y, o_z, h)
            traj_tensor = torch.tensor(traj, dtype=torch.float32, device=self.device)
            
            # Split into query dimensions and target dimensions
            t = traj_tensor[:, 0:1]  # Time (query dimension)
            target_dims = traj_tensor[:, 1:5]  # (e_y, e_z, o_y, o_z) (target dimensions)
            h = traj_tensor[:, 5:6]  # Height (condition)
            
            # Random indices for context points
            seq_len = traj_tensor.shape[0]
            context_idxs = np.random.choice(seq_len, size=min(n_context, seq_len), replace=False)
            
            # Create context observations (t, h, e_y, e_z, o_y, o_z)
            context_t = torch.cat([t[context_idxs], h[context_idxs]], dim=1)
            context_obs = torch.cat([context_t, target_dims[context_idxs]], dim=1)
            
            # Create target queries (t, h)
            target_query = torch.cat([t, h], dim=1)
            
            # Add batch dimension
            context_obs = context_obs.unsqueeze(0)  # [1, n_context, d_x+d_y]
            target_query = target_query.unsqueeze(0)  # [1, n_targets, d_x]
            target_dims = target_dims.unsqueeze(0)  # [1, n_targets, d_y]
            
            # Compute loss
            loss = self.model.nll_loss(context_obs, target_query, target_dims)
            total_loss += loss
        
        # Average loss across the batch
        avg_loss = total_loss / batch_size
        
        # Backpropagate and optimize
        avg_loss.backward()
        self.optimizer.step()
        
        return avg_loss.item()
    
    def test(self, test_dataset, max_context_points=10):
        """
        Test the model on the provided test dataset
        
        Parameters:
        -----------
        test_dataset : list of numpy arrays
            Each array has shape (n_steps, 6) where the columns are (t, e_y, e_z, o_y, o_z, h)
        max_context_points : int
            Maximum number of context points to use
            
        Returns:
        --------
        errors : dict
            Dictionary with MSE values for end-effector and object predictions
        """
        print(f"Testing model on {len(test_dataset)} test cases")
        
        self.model.eval()
        
        # Metrics to track
        ee_errors = []
        obj_errors = []
        
        with torch.no_grad():
            for i, traj in enumerate(test_dataset):
                traj_tensor = torch.tensor(traj, dtype=torch.float32, device=self.device)
                
                # Get all dimensions
                t = traj_tensor[:, 0:1]  # Time
                target_dims = traj_tensor[:, 1:5]  # (e_y, e_z, o_y, o_z)
                h = traj_tensor[:, 5:6]  # Height
                
                # Sample random number of context points
                n_context = np.random.randint(1, min(max_context_points, traj_tensor.shape[0]))
                context_idxs = np.random.choice(traj_tensor.shape[0], size=n_context, replace=False)
                
                # Create context observations
                context_t = torch.cat([t[context_idxs], h[context_idxs]], dim=1)
                context_obs = torch.cat([context_t, target_dims[context_idxs]], dim=1)
                
                # Create target queries - predict for all points
                target_query = torch.cat([t, h], dim=1)
                
                # Add batch dimension
                context_obs = context_obs.unsqueeze(0)  # [1, n_context, d_x+d_y]
                target_query = target_query.unsqueeze(0)  # [1, n_targets, d_x]
                
                # Get predictions
                mean, std = self.model.forward(context_obs, target_query)
                
                # Remove batch dimension
                mean = mean.squeeze(0)
                
                # Compute MSE for end-effector (first 2 dims) and object (last 2 dims)
                ee_mse = ((mean[:, :2] - target_dims[:, :2]) ** 2).mean().item()
                obj_mse = ((mean[:, 2:] - target_dims[:, 2:]) ** 2).mean().item()
                
                ee_errors.append(ee_mse)
                obj_errors.append(obj_mse)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(test_dataset)} test cases", end="\r")
        
        # Calculate statistics
        ee_mse_mean = np.mean(ee_errors)
        ee_mse_std = np.std(ee_errors)
        obj_mse_mean = np.mean(obj_errors)
        obj_mse_std = np.std(obj_errors)
        
        print(f"\nEnd-effector MSE: {ee_mse_mean:.6f} ± {ee_mse_std:.6f}")
        print(f"Object MSE: {obj_mse_mean:.6f} ± {obj_mse_std:.6f}")
        
        # Plot the errors
        plt.figure(figsize=(10, 6))
        
        # Create bar plot with error bars
        x = ['End-effector', 'Object']
        means = [ee_mse_mean, obj_mse_mean]
        stds = [ee_mse_std, obj_mse_std]
        
        plt.bar(x, means, yerr=stds, capsize=10, color=['blue', 'red'], alpha=0.7)
        plt.ylabel('Mean Squared Error')
        plt.title('Prediction Error on Test Set')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig('test_errors.png')
        plt.close()
        
        return {
            'ee_mse_mean': ee_mse_mean,
            'ee_mse_std': ee_mse_std,
            'obj_mse_mean': obj_mse_mean,
            'obj_mse_std': obj_mse_std
        }
    
    def save_model(self, path="cnmp_model.pt"):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="cnmp_model.pt"):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


def save_data(data, path="synthetic_data.npy"):
    """Save generated data"""
    np.save(path, data)
    print(f"Saved {len(data)} trajectories to {path}")


def load_data(path="synthetic_data.npy"):
    """Load pre-saved data"""
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        print(f"Loaded {len(data)} trajectories from {path}")
        return data
    else:
        print(f"No data found at {path}")
        return None


def visualize_trajectory(trajectory, title="Trajectory Visualization"):
    """
    Visualize a single trajectory
    
    Parameters:
    -----------
    trajectory : numpy array
        Trajectory with shape (n_steps, 6) where columns are [t, e_y, e_z, o_y, o_z, h]
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot end-effector trajectory
    plt.subplot(1, 2, 1)
    plt.plot(trajectory[:, 1], trajectory[:, 2], 'b-', label='End-effector')
    plt.scatter(trajectory[0, 1], trajectory[0, 2], c='g', s=50, label='Start')
    plt.scatter(trajectory[-1, 1], trajectory[-1, 2], c='r', s=50, label='End')
    plt.xlabel('Y position')
    plt.ylabel('Z position')
    plt.title('End-effector Trajectory')
    plt.legend()
    plt.grid(True)
    
    # Plot object trajectory
    plt.subplot(1, 2, 2)
    plt.plot(trajectory[:, 3], trajectory[:, 4], 'r-', label='Object')
    plt.scatter(trajectory[0, 3], trajectory[0, 4], c='g', s=50, label='Start')
    plt.scatter(trajectory[-1, 3], trajectory[-1, 4], c='r', s=50, label='End')
    plt.xlabel('Y position')
    plt.ylabel('Z position')
    plt.title('Object Trajectory')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('trajectory_visualization.png')
    plt.close()


def visualize_prediction(model, trajectory, num_context=5, title="Model Prediction"):
    """
    Visualize model prediction on a trajectory
    
    Parameters:
    -----------
    model : RobotCNMP
        Trained model
    trajectory : numpy array
        Trajectory with shape (n_steps, 6) where columns are [t, e_y, e_z, o_y, o_z, h]
    num_context : int
        Number of context points to use
    title : str
        Plot title
    """
    model.model.eval()
    
    traj_tensor = torch.tensor(trajectory, dtype=torch.float32, device=model.device)
    
    # Get dimensions
    t = traj_tensor[:, 0:1]  # Time
    target_dims = traj_tensor[:, 1:5]  # (e_y, e_z, o_y, o_z)
    h = traj_tensor[:, 5:6]  # Height
    
    # Randomly select context points
    seq_len = traj_tensor.shape[0]
    context_idxs = np.random.choice(seq_len, size=num_context, replace=False)
    
    # Create context observations
    context_t = torch.cat([t[context_idxs], h[context_idxs]], dim=1)
    context_obs = torch.cat([context_t, target_dims[context_idxs]], dim=1)
    
    # Create target queries - predict for all points
    target_query = torch.cat([t, h], dim=1)
    
    # Add batch dimension
    context_obs = context_obs.unsqueeze(0)  # [1, n_context, d_x+d_y]
    target_query = target_query.unsqueeze(0)  # [1, n_targets, d_x]
    
    # Get predictions
    with torch.no_grad():
        mean, std = model.model.forward(context_obs, target_query)
    
    # Remove batch dimension
    mean = mean.squeeze(0)
    std = std.squeeze(0)
    
    # Convert to numpy
    t_np = t.cpu().numpy()
    mean_np = mean.cpu().numpy()
    std_np = std.cpu().numpy()
    target_np = target_dims.cpu().numpy()
    context_t_np = t[context_idxs].cpu().numpy()
    context_target_np = target_dims[context_idxs].cpu().numpy()
    
    # Plotting
    plt.figure(figsize=(16, 8))
    
    # End-effector y-coordinate
    plt.subplot(2, 2, 1)
    plt.plot(t_np, target_np[:, 0], 'b-', label='Ground Truth')
    plt.plot(t_np, mean_np[:, 0], 'r--', label='Prediction')
    plt.fill_between(t_np.flatten(), 
                    (mean_np[:, 0] - 2 * std_np[:, 0]).flatten(), 
                    (mean_np[:, 0] + 2 * std_np[:, 0]).flatten(), 
                    alpha=0.2, color='r')
    plt.scatter(context_t_np, context_target_np[:, 0], c='g', s=50, label='Context Points')
    plt.xlabel('Time')
    plt.ylabel('End-effector Y')
    plt.title('End-effector Y Position')
    plt.legend()
    plt.grid(True)
    
    # End-effector z-coordinate
    plt.subplot(2, 2, 2)
    plt.plot(t_np, target_np[:, 1], 'b-', label='Ground Truth')
    plt.plot(t_np, mean_np[:, 1], 'r--', label='Prediction')
    plt.fill_between(t_np.flatten(), 
                    (mean_np[:, 1] - 2 * std_np[:, 1]).flatten(), 
                    (mean_np[:, 1] + 2 * std_np[:, 1]).flatten(), 
                    alpha=0.2, color='r')
    plt.scatter(context_t_np, context_target_np[:, 1], c='g', s=50, label='Context Points')
    plt.xlabel('Time')
    plt.ylabel('End-effector Z')
    plt.title('End-effector Z Position')
    plt.legend()
    plt.grid(True)
    
    # Object y-coordinate
    plt.subplot(2, 2, 3)
    plt.plot(t_np, target_np[:, 2], 'b-', label='Ground Truth')
    plt.plot(t_np, mean_np[:, 2], 'r--', label='Prediction')
    plt.fill_between(t_np.flatten(), 
                    (mean_np[:, 2] - 2 * std_np[:, 2]).flatten(), 
                    (mean_np[:, 2] + 2 * std_np[:, 2]).flatten(), 
                    alpha=0.2, color='r')
    plt.scatter(context_t_np, context_target_np[:, 2], c='g', s=50, label='Context Points')
    plt.xlabel('Time')
    plt.ylabel('Object Y')
    plt.title('Object Y Position')
    plt.legend()
    plt.grid(True)
    
    # Object z-coordinate
    plt.subplot(2, 2, 4)
    plt.plot(t_np, target_np[:, 3], 'b-', label='Ground Truth')
    plt.plot(t_np, mean_np[:, 3], 'r--', label='Prediction')
    plt.fill_between(t_np.flatten(), 
                    (mean_np[:, 3] - 2 * std_np[:, 3]).flatten(), 
                    (mean_np[:, 3] + 2 * std_np[:, 3]).flatten(), 
                    alpha=0.2, color='r')
    plt.scatter(context_t_np, context_target_np[:, 3], c='g', s=50, label='Context Points')
    plt.xlabel('Time')
    plt.ylabel('Object Z')
    plt.title('Object Z Position')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.close() 