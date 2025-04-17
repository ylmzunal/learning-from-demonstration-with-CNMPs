import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from cnp import CNP
from utils import generate_synthetic_data, save_data, load_data, visualize_trajectory, RobotCNMP

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def train_with_trajectories(n_trajectories, epochs=50):
    """Train a model with a specific number of trajectories"""
    print(f"\n=== Training with {n_trajectories} trajectories ===")
    
    # Generate new synthetic data
    train_data = generate_synthetic_data(n_trajectories=n_trajectories, n_points=100)
    
    # Split into training and validation sets (80/20 split)
    np.random.shuffle(train_data)
    val_size = int(len(train_data) * 0.2)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]
    
    print(f"Training set: {len(train_data)} trajectories")
    print(f"Validation set: {len(val_data)} trajectories")
    
    # Create and train the model
    model = RobotCNMP(hidden_size=128, num_hidden_layers=3)
    
    # Train for specified epochs
    train_losses = []
    val_ee_mse = []
    val_obj_mse = []
    
    for epoch in range(epochs):
        # Train for one epoch
        loss = model._train_epoch(train_data, batch_size=16)
        train_losses.append(loss)
        
        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Evaluate on validation set
            val_results = model.test(val_data)
            val_ee_mse.append(val_results['ee_mse_mean'])
            val_obj_mse.append(val_results['obj_mse_mean'])
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}, Validation EE MSE: {val_results['ee_mse_mean']:.6f}, Object MSE: {val_results['obj_mse_mean']:.6f}")
    
    # Get final validation results
    final_val_results = model.test(val_data)
    
    # Save model
    model.save_model(f"cnmp_model_{n_trajectories}_traj.pt")
    
    return {
        'n_trajectories': n_trajectories,
        'final_ee_mse': final_val_results['ee_mse_mean'],
        'final_obj_mse': final_val_results['obj_mse_mean'],
        'ee_mse_std': final_val_results['ee_mse_std'],
        'obj_mse_std': final_val_results['obj_mse_std'],
        'train_losses': train_losses,
        'val_ee_mse': val_ee_mse,
        'val_obj_mse': val_obj_mse
    }

def main():
    # Start time
    start_time = time.time()
    
    # Test with different numbers of trajectories
    trajectory_counts = [25, 50, 100, 200]
    results = []
    
    for n_traj in trajectory_counts:
        result = train_with_trajectories(n_traj, epochs=50)
        results.append(result)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot end-effector MSE vs number of trajectories
    plt.subplot(2, 1, 1)
    ee_means = [r['final_ee_mse'] for r in results]
    ee_stds = [r['ee_mse_std'] for r in results]
    plt.errorbar(trajectory_counts, ee_means, yerr=ee_stds, marker='o', linestyle='-', capsize=5)
    plt.xlabel('Number of Trajectories')
    plt.ylabel('End-effector MSE')
    plt.title('End-effector Prediction Error vs Number of Training Trajectories')
    plt.grid(True)
    
    # Plot object MSE vs number of trajectories
    plt.subplot(2, 1, 2)
    obj_means = [r['final_obj_mse'] for r in results]
    obj_stds = [r['obj_mse_std'] for r in results]
    plt.errorbar(trajectory_counts, obj_means, yerr=obj_stds, marker='o', linestyle='-', capsize=5)
    plt.xlabel('Number of Trajectories')
    plt.ylabel('Object MSE')
    plt.title('Object Prediction Error vs Number of Training Trajectories')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('trajectory_count_vs_performance.png')
    plt.close()
    
    # Print summary table
    print("\n=== Results Summary ===")
    print(f"{'Trajectories':<15}{'End-effector MSE':<20}{'Object MSE':<20}")
    print("-" * 55)
    for r in results:
        print(f"{r['n_trajectories']:<15}{r['final_ee_mse']:.6f} ± {r['ee_mse_std']:.6f}{r['final_obj_mse']:.6f} ± {r['obj_mse_std']:.6f}")
    
    # Print execution time
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print("Experiments completed successfully!")


if __name__ == "__main__":
    main() 