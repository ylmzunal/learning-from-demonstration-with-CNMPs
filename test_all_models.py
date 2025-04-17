import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from utils import load_data, RobotCNMP

# Load test data
test_data = load_data("synthetic_test_data.npy")

# Models to test
model_paths = [
    "cnmp_model_25_traj.pt", 
    "cnmp_model_50_traj.pt", 
    "cnmp_model_100_traj.pt", 
    "cnmp_model_200_traj.pt"
]

# Test each model
results = []

for path in model_paths:
    try:
        # Extract number of trajectories using regex
        match = re.search(r'_(\d+)_traj', path)
        if match:
            n_traj = int(match.group(1))
        else:
            print(f"Could not parse trajectory count from {path}")
            continue
        
        # Create model and load weights
        model = RobotCNMP(hidden_size=128, num_hidden_layers=3)
        model.load_model(path)
        
        # Test on full test set
        print(f"\nTesting model trained with {n_traj} trajectories...")
        metrics = model.test(test_data)
        
        # Store results
        results.append({
            'n_trajectories': n_traj,
            'ee_mse_mean': metrics['ee_mse_mean'],
            'ee_mse_std': metrics['ee_mse_std'],
            'obj_mse_mean': metrics['obj_mse_mean'],
            'obj_mse_std': metrics['obj_mse_std']
        })
    except Exception as e:
        print(f"Error testing model {path}: {e}")

# Sort results by number of trajectories
results.sort(key=lambda x: x['n_trajectories'])

# Plot results
plt.figure(figsize=(15, 10))

# Plot end-effector MSE vs number of trajectories
plt.subplot(2, 1, 1)
trajectory_counts = [r['n_trajectories'] for r in results]
ee_means = [r['ee_mse_mean'] for r in results]
ee_stds = [r['ee_mse_std'] for r in results]
plt.errorbar(trajectory_counts, ee_means, yerr=ee_stds, marker='o', linestyle='-', capsize=5)
plt.xlabel('Number of Training Trajectories')
plt.ylabel('End-effector MSE')
plt.title('End-effector Prediction Error vs Number of Training Trajectories')
plt.grid(True)

# Plot object MSE vs number of trajectories
plt.subplot(2, 1, 2)
obj_means = [r['obj_mse_mean'] for r in results]
obj_stds = [r['obj_mse_std'] for r in results]
plt.errorbar(trajectory_counts, obj_means, yerr=obj_stds, marker='o', linestyle='-', capsize=5)
plt.xlabel('Number of Training Trajectories')
plt.ylabel('Object MSE')
plt.title('Object Prediction Error vs Number of Training Trajectories')
plt.grid(True)

plt.tight_layout()
plt.savefig('test_performance_vs_trajectories.png')
plt.show()

# Print summary table
print("\n=== Results Summary (Test Set) ===")
print(f"{'Trajectories':<15}{'End-effector MSE':<20}{'Object MSE':<20}")
print("-" * 55)
for r in results:
    print(f"{r['n_trajectories']:<15}{r['ee_mse_mean']:.6f} ± {r['ee_mse_std']:.6f}{r['obj_mse_mean']:.6f} ± {r['obj_mse_std']:.6f}")

# Analysis of diminishing returns
if len(results) > 1:
    print("\n=== Analysis of Diminishing Returns ===")
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        ee_improvement = (prev['ee_mse_mean'] - curr['ee_mse_mean']) / prev['ee_mse_mean'] * 100
        obj_improvement = (prev['obj_mse_mean'] - curr['obj_mse_mean']) / prev['obj_mse_mean'] * 100
        traj_increase = (curr['n_trajectories'] - prev['n_trajectories']) / prev['n_trajectories'] * 100
        
        print(f"Increasing from {prev['n_trajectories']} to {curr['n_trajectories']} trajectories ({traj_increase:.1f}% increase):")
        print(f"  End-effector MSE improved by {ee_improvement:.2f}%")
        print(f"  Object MSE improved by {obj_improvement:.2f}%") 