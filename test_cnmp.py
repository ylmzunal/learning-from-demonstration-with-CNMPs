import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from cnp import CNP
from utils import generate_synthetic_data, save_data, load_data, visualize_prediction, RobotCNMP

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def main():
    # Start time
    start_time = time.time()
    
    # Generate or load test data
    test_data = load_data("synthetic_test_data.npy")
    if test_data is None:
        print("Generating synthetic test data...")
        test_data = generate_synthetic_data(n_trajectories=100, n_points=100)
        save_data(test_data, "synthetic_test_data.npy")
    
    # Load a trained model
    model = RobotCNMP(hidden_size=128, num_hidden_layers=3)
    model.load_model()
    
    # Visualize a prediction
    visualize_prediction(model, test_data[0], num_context=5, title="Model Prediction Example")
    
    # Test the model
    metrics = model.test(test_data)
    
    # Print metrics
    print("\nTest Results:")
    print(f"End-effector MSE: {metrics['ee_mse_mean']:.6f} ± {metrics['ee_mse_std']:.6f}")
    print(f"Object MSE: {metrics['obj_mse_mean']:.6f} ± {metrics['obj_mse_std']:.6f}")
    
    # Print execution time
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Done!
    print("Testing completed successfully!")


if __name__ == "__main__":
    main() 