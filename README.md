# CMPE591 - Homework 4: Learning from Demonstration with CNMPs

This repository contains the implementation of a Conditional Neural Process (CNP) model for learning robot manipulation tasks from demonstrations. The model is trained to predict both end-effector and object positions based on observed context points.

## Problem Description

In this homework, we collect demonstrations consisting of robot end-effector and object trajectories and use Conditional Neural Processes (CNMPs) to learn from these demonstrations. The goal is to predict both the end-effector and object positions based on context points given the time and object height.

The dataset is represented as: {(t, e_y, e_z, o_y, o_z), h}^N_{i=0} where:
- t: time (query dimension)
- e_y, e_z: end-effector y and z coordinates
- o_y, o_z: object y and z coordinates
- h: height of the object (condition)

The model is trained to predict end-effector and object positions given a set of context points and query points.

## Implementation

The solution consists of the following components:

1. **Data Generation**: We generate synthetic data that mimics robot demonstrations. Each trajectory contains time, end-effector coordinates, object coordinates, and object height.

2. **CNMP Model**: We implement a Conditional Neural Process model that:
   - Takes a set of context points with all dimensions (t, h, e_y, e_z, o_y, o_z)
   - Predicts target dimensions (e_y, e_z, o_y, o_z) for a set of query points (t, h)

3. **Training**: The model is trained to minimize the negative log-likelihood of the target points given the context points.

4. **Evaluation**: We test the model on new randomly generated trajectories and compute the mean squared error between the predicted and ground truth values.

## Files

- `cnp.py`: Standalone implementation of the Conditional Neural Process model
- `utils.py`: Utility functions for data generation, visualization, and the RobotCNMP class
- `train_cnmp.py`: Script for training the CNMP model
- `test_cnmp.py`: Script for testing the trained CNMP model
- `README.md`: Documentation for the project

## How to Run

To train the model:

```bash
python train_cnmp.py
```

This will:
1. Generate 100 synthetic training trajectories (or load existing ones if available)
2. Train the CNMP model for 100 epochs
3. Save the trained model to disk

To test the model:

```bash
python test_cnmp.py
```

This will:
1. Generate 100 synthetic test trajectories (or load existing ones if available)
2. Load the trained model
3. Test the model on the test data
4. Generate visualization plots for model predictions and test errors

## Note on Implementation

While the original assignment involves using the provided robot environment to collect demonstrations, this implementation uses synthetic data that simulates similar trajectories. This approach allows us to demonstrate the core concepts of Conditional Neural Processes without requiring the complex environment setup.

The synthetic data generation creates trajectories that mimic the behavior of the robot's end-effector and the object it manipulates, with the object's movement influenced by both the end-effector movement and the object's height.

## Results

The model is evaluated on two metrics:
1. **End-effector prediction MSE**: Measures how well the model predicts the robot's end-effector position
2. **Object prediction MSE**: Measures how well the model predicts the object's position

We perform 100 test runs with randomly generated observations and queries and compute the mean and standard deviation of these errors. The results are visualized in a bar plot.

## Visualizations

The code generates several visualizations:
- `training_loss.png`: Plot of the training loss over epochs
- `trajectory_visualization.png`: Visualization of an example trajectory showing end-effector and object positions
- `prediction_visualization.png`: Visualization of model predictions against ground truth for an example trajectory
- `test_errors.png`: Bar plot showing the mean and standard deviation of prediction errors

## References

This implementation is based on the Conditional Neural Process model described in:
- Garnelo, M., Rosenbaum, D., Maddison, C., Ramalho, T., Saxton, D., Shanahan, M., Teh, Y. W., Rezende, D., & Eslami, S. M. A. (2018). Conditional Neural Processes. International Conference on Machine Learning (ICML).