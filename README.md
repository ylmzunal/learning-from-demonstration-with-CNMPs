# CMPE591: Homework 4 - Learning from Demonstration with CNMPs

This repository contains the implementation of Homework 4 for CMPE591, which focuses on learning from demonstration using Conditional Neural Processes (CNPs). The goal is to train a CNP model to predict robot end-effector and object positions based on partial demonstrations and object height.

## Overview

In this homework, we collect demonstrations that consist of (t, e_y, e_z, o_y, o_z, h) where:
- t is the time (query dimension)
- e_y, e_z are the end-effector cartesian coordinates
- o_y, o_z are the object cartesian coordinates
- h is the height of the object (condition)

The robot randomly moves its end-effector in the y-z plane, sometimes hitting the object and sometimes not. The height of the object is random and is provided from the environment.

We train a CNP model to predict the end-effector and object positions given:
1. The time t (query dimension)
2. The object height h (condition)
3. Context points consisting of other (t, e_y, e_z, o_y, o_z) points

## Project Structure

```
.
├── cnp_model.py         # Implementation of the Conditional Neural Process model
├── train.py             # Script to collect demonstrations and train the CNP model
├── test.py              # Script to test the trained model
├── data/                # Directory to store collected demonstrations and results
├── model/               # Directory to store trained models
└── README.md            # This file
```

## Setup and Requirements

The code requires the following packages:
- PyTorch >= 2.0.0 (for MPS support on Apple Silicon)
- NumPy
- Matplotlib
- tqdm
- scikit-learn

Install the requirements with:
```bash
pip install -r requirements.txt
```

## Usage

The code is designed to be run with just two scripts: `train.py` and `test.py`. Each script will automatically generate demonstrations if needed, so there's no dependency on external data collection processes.

### 1. Training the CNP Model

To train the CNP model, run:

```bash
python train.py [--data_path DATA_PATH] [--hidden_size HIDDEN_SIZE] [--num_hidden_layers NUM_HIDDEN_LAYERS] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS] [--min_context MIN_CONTEXT] [--max_context MAX_CONTEXT] [--min_std MIN_STD] [--val_split VAL_SPLIT] [--device DEVICE] [--num_demonstrations NUM_DEMONSTRATIONS] [--model_dir MODEL_DIR] [--model_name MODEL_NAME]
```

Arguments:
- `--data_path`: Path to the demonstration data (default: 'data/demonstrations.pkl')
- `--hidden_size`: Hidden size of the model (default: 128)
- `--num_hidden_layers`: Number of hidden layers (default: 3)
- `--learning_rate`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of epochs (default: 100)
- `--min_context`: Minimum number of context points (default: 3)
- `--max_context`: Maximum number of context points (default: 20)
- `--min_std`: Minimum standard deviation (default: 0.05)
- `--val_split`: Validation split ratio (default: 0.2)
- `--device`: Device to use (auto, cuda, mps, cpu) (default: 'auto')
- `--num_demonstrations`: Number of demonstrations to collect if data file not found (default: 500)
- `--model_dir`: Directory to save the trained model (default: 'model')
- `--model_name`: Custom name for the model file (default: cnp_model_TIMESTAMP.pt)

For Apple Silicon (M1, M2, M3) Macs, you can take advantage of MPS acceleration:

```bash
python train.py --device mps --hidden_size 256 --batch_size 64 --num_epochs 50
```

The script will:
1. Generate demonstrations if the specified data file doesn't exist
2. Train the CNP model on the demonstrations
3. Save the trained model to the specified directory
4. Plot and save the training/validation loss curves

### 2. Testing the CNP Model

To test the trained model, run:

```bash
python test.py --model_path MODEL_PATH [--data_path DATA_PATH] [--num_tests NUM_TESTS] [--num_plots NUM_PLOTS] [--device DEVICE] [--output_dir OUTPUT_DIR] [--num_demonstrations NUM_DEMONSTRATIONS]
```

Arguments:
- `--model_path`: Path to the trained model (required)
- `--data_path`: Path to the demonstration data (default: 'data/demonstrations.pkl')
- `--num_tests`: Number of test cases (default: 100)
- `--num_plots`: Number of examples to plot (default: 3)
- `--device`: Device to use (auto, cuda, mps, cpu) (default: 'auto')
- `--output_dir`: Directory to save output files (default: 'data')
- `--num_demonstrations`: Number of demonstrations to collect if data file not found (default: 500)

For Apple Silicon (M1, M2, M3) Macs:

```bash
python test.py --model_path model/cnp_model_YYYYMMDD_HHMMSS.pt --device mps
```

The script will:
1. Load the trained model
2. Generate or load demonstrations
3. Create a test set with randomly sampled context points
4. Compute MSE metrics for end-effector and object predictions
5. Plot and save error bars and prediction visualizations

## Results

The test script produces:
1. MSE statistics for both end-effector and object predictions
2. Error bar plots comparing the MSE for end-effector and object predictions
3. Visualization of predictions vs. ground truth for selected test cases

## Optimized Settings for MacBook Pro M3

For optimal performance on Apple Silicon (M3):
```bash
# Training
python train.py --device mps --hidden_size 256 --batch_size 64 --num_epochs 50 --learning_rate 0.001

# Testing
python test.py --model_path model/cnp_model_YYYYMMDD_HHMMSS.pt --device mps
```

## Acknowledgements

The homework is based on CMPE591: Deep Learning in Robotics course materials.
