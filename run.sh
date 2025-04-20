#!/bin/bash

# Create directories if they don't exist
mkdir -p data model

# Step 1: Collect demonstrations
echo "Step 1: Collecting demonstrations..."
python collect_data.py

# Step 2: Train the CNP model with optimized parameters for M3 MacBook Pro
echo "Step 2: Training the CNP model..."
python train.py --num_epochs 50 --hidden_size 256 --num_hidden_layers 3 --batch_size 64 --device mps --learning_rate 0.001

# Get the most recent model file
MODEL_FILE=$(ls -t model/cnp_model_*.pt | head -1)

# Step 3: Test the CNP model
echo "Step 3: Testing the CNP model..."
python test.py 

echo "Done! Results are saved in the data/ directory." 