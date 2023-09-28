#!/bin/bash

# EDIT YOUR CUSTOM COMMAND BELOW

# Set default values
CUDA_PREFIX=""
PORT=12300

# Check if an argument is provided
if [ "$#" -ne 0 ]; then
    # Use the provided GPU ID for the CUDA_VISIBLE_DEVICES prefix
    GPU_ID=$1
    # Extract the first GPU id if multiple are provided
    FIRST_GPU_ID=$(echo $GPU_ID | cut -d, -f1)
    # Set the CUDA prefix and port using the first GPU id
    CUDA_PREFIX="CUDA_VISIBLE_DEVICES=$GPU_ID"
    PORT=$((PORT + FIRST_GPU_ID))
    
    # Count the number of GPUs provided
    NUM_GPUS=$(echo $GPU_ID | tr -cd ',' | wc -c)
    (( NUM_GPUS++ ))
    
    INFO_MESSAGE="Running on GPU $GPU_ID"
else
    # Count the number of available GPUs in the system
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    
    INFO_MESSAGE="Running on all $NUM_GPUS GPUs ..."
fi

# Run the command
echo "$INFO_MESSAGE and master_port=$PORT"

# ENTER YOUR COMMAND
env $CUDA_PREFIX torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=localhost \
    --master_port=$PORT \
    languini/projects/gpt/main.py mini \
    --gradient_accumulation_steps 1 \
    --eval_every 1000 \
    --log_terminal_every 20 \
    --log_metrics_every 20 \
    --log_grads_every 1000