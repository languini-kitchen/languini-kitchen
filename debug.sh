#!/bin/bash

# debug script for running on single CPU/GPU

project=gpt
# project=lstm

export WANDB_MODE=disabled

# use "yes" to automatically confirm overwrties of old project folders
./venv/bin/torchrun -m pdb languini/projects/$project/main.py tiny --train_batch_size 8 --gradient_accumulation_steps 4 --max_train_steps 512 --decay_steps 512 --compile None --logger_type tb

