#!/bin/bash

# Define the base directory
base_dir="/gpu-data4/dbek/thesis/tuning/UNCASED/batch_32"

# Array of tasks 
tasks=("mrpc" "sst2" "wnli" "rte" "qqp")

# Array of seeds
seeds=("seed_17" "seed_128" "seed_42")

# Loop through each task and seed
for task in "${tasks[@]}"; do
  for seed in "${seeds[@]}"; do
    # Create the new directory structure
    mkdir -p "${base_dir}/${task}/${seed}"

    # Move the files from the old structure to the new one
    mv "${base_dir}/${seed}/${task}/"* "${base_dir}/${task}/${seed}/"
  done
done
