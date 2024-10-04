import os
import pickle
import torch
import numpy as np
import logging
# from variables import *

logger = logging.getLogger("merge")

# Define the paths to the directories
directories = [
    f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_17/IGSPF/corr/target_None/one_shot_ISP/evaluation",
    f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_42/IGSPF/corr/target_None/one_shot_ISP/evaluation",
    f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_128/IGSPF/corr/target_None/one_shot_ISP/evaluation"
]
average_base_path="/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/evaluation/one_shot_ISP/"

# Dictionary to accumulate the scores
accumulated_scores = {}
count = {}

import os

def create_directories(path):
    # Check if the path exists
    if os.path.exists(path):
        # Delete files inside the directory
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))
        print(f"Deleted all files inside the directory: {path}")
    else:
        # Create the directories recursively
        os.makedirs(path)
        print(f"Created directories for path: {path}")

create_directories(average_base_path)

log_file = average_base_path + "evaluation_logfile.log"

logging.basicConfig(
level=logging.INFO,  # Set the logging level
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Set the logging format
filename=log_file,  # Specify the file to save the logs
filemode='w'  # Set the file mode to write (overwrite the log file each time)
)

# Read each evaluation_scores.pkl file and accumulate the scores
for directory in directories:
    filepath = os.path.join(directory, "evaluation_scores.pkl")
    
    # Load the evaluation_scores.pkl file
    with open(filepath, 'rb') as file:
        scores = pickle.load(file)
        
    # Accumulate the scores for each dataset
    for dataset, values in scores.items():

        if dataset not in accumulated_scores:
            accumulated_scores[dataset] = np.zeros(len(values))
            count[dataset] = 0
        
        # Convert tensors to floats if necessary
        float_values = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        
        accumulated_scores[dataset] += np.array(float_values)
        count[dataset] += 1

# Calculate the average scores for each dataset
average_scores = {dataset: list(accumulated_scores[dataset] / count[dataset]) for dataset in accumulated_scores}

# Print the average scores
for dataset, avg_scores in average_scores.items():
    for i, score in enumerate(avg_scores, 1):
        logger.info(f"Task {dataset}, head_mask {i}:")
        logger.info("Score with pruning: %.4f", score)



# Save sorted boundaries dictionary to a pickle file
with open(average_base_path + 'evaluation_scores.pkl', 'wb') as f:
    pickle.dump(average_scores, f)
    