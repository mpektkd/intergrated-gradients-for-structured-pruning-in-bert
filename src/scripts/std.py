import torch
import numpy as np
import sys

import argparse
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

def compute_mean_std(file1, file2, file3):
    tasks = file1.keys()  # Assume all files have the same keys

    mean_std_results = {}

    for task in tasks:
        values_1 = [v.item() if isinstance(v, torch.Tensor) else v for v in file1[task]]
        values_2 = [v.item() if isinstance(v, torch.Tensor) else v for v in file2[task]]
        values_3 = [v.item() if isinstance(v, torch.Tensor) else v for v in file3[task]]

        # Combine the values from all three files
        combined_values = np.array([values_1, values_2, values_3])

        # Calculate mean and standard deviation across the three seeds
        mean_values = np.mean(combined_values, axis=0)
        std_values = np.std(combined_values, axis=0)

        mean_std_results[task] = {
            'mean': mean_values,
            'std': std_values
        }

    return mean_std_results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--igspf", action="store_true")
    
    parser.add_argument("--igspp", action="store_true")
    
    parser.add_argument("--one_shot", action="store_true")

    args = parser.parse_args()
    
    if args.igspf:
        algo = 'IGSPF'
    elif args.igspp:
        algo = 'IGSPP'
    elif args.one_shot:

        import pickle 

        with open(f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_42/IGSPF/corr/target_None/one_shot_ISP/evaluation/evaluation_scores.pkl", "rb") as f1:
            file1 = pickle.load(f1)

        with open(f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_17/IGSPF/corr/target_None/one_shot_ISP/evaluation/evaluation_scores.pkl", "rb") as f2:
            file2 = pickle.load(f2)

        with open(f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_128/IGSPF/corr/target_None/one_shot_ISP/evaluation/evaluation_scores.pkl", "rb") as f3:
            file3 = pickle.load(f3)


        result = compute_mean_std(file1, file2, file3)

        with open(f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/winning_tickets/one_shot_ISP/mean_std.pkl", "wb") as f4:
            pickle.dump(result, f4)

            
        sys.exit()


    import pickle 

    with open(f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_42/{algo}/corr/target_None/evaluation/evaluation_scores.pkl", "rb") as f1:
        file1 = pickle.load(f1)

    with open(f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_17/{algo}/corr/target_None/evaluation/evaluation_scores.pkl", "rb") as f2:
        file2 = pickle.load(f2)

    with open(f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_128/{algo}/corr/target_None/evaluation/evaluation_scores.pkl", "rb") as f3:
        file3 = pickle.load(f3)


    result = compute_mean_std(file1, file2, file3)

    with open(f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/winning_tickets/{algo}/mean_std.pkl", "wb") as f4:
        pickle.dump(result, f4)

        
        
