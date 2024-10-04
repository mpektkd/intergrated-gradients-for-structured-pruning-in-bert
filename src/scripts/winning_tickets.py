import numpy as np

import argparse

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

def find_winning_tickets(pruned_metrics, full_model_metrics):
    winning_tickets = {}

    for task, metrics in pruned_metrics.items():
        pruned_means = metrics['mean']
        pruned_stds = metrics['std']
        full_model_mean, full_model_std = full_model_metrics[task]

        deepest_pruning_ratio = None

        for i in range(len(pruned_means)):
            pruned_mean = pruned_means[i]
            pruned_std = pruned_stds[i]

            upper_bound = pruned_mean + pruned_std

            # Check if the full model performance is within one standard deviation of the pruned subnetwork performance
            
            if full_model_mean <= upper_bound:
                deepest_pruning_ratio = i + 1  # Update the deepest pruning ratio where this condition holds

        # Store the deepest pruning ratio that satisfies the winning ticket condition
        winning_tickets[task] = deepest_pruning_ratio

        if deepest_pruning_ratio is None:
            print(task, None)
        else:
            print(task, deepest_pruning_ratio * 10, pruned_means[deepest_pruning_ratio-1], pruned_stds[deepest_pruning_ratio-1])

    return winning_tickets


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
        algo = 'one_shot_ISP'

    import pickle

    with open(f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/winning_tickets/{algo}/mean_std.pkl", "rb") as f3:
        pruned_metrics = pickle.load(f3)

    with open(f"/gpu-data4/dbek/thesis/tuning/UNCASED/batch_32/evaluation/mean_std.pkl", "rb") as ft:
        full_model_metrics = pickle.load(ft)

    result = find_winning_tickets(pruned_metrics, full_model_metrics)
    print(result)