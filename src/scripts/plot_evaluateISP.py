import numpy as np
import os
import re
import pickle
import json
import matplotlib.pyplot as plt

from variables import *

def is_empty_directory(path):
    return not os.listdir(path)

def get_sorted_directories(base_path):
    task_dict = {}
    for root, dirs, files in os.walk(base_path):
        match = re.search(rf'/target_None/(?P<task>[^/]+)/balanced_samples_{_samples}/head_mask_(?P<head_mask>\d+)$', root)
        if match:
            task = match.group('task')
            head_mask = int(match.group('head_mask'))
            if task not in task_dict:
                task_dict[task] = []
            task_dict[task].append((head_mask, root))
    sorted_tasks = sorted(task_dict.keys())
    sorted_task_dict = {}
    for task in sorted_tasks:
        non_empty_dirs = [(head_mask, directory) for head_mask, directory in task_dict[task] if not is_empty_directory(directory)]
        if non_empty_dirs:
            sorted_task_dict[task] = sorted(non_empty_dirs, key=lambda x: x[0])
    return sorted_task_dict

def load_evaluation_scores(file_path):
    with open(file_path, 'rb') as f:
        evaluation_scores = pickle.load(f)
    return evaluation_scores

def prepend_non_pruned_scores(evaluation_scores, tuning_base_path):
    for root, dirs, files in os.walk(tuning_base_path):
        if 'eval_results.json' in files:
            task = os.path.basename(root)
            eval_results_path = os.path.join(root, 'eval_results.json')
            with open(eval_results_path, 'r') as f:
                eval_results = json.load(f)
                eval_accuracy = eval_results.get('eval_accuracy')
                if eval_accuracy is None:
                    eval_accuracy = eval_results.get('eval_matthews_correlation')
                if eval_accuracy is not None:
                    if task in evaluation_scores:
                        evaluation_scores[task].insert(0, eval_accuracy)
    return evaluation_scores

def create_directories(path):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))
        print(f"Deleted all files inside the directory: {path}")
    else:
        os.makedirs(path)
        print(f"Created directories for path: {path}")


def plot_evaluation_scores(evaluation_scores, sorted_task_dict, base_path):

    initial_weights = 144  # Total number of weights (or heads in this context) in the full model
    prune_ratio = 0.1      # Ratio of weights to prune each iteration

    for task, head_mask_dirs in sorted_task_dict.items():
        head_masks = [head_mask for head_mask, head_mask_dir in head_mask_dirs if os.path.isfile(os.path.join(head_mask_dir, 'config.json'))]
        scores = evaluation_scores.get(task, [])
        if scores:
            survived_weights = [initial_weights]
            for i in range(len(head_masks)):
                new_survived = survived_weights[-1] * (1 - prune_ratio)
                survived_weights.append(new_survived)

            
            # Calculate percentage of survived weights
            survived_percentages = [(weights / initial_weights) * 100 for weights in survived_weights]
            survived_percentages = [round(percentage, 2) for percentage in survived_percentages]
            

            for (i, j) in zip(survived_percentages, scores):
                print(i, j)

            fig, plot1 = plt.subplots(1, 1)
            
            # Plotting
            plt.axhline(y=scores[0], color='purple', linestyle='--', linewidth=0.5, label='Initial Value')
            plot1.plot(survived_percentages, scores, marker='o', linestyle='-', label=task)
            
            # Determine x-ticks to avoid overlap
            if len(survived_percentages) > 10:  # If there are too many labels, skip some
                tick_indices = np.array([0, 5, 10, 15, 20, 25, 39])
                plot1.set_xticks([survived_percentages[i] for i in tick_indices])
                plot1.set_xticklabels([f'{survived_percentages[i]:.2f}%' for i in tick_indices], rotation=45, ha="right")
            else:
                plot1.set_xticks(survived_percentages)
                plot1.set_xticklabels([f'{p:.2f}%' for p in survived_percentages], rotation=45, ha="right")

            plot1.invert_xaxis()

            # Set labels and title
            plot1.set_xlabel('Percentage of Survived Weights')
            plot1.set_ylabel('Evaluation Score')
            plot1.set_title(f'Evaluation Scores for Task: {task}')
            plot1.legend()
            plot1.grid(True)

            # Adjust layout and save the figure
            output_dir = os.path.join(base_path, f"{task}")
            create_directories(output_dir)
            file_name = os.path.join(output_dir, "scores.png")
            fig.savefig(file_name, bbox_inches='tight')  # Save with tight layout to handle label rotation
            plt.close(fig)

base_path = EVALUATE_BASE_PATH
base_path2 = EVALUATE_BASE_PATH2
_samples = SAMPLES

evaluation_scores_file = os.path.join(base_path2, 'evaluation_scores.pkl')

tuning_base_path = TUNING_BASE_PATH

sorted_directories = get_sorted_directories(base_path)
evaluation_scores = load_evaluation_scores(evaluation_scores_file)

evaluation_scores = prepend_non_pruned_scores(evaluation_scores, tuning_base_path)
plot_evaluation_scores(evaluation_scores, sorted_directories, base_path2)
