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
    _scores = {}
    for root, dirs, files in os.walk(tuning_base_path):
        
        if 'eval_results.json' in files:
            task = os.path.basename(os.path.dirname(root))
            seed = os.path.basename(root)

            
            if seed != "seed_42":
                continue
            eval_results_path = os.path.join(root, 'eval_results.json')
            with open(eval_results_path, 'r') as f:
                eval_results = json.load(f)
                eval_accuracy = eval_results.get('eval_accuracy')
                if eval_accuracy is None:
                    eval_accuracy = eval_results.get('eval_matthews_correlation')
                if eval_accuracy is not None:
                    if _scores.get(task) is None:
                        _scores[task] = [eval_accuracy]
                    else:
                        _scores[task].append(eval_accuracy)
                    
    

    tuned_metrics = {}
    for task in list(_scores.keys()):
        
        mean_values = np.mean(np.array(_scores[task]), axis=0)
        std_values = np.std(np.array(_scores[task]), axis=0)


        tuned_metrics[task] = (mean_values, std_values)
    
        if task in evaluation_scores:
            evaluation_scores[task].insert(0, mean_values)

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

batch=32 # 32, 64
algo="IGSPF" # IGSPF, IGSPP, one_shot_IGSP 
model='UNCASED' # UNCASED, CASED
_samples=2000
_merge=merge

base_path = "/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_17/IGSPF/corr/target_None"
base_paths2 = {
    "one_shot_ISP": f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_42/IGSPF/corr/target_None/one_shot_ISP/evaluation",
    "ISP": f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/seed_42/ISP/corr/target_None/evaluation/"
}

final_base_path = f"/gpu-data4/dbek/thesis/pruned_finetuned/UNCASED/batch_32/evaluation/seed_42/iterative/"

tuning_base_path = TUNING_BASE_PATH
sorted_directories = get_sorted_directories(base_path)


initial_heads = 144
prune_ratio = 0.1      # Ratio of weights to prune each iteration

heads_pruned_per_iteration = 14

# Get eval scores manually ##
one_shot_ISP_evaluation_scores_file = os.path.join(base_paths2["one_shot_ISP"], 'evaluation_scores.pkl')

one_shot_ISP_evaluation_scores = load_evaluation_scores(one_shot_ISP_evaluation_scores_file)
one_shot_ISP_evaluation_scores = prepend_non_pruned_scores(one_shot_ISP_evaluation_scores, tuning_base_path)

##
ISP_evaluation_scores_file = os.path.join(base_paths2["ISP"], 'evaluation_scores.pkl')

ISP_evaluation_scores = load_evaluation_scores(ISP_evaluation_scores_file)
ISP_evaluation_scores = prepend_non_pruned_scores(ISP_evaluation_scores, tuning_base_path)

for task, head_mask_dirs in sorted_directories.items():
    sorted_directoriesISP = get_sorted_directories(base_path)

    head_masks = [head_mask for head_mask, _ in head_mask_dirs]
    ISPhead_masks = [i for i in range(1, 41)]

    one_shot_ISP_scores = one_shot_ISP_evaluation_scores.get(task, [])
    ISP_scores = ISP_evaluation_scores.get(task, [])

    
    if ISP_scores:
        print(one_shot_ISP_scores)
            
        ylabel = 'Accuaracy'

        if task == 'cola':
            ylabel = 'MCC'

        if task == 'mnli':
            ylabel = 'Matched Accuracy'

        remaining_heads = [initial_heads - i * heads_pruned_per_iteration for i in range(len(head_masks) + 1)]
        pruned_heads = [initial_heads - remaining for remaining in remaining_heads]

        survived_percentages = list(reversed([(weights / initial_heads) * 100 for weights in pruned_heads]))
         

        ISPsurvived_weights = [initial_heads]
        for i in range(len(ISPhead_masks)):

            new_survived = ISPsurvived_weights[-1] * (1 - prune_ratio)
            ISPsurvived_weights.append(new_survived)

        # Calculate percentage of survived weights
        ISPsurvived_percentages = [(weights / initial_heads) * 100 for weights in ISPsurvived_weights]
        ISPsurvived_percentages = [round(percentage, 2) for percentage in ISPsurvived_percentages]
            

        # for (i, j) in zip(ISPsurvived_percentages, ISP_scores):
        #     print(i, j)

        fig, plot1 = plt.subplots(1, 1)

         # Plotting
        plt.axhline(y=ISP_scores[0], color='purple', linestyle='--', linewidth=0.5, label='Initial Value')
        plot1.plot(survived_percentages, one_shot_ISP_scores, marker='.', linestyle='-', label="one shot ISP")
        plot1.plot(ISPsurvived_percentages, ISP_scores, marker='.', linestyle='-', label="ISP")
        
        # Determine x-ticks to avoid overlap
        if len(ISPsurvived_percentages) > 10:  # If there are too many labels, skip some
            tick_indices = np.array([0, 5, 10, 15, 20, 25, 39])
            plot1.set_xticks([ISPsurvived_percentages[i] for i in tick_indices])
            plot1.set_xticklabels([f'{ISPsurvived_percentages[i]:.2f}%' for i in tick_indices], rotation=45, ha="right")
        else:
            plot1.set_xticks(ISPsurvived_percentages)
            plot1.set_xticklabels([f'{p:.2f}%' for p in ISPsurvived_percentages], rotation=45, ha="right")

        plot1.invert_xaxis()

        # Set labels and title
        plot1.set_xlabel('Percentage of Survived Weights')
        plot1.set_ylabel(ylabel)
        plot1.set_title(f'{task.upper()} (BERT base uncased)')
        plot1.legend()
        plot1.grid(True)

        # Adjust layout and save the figure
        output_dir = os.path.join(final_base_path, f"{task}")
        create_directories(output_dir)
        file_name = os.path.join(output_dir, "scores.png")
        fig.savefig(file_name, bbox_inches='tight')  # Save with tight layout to handle label rotation
        plt.close(fig)
