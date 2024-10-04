import os
import re
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np

from variables import *

def is_empty_directory(path):
    return not os.listdir(path)

def get_sorted_directories(base_path):
    task_dict = {}
    for root, dirs, files in os.walk(base_path):
        match = re.search(rf'/target_golden/one_shot_ISP/(?P<task>[^/]+)/balanced_samples_{_samples}/head_mask_(?P<head_mask>\d+)$', root)
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

def prepend_non_pruned_scores(evaluation_scores, tuning_base_path, _merge):
    _scores = {}
    for root, dirs, files in os.walk(tuning_base_path):
        
        if 'eval_results.json' in files:
            seed = os.path.basename(root)
            task = os.path.basename(os.path.dirname(root))

            if not _merge and seed != f"seed_{_seed}":
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

    if _merge:
        with open(os.path.join("/gpu-data4/dbek/thesis/tuning/UNCASED/batch_32/evaluation", 'mean_std.pkl'), "wb") as ft:
            pickle.dump(tuned_metrics, ft)

    print(tuned_metrics)

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

    initial_heads = 144
    heads_pruned_per_iteration = 14

    for task, head_mask_dirs in sorted_task_dict.items():
        head_masks = [head_mask for head_mask, _ in head_mask_dirs]
        scores = evaluation_scores.get(task, [])

        if scores:
            remaining_heads = [initial_heads - i * heads_pruned_per_iteration for i in range(len(head_masks) + 1)]
            pruned_heads = [initial_heads - remaining for remaining in remaining_heads]
            plt.figure()
            plt.plot(pruned_heads[:len(scores)], scores, marker='o', linestyle='-', label=task)
            plt.xticks(pruned_heads)
            plt.xlabel('Number of Pruned Heads')
            plt.ylabel('Evaluation Score')
            plt.title(f'Evaluation Scores for Task: {task}')
            plt.legend()
            plt.grid(True)
            output_dir = os.path.join(base_path, f"{task}")
            create_directories(output_dir)
            file_name = os.path.join(output_dir, "scores.png")
            plt.savefig(file_name)
            plt.close()


base_path = EVALUATE_BASE_PATH
base_path2 = EVALUATE_BASE_PATH2
_samples=SAMPLES
_merge=merge
_seed=SEED

evaluation_scores_file = os.path.join(base_path2, 'evaluation_scores.pkl')

tuning_base_path = TUNED_MODEL

sorted_directories = get_sorted_directories(base_path)
evaluation_scores = load_evaluation_scores(evaluation_scores_file)

evaluation_scores = prepend_non_pruned_scores(evaluation_scores, tuning_base_path, _merge)
plot_evaluation_scores(evaluation_scores, sorted_directories, base_path2)
