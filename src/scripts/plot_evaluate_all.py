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
    "once": f"/gpu-data4/dbek/thesis/pruned_finetuned/{model}/batch_{batch}/evaluation/{algo}/balanced_{_samples}/importance_once/",
    "not_LTH": f"/gpu-data4/dbek/thesis/pruned_finetuned/{model}/batch_{batch}/evaluation/{algo}/balanced_{_samples}/not_LTH/",
    "igspf": f"/gpu-data4/dbek/thesis/pruned_finetuned/{model}/batch_{batch}/evaluation/{algo}/",
    "igspp": f"/gpu-data4/dbek/thesis/pruned_finetuned/{model}/batch_{batch}/evaluation/IGSPP/",
    "one_shot_ISP": f"/gpu-data4/dbek/thesis/pruned_finetuned/{model}/batch_{batch}/evaluation/one_shot_ISP/",
}

final_base_path = f"/gpu-data4/dbek/thesis/pruned_finetuned/{model}/batch_{batch}/evaluation/all/"

tuning_base_path = TUNING_BASE_PATH
sorted_directories = get_sorted_directories(base_path)


initial_heads = 144
heads_pruned_per_iteration = 14

# Get eval scores manually ##
once_evaluation_scores_file = os.path.join(base_paths2["once"], 'evaluation_scores.pkl')

once_evaluation_scores = load_evaluation_scores(once_evaluation_scores_file)
once_evaluation_scores = prepend_non_pruned_scores(once_evaluation_scores, tuning_base_path)

##
not_evaluation_scores_file = os.path.join(base_paths2["not_LTH"], 'evaluation_scores.pkl')

not_evaluation_scores = load_evaluation_scores(not_evaluation_scores_file)
not_evaluation_scores = prepend_non_pruned_scores(not_evaluation_scores, tuning_base_path)

##
igspf_evaluation_scores_file = os.path.join(base_paths2["igspf"], 'evaluation_scores.pkl')

igspf_evaluation_scores = load_evaluation_scores(igspf_evaluation_scores_file)
igspf_evaluation_scores = prepend_non_pruned_scores(igspf_evaluation_scores, tuning_base_path)

##
igspp_evaluation_scores_file = os.path.join(base_paths2["igspp"], 'evaluation_scores.pkl')

igspp_evaluation_scores = load_evaluation_scores(igspp_evaluation_scores_file)
igspp_evaluation_scores = prepend_non_pruned_scores(igspp_evaluation_scores, tuning_base_path)

##
one_shot_evaluation_scores_file = os.path.join(base_paths2["one_shot_ISP"], 'evaluation_scores.pkl')

one_shot_evaluation_scores = load_evaluation_scores(one_shot_evaluation_scores_file)
one_shot_evaluation_scores = prepend_non_pruned_scores(one_shot_evaluation_scores, tuning_base_path)

for task, head_mask_dirs in sorted_directories.items():
    head_masks = [head_mask for head_mask, _ in head_mask_dirs]
    
    once_scores = once_evaluation_scores.get(task, [])
    not_scores = not_evaluation_scores.get(task, [])
    igspf_scores = igspf_evaluation_scores.get(task, [])
    igspp_scores = igspp_evaluation_scores.get(task, [])
    one_shot_scores = one_shot_evaluation_scores.get(task, [])
    
    remaining_heads = [initial_heads - i * heads_pruned_per_iteration for i in range(len(head_masks) + 1)]
    pruned_heads = [initial_heads - remaining for remaining in remaining_heads]

    ylabel = 'Accuaracy'

    if task == 'cola':
        ylabel = 'MCC'

    if task == 'mnli':
        ylabel = 'Matched Accuracy'

    plt.figure()
    plt.plot(pruned_heads, once_scores, marker='.', linestyle='-', label="One importance")
    plt.plot(pruned_heads, not_scores, marker='.', linestyle='-', label="Importance Score")
    plt.plot(pruned_heads, igspf_scores, marker='.', linestyle='-', label="IGSPF")
    plt.plot(pruned_heads, igspp_scores, marker='.', linestyle='-', label="IGSPP")
    plt.plot(pruned_heads, one_shot_scores, marker='.', linestyle='-', label="one shot ISP")
    plt.axhline(y=once_scores[0], color='purple', linestyle='--', linewidth=0.5, label='Initial Value')

    plt.xticks(pruned_heads)
    plt.xlabel('Number of Pruned Heads')
    plt.ylabel(ylabel)
    plt.title(f'{task.upper()} (BERT base uncased)')
    plt.legend()
    plt.grid(True)
    output_dir = os.path.join(final_base_path, f"{task}")
    create_directories(output_dir)
    file_name = os.path.join(output_dir, "scores_all.png")


    plt.savefig(file_name)
    plt.close()