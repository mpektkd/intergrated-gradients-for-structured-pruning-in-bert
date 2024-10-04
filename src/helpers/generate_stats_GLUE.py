import os
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np
from scipy.stats import skew
import json
import logging
import pickle
import matplotlib.cm as cm

def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_empty_region(counts, bin_edges):
    # Find the bin with the lowest count
    min_count_index = np.argmin(counts)
    min_count = counts[min_count_index]
    
    # Get the center of the bin with the lowest count
    x_position = (bin_edges[min_count_index] + bin_edges[min_count_index + 1]) / 2
    
    return x_position, min_count

def plot_histogram(data, title, xlabel, ylabel, filename, bins=30, filter_method=None, lower_bound=None, upper_bound=None):
    plt.figure(figsize=(10, 6))
    counts, bin_edges, _ = plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    max_value = max(data)
    min_value = min(data)
    mean_value = np.mean(data)
    std_value = np.std(data)
    skewness_value = skew(data)
    
    plt.axvline(max_value, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(min_value, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mean_value, color='b', linestyle='dashed', linewidth=1)
    plt.axvline(mean_value - std_value, color='purple', linestyle='dashed', linewidth=1)
    plt.axvline(mean_value + std_value, color='purple', linestyle='dashed', linewidth=1)
    
    # Annotate max, min, mean, std, and skewness values
    y_max = plt.ylim()[1]
    annotation_positions = [y_max*0.8, y_max*0.8, y_max*0.7, y_max*0.6, y_max*0.6]
    
    plt.text(max_value, annotation_positions[0], f'Max: {max_value}', color='r', ha='right', va='bottom', rotation=90)
    plt.text(min_value, annotation_positions[1], f'Min: {min_value}', color='r', ha='right', va='bottom', rotation=90)
    plt.text(mean_value, annotation_positions[2], f'Mean: {mean_value:.2f}', color='b', ha='right', va='bottom', rotation=90)
    plt.text(mean_value + std_value, annotation_positions[3], f'Mean + 1 Std: {mean_value + std_value:.2f}', color='purple', ha='right', va='bottom', rotation=90)
    plt.text(mean_value - std_value, annotation_positions[4], f'Mean - 1 Std: {mean_value - std_value:.2f}', color='purple', ha='right', va='bottom', rotation=90)
    plt.text(mean_value, y_max*0.4, f'Skewness: {skewness_value:.2f}', color='black', ha='right', va='bottom', rotation=90)
    
    # Annotate most popular bin
    most_popular_bin_index = np.argmax(counts)
    most_popular_bin_value = bin_edges[most_popular_bin_index]
    most_popular_bin_height = counts[most_popular_bin_index]
    plt.axvline(most_popular_bin_value, color='g', linestyle='dashed', linewidth=1)
    plt.text(most_popular_bin_value, most_popular_bin_height * 1.1 if most_popular_bin_height * 1.1 < y_max else y_max*0.95, 
             f'Most Popular: {int(most_popular_bin_value)}', color='g', ha='center', va='bottom')
    
    # Annotate filtering boundaries and method
    if filter_method and lower_bound is not None and upper_bound is not None:
        plt.axvline(lower_bound, color='orange', linestyle='dashed', linewidth=1)
        plt.axvline(upper_bound, color='orange', linestyle='dashed', linewidth=1)
        plt.text(lower_bound, y_max*0.2, f'Lower Bound: {lower_bound:.2f}', color='orange', ha='right', va='bottom', rotation=90)
        plt.text(upper_bound, y_max*0.2, f'Upper Bound: {upper_bound:.2f}', color='orange', ha='right', va='bottom', rotation=90)
        
        # Find empty region to place the filter method text
        x_position, min_count = find_empty_region(counts, bin_edges)
        plt.text(x_position, min_count + y_max*0.1, f'Filter Method: {filter_method}', ha='center', va='center', color='black')

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_bar_chart(data_initial, data_filtered, title_initial, title_filtered, xlabel, ylabel, filename):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    
    classes_initial, counts_initial = zip(*sorted(data_initial.items()))
    classes_filtered, counts_filtered = zip(*sorted(data_filtered.items()))
    
    cmap = cm.get_cmap('tab10', len(classes_initial))
    colors_initial = [cmap(i) for i in range(len(classes_initial))]
    colors_filtered = [cmap(i) for i in range(len(classes_filtered))]
    
    axes[0].bar(classes_initial, counts_initial, color=colors_initial, edgecolor='black')
    axes[0].set_title(title_initial)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_xticks(classes_initial)
    
    axes[1].bar(classes_filtered, counts_filtered, color=colors_filtered, edgecolor='black')
    axes[1].set_title(title_filtered)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_xticks(classes_filtered)
    
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

def generate_statistics(base_plot_path, glue_task_name, task_to_keys, boundaries_dict):
    # Load the GLUE dataset
    dataset = load_dataset('glue', glue_task_name)
    
    subset = 'validation' if glue_task_name != 'mnli' else 'validation_matched'
    
    train_dataset = dataset[subset]

    # Load the tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Get the appropriate keys for the task
    sentence1_key, sentence2_key = task_to_keys[glue_task_name]

    # Calculate the number of tokens for each example
    num_tokens = []
    for example in train_dataset:
        if sentence2_key is None:
            tokens = tokenizer.tokenize(example[sentence1_key])
        else:
            tokens = tokenizer.tokenize(example[sentence1_key] + " " + example[sentence2_key])
        num_tokens.append(len(tokens))
    
    # Calculate skewness
    skewness_value = skew(num_tokens)
    logging.info(f'Skewness for {glue_task_name}: {skewness_value:.2f}')

    # Apply filtering based on skewness
    if abs(skewness_value) > 0.5:  # Considered skewed if the absolute value of skewness is greater than 0.5
        lower_bound = np.percentile(num_tokens, 10)  # Lower 10%
        upper_bound = np.percentile(num_tokens, 90)  # Upper 10%
        filter_method = 'Quantile'
        logging.info(f'For {glue_task_name}, keeping sentences with token length between {lower_bound:.2f} and {upper_bound:.2f}')
    else:
        mean_value = np.mean(num_tokens)
        std_value = np.std(num_tokens)
        lower_bound = mean_value - std_value
        upper_bound = mean_value + std_value
        filter_method = 'Standard Deviation'
        logging.info(f'For {glue_task_name}, keeping sentences with token length between {lower_bound:.2f} and {upper_bound:.2f}')

    # Save boundaries to dictionary
    boundaries_dict[glue_task_name] = {
        'filter_method': filter_method,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    filtered_dataset = [example for example in train_dataset if lower_bound <= len(tokenizer.tokenize(example[sentence1_key] + " " + example[sentence2_key] if sentence2_key else example[sentence1_key])) <= upper_bound]
    logging.info(f'Filtered dataset size: {len(filtered_dataset)}')

    # Plot histogram of the number of tokens
    n_tokens_filename = base_plot_path + f'n_tokens/{glue_task_name}_num_tokens_hist.png'
    plot_histogram(num_tokens, f'Number of Tokens in {glue_task_name.capitalize()} {subset} Dataset', 'Number of Tokens', 'Frequency', n_tokens_filename, bins=30, filter_method=filter_method, lower_bound=lower_bound, upper_bound=upper_bound)

    # Calculate class distribution for initial dataset
    class_counts_initial = {}
    for example in train_dataset:
        label = example['label']
        if label in class_counts_initial:
            class_counts_initial[label] += 1
        else:
            class_counts_initial[label] = 1
    
    # Calculate class distribution for filtered dataset
    class_counts_filtered = {}
    for example in filtered_dataset:
        label = example['label']
        if label in class_counts_filtered:
            class_counts_filtered[label] += 1
        else:
            class_counts_filtered[label] = 1

    # Plot bar chart of the class distributions
    class_distribution_filename = base_plot_path + f'class_distribution/{glue_task_name}_class_distribution_bar.png'
    plot_bar_chart(class_counts_initial, class_counts_filtered, 
                   f'Class Distribution in {glue_task_name.capitalize()} Initial {subset} Dataset', 
                   f'Class Distribution in {glue_task_name.capitalize()} Filtered {subset} Dataset',
                   'Class', 'Number of Samples', class_distribution_filename)

def main():
    seed = 42 # den paizei rolo
    set_seed(seed)  # Ensure reproducibility

    logging.basicConfig(filename='glue_tasks_statistics.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    task_to_keys = {
        "cola": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "wnli": ("sentence1", "sentence2"),
        "mnli": ("premise", "hypothesis"),
    }

    
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

    base_plot_path = f"/home/dbek/src/thesis/thesis-layerconductance-structured-pruning-bert/src/helpers/GLUE_tasks/UNCASED/validation/"
    # base_plot_path = f"/home/dbek/src/thesis/thesis-layerconductance-structured-pruning-bert/src/helpers/GLUE_tasks/CASED/validation/"
    create_directories(base_plot_path)

    glue_tasks = list(task_to_keys.keys())

    boundaries_dict = {}

    for task in glue_tasks:
        logging.info(f'Generating statistics for {task}...')
        generate_statistics(base_plot_path, task, task_to_keys, boundaries_dict)

    # Sort boundaries dictionary by upper bound
    sorted_boundaries_dict = dict(sorted(boundaries_dict.items(), key=lambda item: item[1]['upper_bound']))

    base_path = f"/home/dbek/src/thesis/thesis-layerconductance-structured-pruning-bert/src/helpers/boundaries/UNCASED/validation/"
    # base_path = f"/home/dbek/src/thesis/thesis-layerconductance-structured-pruning-bert/src/helpers/boundaries/CASED/"

    create_directories(base_path)

    # Save sorted boundaries dictionary to a pickle file
    with open(f'{base_path}/boundaries.pkl', 'wb') as f:
        pickle.dump(sorted_boundaries_dict, f)

if __name__ == "__main__":
    main()
