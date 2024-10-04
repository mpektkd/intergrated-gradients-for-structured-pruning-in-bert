import os
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from datasets import load_dataset, Dataset
import numpy as np
from scipy.stats import skew
import json
import logging
import pickle
import random
import matplotlib.cm as cm
import torch

def set_seed(seed: int, rank: int = 0):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
        rank (`int`): For mutliprocessing script
    """
        
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed + rank)

def load_boundaries_dict(file_path='sorted_boundaries.pkl'):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def filter_dataset(glue_task_name, boundaries_dict, task_to_keys):
    # Load the GLUE dataset
    dataset = load_dataset('glue', glue_task_name)
    train_dataset = dataset['train']

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Get the appropriate keys for the task
    sentence1_key, sentence2_key = task_to_keys[glue_task_name]

    # Get the filtering boundaries
    boundaries = boundaries_dict[glue_task_name]
    lower_bound = boundaries['lower_bound']
    upper_bound = boundaries['upper_bound']

    # Filter the dataset based on the boundaries
    filtered_dataset = train_dataset.filter(lambda example: 
                                            lower_bound <= len(tokenizer.tokenize(example[sentence1_key] + " " + example[sentence2_key] if sentence2_key else example[sentence1_key])) <= upper_bound)
    
    return filtered_dataset

def create_balanced_subset(filtered_dataset, num_samples=2000):
    # Calculate class distribution
    label_column = 'label'
    class_counts = filtered_dataset.features[label_column].names
    label_indices = {label: [] for label in class_counts}

    for idx, example in enumerate(filtered_dataset):
        label = example[label_column]
        label_indices[class_counts[label]].append(idx)

    # Determine the minimum number of samples per class to ensure balance
    min_samples_per_class = min(len(indices) for indices in label_indices.values())
    samples_per_class = min(num_samples // len(class_counts), min_samples_per_class)

    # Select samples
    selected_indices = []
    for indices in label_indices.values():
        selected_indices.extend(random.sample(indices, samples_per_class))

    # Shuffle the selected indices to ensure random distribution
    random.shuffle(selected_indices)

    # Create a subset dataset
    subset_dataset = filtered_dataset.select(selected_indices)

    return subset_dataset

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

def extract_statistics(dataset, task_to_keys, glue_task_name, suffix=""):
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Get the appropriate keys for the task
    sentence1_key, sentence2_key = task_to_keys[glue_task_name]

    # Calculate the number of tokens for each example
    num_tokens = []
    for example in dataset:
        if sentence2_key is None:
            tokens = tokenizer.tokenize(example[sentence1_key])
        else:
            tokens = tokenizer.tokenize(example[sentence1_key] + " " + example[sentence2_key])
        num_tokens.append(len(tokens))

    # Plot histogram of the number of tokens
    n_tokens_filename = f'GLUE_tasks/n_tokens/{glue_task_name}_num_tokens_hist{suffix}.png'
    plot_histogram(num_tokens, f'Number of Tokens in {glue_task_name.capitalize()} Train Dataset {suffix}', 'Number of Tokens', 'Frequency', n_tokens_filename)

    # Calculate class distribution
    class_counts = {}
    for example in dataset:
        label = example['label']
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    # Plot bar chart of the class distribution
    class_distribution_filename = f'GLUE_tasks/class_distribution/{glue_task_name}_class_distribution_bar{suffix}.png'
    plot_bar_chart(class_counts, class_counts, 
                   f'Class Distribution in {glue_task_name.capitalize()} Train Dataset {suffix}', 
                   f'Class Distribution in {glue_task_name.capitalize()} Train Dataset {suffix}',
                   'Class', 'Number of Samples', class_distribution_filename)

def main():
    set_seed(42)  # it's the same seed for the main script and i use the same function for set_seed

    logging.basicConfig(filename='filtering_log.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load boundaries dictionary
    boundaries_dict = load_boundaries_dict()

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

    glue_tasks = list(task_to_keys.keys())

    for task in glue_tasks:
        logging.info(f'Creating filtered and balanced dataset for {task}...')

        # Filter the dataset based on the boundaries
        filtered_dataset = filter_dataset(task, boundaries_dict, task_to_keys)
        logging.info(f'Filtered {task} dataset size: {len(filtered_dataset)}')

        # Create a balanced subset
        balanced_subset = create_balanced_subset(filtered_dataset)
        logging.info(f'Balanced subset size: {len(balanced_subset)}')

        # Extract statistics for the filtered dataset
        extract_statistics(filtered_dataset, task_to_keys, task, suffix="Filtered")

        # Extract statistics for the balanced subset
        extract_statistics(balanced_subset, task_to_keys, task, suffix="BalancedSubset")


if __name__ == "__main__":
    
    main()
