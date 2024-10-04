import seaborn as sns
import torch
from lib import create_nested_directories
from tqdm import tqdm
import matplotlib.pyplot as plt 
import itertools

MAX_LEN = 200
targets = [None, 0, 1]

# Config for fine-tuned models

models_datasets = [
    ('textattack/bert-base-uncased-imdb', 'imdb'),
    ('textattack/bert-base-uncased-SST-2', 'sst2'),
    ('textattack/bert-base-uncased-rotten-tomatoes', 'rotten_tomatoes')
]
model_infos = ['fine_tuned']

# Get the cartesian product of the parameters
combos = list(itertools.product(models_datasets, targets, model_infos))
config = {}

for ((_, dataset), target, model_info) in tqdm(combos, desc="Iterate through the variant configurations for stats extraction", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
    # update config
    config['dataset'] = dataset
    config['target'] = f'{target}Class'
    config['max_len'] = f'{MAX_LEN}tokens'
    config['model_info'] = model_info

    # load the data
    path = create_nested_directories(config)
    visualization = torch.load(path + 'visualization.pt')
    
    fig, ax = plt.subplots(figsize=(15,5))
    xticklabels=list(range(1,13))
    yticklabels=list(range(1,13))
    ax = sns.heatmap(visualization.cpu().detach().numpy(), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    plt.title(config['dataset'] + '-' + config['model_info'] + '-' + config['target'] + '-'+ config['max_len'])
    plt.savefig(path + 'visualization_fig')
    # plt.show()