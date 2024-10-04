from tqdm import tqdm
import itertools
import subprocess


# Initialize global configuration
D_size = 1000
RANDOM_SEED = 0
BATCH_SIZE = 1
MAX_LEN = 200


# targets = [None, 0, 1]

# # Config for fine-tuned models

# models_datasets = [
    # ('textattack/bert-base-uncased-imdb', 'imdb'),
    # ('textattack/bert-base-uncased-SST-2', 'sst2'),
    # ('textattack/bert-base-uncased-rotten-tomatoes', 'rotten_tomatoes')
# ]
# model_infos = ['fine_tuned']

# MLM = False

# # # Get the cartesian product of the parameters
# combos = list(itertools.product(models_datasets, targets, model_infos))



# subprocess.run([
#         'python',
#         'scores_computation.py',
#         '--model', str('textattack/bert-base-uncased-imdb'), 
#         '--dataset', str('sst2'), 
#         '--target', str(None), 
#         '--tokens', str(MAX_LEN), 
#         '--model_info', str('fine-tuned'),
#         '--D_size', str(D_size),
#         '--seed', str(RANDOM_SEED),
#         '--batch', str(BATCH_SIZE),
#         '--mlm', str(MLM)
#         ])

# for ((model, dataset), target, model_info) in tqdm(combos, desc="Iterate through the variant configurations", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
    
#     subprocess.run([
#         'python',
#         'scores_computation.py',
#         '--model', str(model), 
#         '--dataset', str(dataset), 
#         '--target', str(target), 
#         '--tokens', str(MAX_LEN), 
#         '--model_info', str(model_info),
#         '--D_size', str(D_size),
#         '--seed', str(RANDOM_SEED),
#         '--batch', str(BATCH_SIZE),
#         '--mlm', str(MLM)
#         ])


# Begin stats extraction

# for ((_, dataset), target, model_info) in tqdm(combos, desc="Iterate through the variant configurations for stats extraction", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):

#     subprocess.run([
#         'python',
#         'stats_extraction.py',
#         '--dataset', str(dataset), 
#         '--target', str(target), 
#         '--tokens', str(MAX_LEN), 
#         '--model_info', str(model_info),
#         # '--metric', 'cor',
#         ])


# Config for pre-trained bert-base-uncased

# model = ['bert-base-uncased']
# datasets = ['rotten_tomatoes', 'sst2'] # 200 tokens
# datasets = ['imdb'] # 150 tokens
# model_infos = ['pre_trained']

# Get the cartesian product of the parameters
# combos = list(itertools.product(model, datasets, model_infos))

# MLM = True
# MAX_LEN = 200 
# MAX_LEN = 150 


# for (model, dataset, model_info) in tqdm(combos, desc="Iterate through the variant configurations", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        
#     subprocess.run([
#     'python',
#     'scores_computation.py',
#     '--model', str(model), 
#     '--dataset', str(dataset), 
#     '--target', 'class_reduce', 
#     '--tokens', str(MAX_LEN), 
#     '--model_info', str(model_info),
#     '--D_size', str(D_size),
#     '--seed', str(RANDOM_SEED),
#     '--batch', str(BATCH_SIZE),
#     '--mlm', str(MLM)
#     ])

    # break
# Begin stats extraction

# for (_, dataset, model_info) in tqdm(combos, desc="Iterate through the variant configurations for stats extraction", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):

#     subprocess.run([
#         'python',
#         'stats_extraction.py',
#         '--dataset', str(dataset), 
#         '--target', 'class_reduce', 
#         '--tokens', str(MAX_LEN), 
#         '--model_info', str(model_info),
#         '--mlm', str(MLM),
#         # '--metric', 'cor'
#         ])
