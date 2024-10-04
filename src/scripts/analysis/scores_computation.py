from transformers import (
  AutoConfig,
  AutoTokenizer,
  AutoModelForSequenceClassification
)

from sklearn.model_selection import StratifiedShuffleSplit

from argparse import ArgumentParser

import pandas as pd

from lib import *

import torch

from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForMaskedLM

from datasets import load_dataset, concatenate_datasets

import time

def main(path, model_path, dataset_name, D_size, RANDOM_SEED, BATCH_SIZE, target, MAX_LEN, mlm=False):

  # TODO: have to pass the device dynamically
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

  # # load model
  # if mlm:
  #   model = AutoModelForMaskedLM.from_pretrained(model_path, output_attentions=True)
  # else:
  #   model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
  
  # # load tokenizer
  # tokenizer = BertTokenizer.from_pretrained(model_path)

  # Load Dataset
  dataset = load_dataset(dataset_name)

  # Labels
  if dataset_name is not None:
      is_regression = dataset_name == "stsb"
      if not is_regression:
          label_list = dataset["train"].features["label"].names
          num_labels = len(label_list)
      else:
          num_labels = 1
  else:
      # Trying to have good defaults here, don't hesitate to tweak to your needs.
      is_regression = dataset["train"].features["label"].dtype in ["float32", "float64"]
      if is_regression:
          num_labels = 1
      else:
          # A useful fast method:
          # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
          label_list = dataset["train"].unique("label")
          label_list.sort()  # Let's sort it for determinism
          num_labels = len(label_list)

  # Load configuration (Config, Model, Tokenizer)
  config = AutoConfig.from_pretrained(
      model_path,
      num_labels=num_labels,
      finetuning_task=dataset_name,
      cache_dir=None,
      revision='main',
      token=None,
      trust_remote_code=False,
      output_attentions=True
  )
  tokenizer = AutoTokenizer.from_pretrained(
      model_path,
      cache_dir=None,
      use_fast=True,
      revision='main',
      token=None,
      trust_remote_code=False,
  )
  model = AutoModelForSequenceClassification.from_pretrained(
      model_path,
      from_tf=bool(".ckpt" in model_path),
      config=config,
      cache_dir=None,
      revision='main',
      token=None,
      trust_remote_code=False,
      ignore_mismatched_sizes=False, 
      attn_implementation="eager"
  )

  # eval mode and deactivate gradients to avoid memory leak
  model.eval()
  model.zero_grad()

  # transfer the model to device
  model.to(device)

  # Take the whole dataset, for being more valid
  datasets_to_concatenate = [ds for name, ds in dataset.items() if name != 'unsupervised']

  # Concatenate the remaining datasets
  dataset = concatenate_datasets(datasets_to_concatenate)

  # Normalize the columns
  dataset = normalize_columns(dataset, 'sentence', 'text')

  # Tokenize and filter dataset
  dataset = tokenize_and_filter(dataset, tokenizer, max_token_length=MAX_LEN)

  # define parameters for stratified splitting
  D_tokens = np.array(dataset["tokens"], dtype='object')
  D_label = np.array(dataset["label"])

  # split the dataset with stratification
  sss = StratifiedShuffleSplit(n_splits=1, test_size=D_size, random_state=RANDOM_SEED)
  _, test_index = list(sss.split(D_tokens, D_label))[0]

  x_test, y_test = D_tokens[test_index], D_label[test_index]

  stratified_data = {
      "tokens": x_test,
      "label": y_test
  }

  # pass the stratified dataset
  data = pd.DataFrame(data=stratified_data)

  del dataset, D_label, D_tokens
  torch.cuda.empty_cache()


  # Define parameters for Collator(tokenizer)
  kwargs = {
      "add_special_tokens": True,
      "return_token_type_ids": False,
      "max_length": MAX_LEN + 2, # 1 for cls, 1 for sep
      "padding": True,
      "return_attention_mask": True,
      "truncation": True, # it is already done in pre-processing
      "return_tensors": "pt",
      "is_split_into_words": True
  }

  # Create data loader
  if mlm:
    _collator = MyMLMCollator(tokenizer=tokenizer, mlm=mlm, mlm_probability=0.15, **kwargs)
    data_loader = create_data_loader(data, BATCH_SIZE, tokenizer, _collator, bucket_sampling=True, shuffle=False)

    total_attentions, total_attributions = get_matrices(
        model,
        device,
        data_loader
    )

    torch.save(total_attentions, path + 'attentions.pt')
    torch.save(total_attributions, path + 'attributions.pt')

    del total_attentions, total_attributions
    torch.cuda.empty_cache()

  else:
    _collator = Collator(tokenizer=tokenizer, **kwargs)
    data_loader = create_data_loader(data, BATCH_SIZE, tokenizer, _collator, bucket_sampling=True, shuffle=False)

    # Attentions Computation

    total_attentions = get_predictions(
        model,
        device,
        data_loader
    )

    torch.save(total_attentions, path + 'attentions.pt')
    del total_attentions
    torch.cuda.empty_cache()

    # Attributions Computation

    total_attributions = get_interpretability_scores(
        model,
        device,
        data_loader,
        target=target
    )

    torch.save(total_attributions, path + 'attributions.pt')
    del total_attributions
    torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--model", help="provide desired model")
    parser.add_argument("--dataset", help="provide desired dataset")
    parser.add_argument("--target", help="provide targeted class")
    parser.add_argument("--tokens", help="provide tokens")
    parser.add_argument("--model_info", help="provide model info")
    parser.add_argument("--D_size", help="provide desired dataset size")
    parser.add_argument("--seed", help="provide random seed")
    parser.add_argument("--batch", help="provide batch size")
    parser.add_argument("--mlm", help="provide mlm flag")

    args = parser.parse_args()
    config = {}

    # update config
    config['dataset'] = args.dataset
    
    if args.mlm == 'True':
      config['target'] = 'class_reduce'
    else:
      config['target'] = 'NoneClass' if args.target == 'None' else f'{args.target}Class'
    
    config['max_len'] = f'{args.tokens}tokens'
    config['model_info'] = args.model_info
    
    # path construction
    path = create_nested_directories(config)
    
    Target = None if args.target == 'None' or (args.mlm == 'True') else int(args.target)

    kwargs = {
        'path': path,
        'model_path': args.model,
        'dataset_name': args.dataset,
        'D_size': int(args.D_size),
        'RANDOM_SEED': int(args.seed),
        'BATCH_SIZE': int(args.batch),
        'target': Target,
        'MAX_LEN':  int(args.tokens),
        'mlm': (args.mlm == 'True')
    }

    # define variable for time measurement
    start_time = time.time()
    
    # call the main() function for scores computation and storing
    main(**kwargs)

    print(f"{path} --- {time.time() - start_time} seconds ---")


# giemoupoupasmana8apawstakaravia
# screen == scores_computation / 1453416