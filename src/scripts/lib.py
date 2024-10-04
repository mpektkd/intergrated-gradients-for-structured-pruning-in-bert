import os
import scipy
import numpy as np

from typing import Optional

from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy

from tqdm import tqdm

import random
from torch.utils.data import Sampler

import torch
import random

from captum.attr import LayerConductance
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

from datasets import Dataset
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

import datetime
import logging

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def get_first_available_cuda_device():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            try:
                # Try allocating a small tensor on the device to test its availability
                torch.tensor([0], device=device)
                return device
            except RuntimeError as e:
                print(f"Device {i} not available: {e}")
    return torch.device('cpu')

def count_heads(model):
    remaining_heads = {}
    for i, layer in enumerate(model.bert.encoder.layer):
        remaining_heads[i] = layer.attention.self.num_attention_heads
    return remaining_heads

def predict(model, inputs, token_type_ids=None, position_ids=None, attention_mask=None, head_mask=None):
    
    output = model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, head_mask=head_mask)

    return output.logits, output.attentions

def squad_pos_forward_func(inputs, model, attention_mask=None, target=None, head_mask=None):
    '''
      Notes:
        1. The need of target is necessary when the output is multi-dimensional.
    '''
    output = model(inputs_embeds=inputs, attention_mask=attention_mask, head_mask=head_mask)

    return output.logits.max(1).values if target==None else output.logits

def normalize_columns(dataset: Dataset, old_name: str, new_name: str):
    """
    Renames a column in the dataset and drop all but 'label' and 'text'.

    Args:
    dataset (Dataset): The HuggingFace dataset to process.
    old_name (str): The current name of the column to be renamed.
    new_name (str): The new name for the column.

    Returns:
    Dataset: The dataset with the renamed column.
    """

    # Check if the old column name exists in the dataset
    if old_name in dataset.column_names:
        # Rename the column
        dataset = dataset.rename_column(old_name, new_name)
    else:
        print(f"The column '{old_name}' does not exist in the dataset.")

    # List all columns except 'label'
    to_be_removed = [col for col in dataset.column_names if col != 'label' and col != 'text']

    dataset = dataset.remove_columns(to_be_removed)

    return dataset

def tokenize_and_filter(dataset: Dataset, tokenizer: PreTrainedTokenizer, max_token_length: int = 200, batch_size: int = 1000):
    """
    Tokenizes the sentences in the dataset and filters out examples with more than max_token_length tokens.

    Args:
    dataset (Dataset): The HuggingFace dataset to process.
    tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the sentences.
    max_token_length (int): Maximum allowed token length for each example.

    Returns:
    Dataset: A new dataset with tokenized sentences and filtered based on token length.
    """

    def batch_tokenize(examples):

      tokens = [tokenizer.tokenize(text) for text in examples['text']]

      return {'tokens': tokens, 'label': examples['label']}

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        batch_tokenize,
        batched=True,
        batch_size=batch_size,
        remove_columns=['text']
    )

    # Filter the dataset to keep examples with <= max_token_length tokens
    filtered_dataset = tokenized_dataset.filter(
        lambda example: len(example['tokens']) <= max_token_length
    )

    return filtered_dataset


class MovieReviewDataset(torch.utils.data.Dataset):
  def __init__(self, reviews, targets, tokenizer):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.reviews)

  # The sentence is alreadt tokenized
  def __getitem__(self, item):
    tokens = self.reviews[item]
    target = self.targets[item]

    return tokens, target

def create_data_loader(df, batch_size, tokenizer, collator, bucket_sampling=False, shuffle=False):

  ds = MovieReviewDataset(
    reviews=df.tokens.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer
  )

  # Implement Bucket Sampling
  if bucket_sampling:
    return torch.utils.data.DataLoader(
        ds,
        batch_sampler=BatchSamplerSimilarLength(dataset=ds, batch_size=batch_size, shuffle=shuffle),
        collate_fn=collator
    )

  return torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size,
    collate_fn=collator
  )

def igspp_get_predictions(model, device, data_loader, _distributed=False, head_mask=None, inference=False):
  '''
  Notes for MLM model:
    1. The MLM model finally produces a tensor (batch, tokens, vocab_size), that for every token
    produces a probability for each token in the vocab. Based on each fine tuning task, we add the
    appropriate head in the end and make the choise.
  '''
  if _distributed:
     model = model.module
     
  total_attentions = []
  preds = []
  labels = []

  desc = "Inference Samples" if inference else "Extracting Attention Weights"

  with torch.no_grad():
    for d in tqdm(data_loader, total=len(data_loader), desc=desc, ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):

      # we do not use 'label' as we do no care about the loss
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      token_type_ids = d["token_type_ids"].to(device)
      label = d["label"]
      
      pred, attentions = predict(
        model,
        inputs=input_ids,
        attention_mask=attention_mask, 
        head_mask=None,
        token_type_ids=token_type_ids
      )
      
      pred = pred.cpu()

      preds.append(pred)
      labels.append(label)
      
      if inference:
         continue
      
      # stack all the layers -> layer x batch x head x tokens x tokens
      # attentions_stack = torch.stack(tuple(i.detach().clone().cpu() for i in attentions)) # transfer data from cuda to cpu, therefore attentions_stack is on CPU
      attentions_stack = [i.detach().clone().cpu() for i in attentions] # transfer data from cuda to cpu, therefore attentions_stack is on CPU
            
      del attentions
      torch.cuda.empty_cache()
      # stack all the attentions of the spescific batch -> layer x batch x head x tokens x tokens
      total_attentions.append(attentions_stack)
      # total_attentions = torch.cat((total_attentions, attentions_stack), dim=1)

      del attentions_stack
      torch.cuda.empty_cache()
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

  return total_attentions, preds, labels

def igspp_get_interpretability_scores(model, device, data_loader, _distributed, target=None, head_mask=None):
  '''
    Notes:
      1. In the source code for LayerConductance.attribute() all the helper-functions
      that are called, activate the grad computation (requires_grad=True) before start
      the calculation.
      https://github.com/pytorch/captum/blob/ed3b1fa4b3d8afc0eff4179b1d1ef4b191f13cc1/captum/_utils/gradient.py#L589
  '''
  head_mask = None # it is igspp algorithm
  if _distributed:
    model = model.module

  interpretable_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')
  
  try:
    total_attributions = []
    for d in tqdm(data_loader, total=len(data_loader), desc=f"Process with rank-{device} calculating Attribution Scores", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
      input_ids = d["input_ids"].to(device)
      ref_input_ids = d["ref_input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)

      input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
      ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)

      layer_attn_mat = []
      for i in range(model.config.num_hidden_layers):
        lc = LayerConductance(squad_pos_forward_func, model.bert.encoder.layer[i])
        layer_attributions = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, target=target, additional_forward_args=(model, attention_mask, target, head_mask))
        layer_attn_mat.append(layer_attributions[1].detach().clone().cpu()) # convert into cpu memory and the delete from cuda to avoid memory leak

        # Delete intermediate tensors to free up memory
        del layer_attributions, lc
        torch.cuda.empty_cache()

      # stack all the layers -> layer x batch x head x tokens x tokens
      # layer_attn_mat = torch.stack(layer_attn_mat)

      del input_ids, ref_input_ids, attention_mask, input_embeddings, ref_input_embeddings
      torch.cuda.empty_cache()

      # # stack all the attentions of the specific batch -> layer x batch x head x tokens x tokens
      total_attributions.append(layer_attn_mat)

      del layer_attn_mat
      torch.cuda.empty_cache()

    # total_attributions = torch.cat(attributions_list, dim=1)

  finally:
    # after we finish the interpretation we need to remove
    # interpretable embedding layer with the following command:
    remove_interpretable_embedding_layer(model, interpretable_embedding)

  return total_attributions

def get_interpretability_scores2(model, device, data_loader, _target=None, mlm=False):
  '''
    Notes:
      1. In the source code for LayerConductance.attribute() all the helper-functions
      that are called, activate the grad computation (requires_grad=True) before start
      the calculation.
      https://github.com/pytorch/captum/blob/ed3b1fa4b3d8afc0eff4179b1d1ef4b191f13cc1/captum/_utils/gradient.py#L589
  '''
  interpretable_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')
  try:
    total_attributions = []
    for d in tqdm(data_loader, total=len(data_loader), desc="Calculating Attribution Scores", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):

      _input_ids = d["input_ids"]
      _ref_input_ids = d["ref_input_ids"]
      _attention_mask = d["attention_mask"]

      # BertForLanguafeModeling
      _target = d["target"] if mlm else _target

      attribution_list_of_copies = []
      for (input_ids, ref_input_ids, attention_mask, target) in zip(_input_ids, _ref_input_ids, _attention_mask, _target):
        input_ids = input_ids.unsqueeze(0).to(device)
        ref_input_ids = ref_input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)

        input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
        ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)

        layer_attn_mat = []
        for i in range(model.config.num_hidden_layers):
          lc = LayerConductance(squad_pos_forward_func, model.bert.encoder.layer[i])
          layer_attributions = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, target=target, additional_forward_args=(model, attention_mask, target))
          layer_attn_mat.append(layer_attributions[1].detach().clone().cpu()) # convert into cpu memory and the delete from cuda to avoid memory leak

          # Delete intermediate tensors to free up memory
          del layer_attributions, lc
          torch.cuda.empty_cache()

        # (12,1,12,tokens,tokens)
        layer_attn_mat = torch.stack(layer_attn_mat)

        # attribution_list_of_copies is a list of attribution for each copy of the sent
        attribution_list_of_copies.append(layer_attn_mat)

        del layer_attn_mat, input_ids, ref_input_ids, attention_mask, input_embeddings, ref_input_embeddings
        torch.cuda.empty_cache()

      # stack the list to take the mean value
      attribution_list_of_copies = torch.cat(attribution_list_of_copies, dim=1)
      total_attributions.append(torch.mean(attribution_list_of_copies, dim=1))

      del attribution_list_of_copies
      torch.cuda.empty_cache()

    # total_attributions = torch.cat(attributions_list, dim=1)

  finally:
    # after we finish the interpretation we need to remove
    # interpretable embedding layer with the following command:
    remove_interpretable_embedding_layer(model, interpretable_embedding)


  return total_attributions

def get_predictions2(model, device, data_loader):
  '''
  Notes for MLM model:
    1. The MLM model finally produces a tensor (batch, tokens, vocab_size), that for every token
    produces a probability for each token in the vocab. Based on each fine tuning task, we add the
    appropriate head in the end and make the choise.
  '''
  total_attentions = []

  with torch.no_grad():
    for d in tqdm(data_loader, total=len(data_loader), desc="Extracting Attention Weights", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):

      # we do not use 'label' as we do no care about the loss
      _input_ids = d["input_ids"]
      _attention_mask = d["attention_mask"]
      attention_list_of_copies = []
      for (input_ids, attention_mask) in zip(_input_ids, _attention_mask):

        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        attentions = predict(
          model,
          inputs=input_ids,
          attention_mask=attention_mask
        )

        # stack all the layers -> layer x batch x head x tokens x tokens
        attentions_stack = torch.stack(tuple(i.detach().clone().cpu() for i in attentions)) # transfer data from cuda to cpu, therefore attentions_stack is on CPU
        attention_list_of_copies.append(attentions_stack)

        del attentions, attentions_stack
        torch.cuda.empty_cache()

      # stack the attention for all the copies and take the mean value
      attention_list_of_copies = torch.cat(attention_list_of_copies, dim=1)
      total_attentions.append(torch.mean(attention_list_of_copies, dim=1))
      # total_attentions = torch.cat((total_attentions, attentions_stack), dim=1)

      del attention_list_of_copies
      torch.cuda.empty_cache()

  return total_attentions
def get_matrices(model, device, data_loader):
  '''
    Notes:
      1. In the source code for LayerConductance.attribute() all the helper-functions
      that are called, activate the grad computation (requires_grad=True) before start
      the calculation.
      https://github.com/pytorch/captum/blob/ed3b1fa4b3d8afc0eff4179b1d1ef4b191f13cc1/captum/_utils/gradient.py#L589
  '''
  '''
    George's Feedback for OOM as for the simultaneous computation att, attr:
      1.  Remove .clone()
      2.  model.eval() (att) / model.train() (attr)
      3.  Comment out "del cuda variables" and empty_cache()
            ---------------------------------------
              if the above would not work
      4.  Choose a specific seed for experiment reproducibility. I should use all the seeds in the pytorch doc, 
          because I use all the libraries. (this would be useful if i would use the implementation 
      5.  For the implementation inside the collator, I could save the dataset, while iterating the batches, 
          in a pickle file and load it when it is necessary.

  '''
  total_attentions, total_attributions = [], []
  for d in tqdm(data_loader, total=len(data_loader), desc="Calculating Both Matrices. It may take a while :P...", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):

    _input_ids = d["input_ids"]
    _ref_input_ids = d["ref_input_ids"]
    _attention_mask = d["attention_mask"]
    _target = d["target"]

    attention_list_of_copies, attribution_list_of_copies = [], []
    for (input_ids, ref_input_ids, attention_mask, target) in zip(_input_ids, _ref_input_ids, _attention_mask, _target):
      input_ids = input_ids.unsqueeze(0).to(device)
      ref_input_ids = ref_input_ids.unsqueeze(0).to(device)
      attention_mask = attention_mask.unsqueeze(0).to(device)
      # Attention Calculation
      #############################################


              ###### ** Dict to be saved HERE comment (5) ######



      # Disable gradients computation
      model.eval()
      with torch.no_grad():
        attentions = predict(
          model,
          inputs=input_ids,
          attention_mask=attention_mask
        )

      # stack all the layers -> layer x batch x head x tokens x tokens
      # .clone() might be removed 
      attentions_stack = torch.stack(tuple(i.detach().cpu() for i in attentions)) # transfer data from cuda to cpu, therefore attentions_stack is on CPU
      attention_list_of_copies.append(attentions_stack)

      del attentions, attentions_stack
      torch.cuda.empty_cache()
      
      #############################################

      # Attribution Calculation

      #############################################
      try:
        interpretable_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')

        input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
        ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)

        layer_attn_mat = []

        # Enable gradients computation
        model.train()

        for i in range(model.config.num_hidden_layers):
          lc = LayerConductance(squad_pos_forward_func, model.bert.encoder.layer[i])
          layer_attributions = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, target=target, additional_forward_args=(model, attention_mask, target))
          # to removed .clone
          layer_attn_mat.append(layer_attributions[1].detach().cpu()) # convert into cpu memory and the delete from cuda to avoid memory leak

          # Delete intermediate tensors to free up memory
          del layer_attributions, lc
          torch.cuda.empty_cache()

        # (12,1,12,tokens,tokens)
        layer_attn_mat = torch.stack(layer_attn_mat)

        # attribution_list_of_copies is a list of attribution for each copy of the sent
        attribution_list_of_copies.append(layer_attn_mat)

        del layer_attn_mat, input_embeddings, ref_input_embeddings
        torch.cuda.empty_cache()
      finally:
        # after we finish the interpretation we need to remove
        # interpretable embedding layer with the following command:
        remove_interpretable_embedding_layer(model, interpretable_embedding)

      # continue
      #############################################

    # stack the attention for all the copies and take the mean value
    # We use unsqueeze here, because we take mean value and the batch dimension is lost.
    # However, we need this dimensio because the mean() is an aggregation fucn of the original
    # sample, so batch = 1.
    attention_list_of_copies = torch.cat(attention_list_of_copies, dim=1)
    attention_mean_of_copies = torch.mean(attention_list_of_copies, dim=1).unsqueeze(1)
    total_attentions.append(attention_mean_of_copies)

    del attention_list_of_copies
    torch.cuda.empty_cache()

    # stack the list to take the mean value
    attribution_list_of_copies = torch.cat(attribution_list_of_copies, dim=1)
    attribution_mean_of_copies = torch.mean(attribution_list_of_copies, dim=1).unsqueeze(1)
    total_attributions.append(attribution_mean_of_copies)

    del attribution_list_of_copies
    torch.cuda.empty_cache()

  return total_attentions, total_attributions

def create_nested_directories(config):
    # Start with the base directory
    current_path = '../scores/' + config['dataset'] + '/'

    # Check if the base directory exists, if not, create it
    if not os.path.exists(current_path):
        os.makedirs(current_path)

    # Iterate over the remaining configuration parameters
    for key in ['target', 'max_len', 'model_info']:
        # Append the next level directory to the current path
        current_path = os.path.join(current_path, config[key])

        # Check if this subdirectory exists, if not, create it
        if not os.path.exists(current_path):
            os.makedirs(current_path)

    current_path += '/'

    return current_path

def calculate_percentages(numbers, condition, params):
    numbers = np.array(numbers)

    percentages = {}
    indices = {}
    for p in params:
        count = np.sum(condition(numbers, p))
        ind = np.where(condition(numbers, p))

        percentages[p] = (count / len(numbers)) * 100 if len(numbers) != 0 else 0
        indices[p] = ind[0]

    return percentages, indices


def f(range):
  bottom = range[0]
  if (bottom == 0.2):
    return 1
  if (bottom == 0.4):
    return 2
  if (bottom == 0.6):
    return 3
  return 4

class Collator(object):
  def __init__(self, tokenizer, **kwargs):
    self.tokenizer = tokenizer
    self.kwargs = kwargs

  def __call__(self, batch):

    tokens, label = [], []
    for _tokens, _label in batch:
      tokens.append(_tokens)
      label.append(_label)

    encoding = self.tokenizer(text=tokens, **self.kwargs)

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # construct reference token ids
    ref_input_ids = torch.zeros_like(input_ids) # pad_token_id == 0

    ref_input_ids[:, 0] = self.tokenizer.cls_token_id
    ref_input_ids[:, -1] = self.tokenizer.sep_token_id

    return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ref_input_ids': ref_input_ids,
            'label': torch.tensor(label)
          }
  
class PruningCollator(object):
  def __init__(self, t_config):
    
    self.t_config = t_config
    self.sentence1_key, self.sentence2_key = task_to_keys[t_config["task"]]


  def __call__(self, batch):

    args = []
    labels = []

    for sample in batch:
      # For datasets 1 cl --> args = ['sent1', 'sent2', 'sent3', 'sent4']
      # For datasets 2 cl --> args = [('sent1a', 'sent1b), ('sent2a', 'sent2b), ('sent3a', 'sent3b), ('sent4a', 'sent4b)]
      # Message: TextinputEncoding must be TextInputSequence (str) or Tuple[InputSequence, InputSequence] InputSequence: Union[str, List[str](pre-tokenized)]
      arg = sample[self.sentence1_key] if self.sentence2_key is None else (sample[self.sentence1_key], sample[self.sentence2_key]) 

      args.append(arg)
      labels.append(sample["label"])

    encoding = self.t_config["tokenizer"](args, **self.t_config["e_config"])

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    token_type_ids = encoding['token_type_ids']

    # construct reference token ids
    ref_input_ids = torch.zeros_like(input_ids) # pad_token_id == 0

    ref_input_ids[:, 0] = self.t_config["cls_id"]
    ref_input_ids[:, -1] = self.t_config["sep_id"]

    batched_data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ref_input_ids': ref_input_ids,
            'label': torch.tensor(labels),
            'token_type_ids': token_type_ids
          }

    return batched_data
  
class MyMLMCollator(DataCollatorForLanguageModeling, Collator):
  '''
  Notes about DataCollatorForLanguageModeling:

    1. It uses tokenizer.pad() that pads the sequence. This function accepts various
    inputs(https://github.com/huggingface/transformers/blob/250032e974359ba8df38c7e5530020050240d305/src/transformers/tokenization_utils_base.py#L3130)
    we pass List[BatchEncoding].

    2. The pad() method checks if padding is necessary, so we could have padded the
    sequence before. It pads anyway, but for another reason. it uses pad_to_multiple_of for
    hardware efficiency, as the inputs in the batch have multiple tokens of the number that
    hardware supports.

    3. It returns batches with 'input_ids' and 'labels' inside the data flow. After the
    collator the batch with these 2 lists are passed into the model. The 'labels' concern
    the MLM loss (-100 for unmasked tokens) NOT classification task.

    4. The 'input_ids' processed and substituted by MASK token based on the algo(80/10/10).

    5. We define a parent Class. __call__() returns
    {
      input_ids,
      label (target)/ ref_input_ids,
      labels(-100 for ignoring non masked tokens),
      attentions_mask
    }.

    6. For MLM task, it is not necessary to index the output manually, as it is done inside the conductance
    https://github.com/pytorch/captum/blob/2efc105b9638383911191581f2617276a1512734/captum/_utils/common.py#L515

  '''
  def __init__(self, tokenizer, mlm=True, mlm_probability=0.15, **kwargs):

    # Initialize the base DataCollatorForLanguageModeling class
    DataCollatorForLanguageModeling.__init__(self, tokenizer, mlm, mlm_probability)
    Collator.__init__(self, tokenizer, **kwargs)

  def __call__(self, batch):
    '''
      1. This collator changes dynamically the batch size. If we notice the data flow
          the only use of the batch size is when creating the data loader. In our case, we use
          backet sampling, so we implement with custom way the batches and yield each one iteratining
          through them.

          In source code, the dataloader has a function that yields the batches, but in our case it is
          custom. So, after yielding each batch, we can process the data as we want. (data augmentation)
    '''

    encoding_of_batches = Collator.__call__(self, batch)

    batch_of_encodings = [
        {key: val if isinstance(val, list) else val for key, val in zip(encoding_of_batches.keys(), item)}
        for item in zip(*encoding_of_batches.values())
    ]

    '''
      **  Sometimes, due to the probabilistic algo, there is no masked token.
          There is an issue with the target variable then. So, we use
          the while condition until masking will be applied.

          Another solution is to mask one token arbitrarily.
    '''

    count = 0
    while(True):
      count += 1
      
      if count < 10:
        final_batch = DataCollatorForLanguageModeling.__call__(self, batch_of_encodings)
      else:
        final_batch['input_ids'][0][2] = 103
        final_batch['labels'][0][2] = 103 # arbitrarily index = 2

      labels = final_batch['labels'][0]

      # Find masked tokens (80/10/10)
      non_100_indices = torch.where(labels != -100)[0]

      num_repeats = len(non_100_indices)

      if num_repeats > 0:
        break
      

    ###### The following code works ONLY  for BATCH_SIZE = 1 !!! ######
    augmented_inputs = []
    target = []

    input_ids_og = encoding_of_batches['input_ids'][0]
    input_ids_masked = final_batch['input_ids'][0]
    
    # We make as much copies of the initial sentenece as the number of the masked tokens
    for indx in non_100_indices:
      new_input_ids = input_ids_og.detach().clone() # Here the tensors are NOT in the device

      new_input_ids[indx] = input_ids_masked[indx]
      pair = (indx.item(), input_ids_og[indx].item())

      augmented_inputs.append(new_input_ids)
      target.append(pair)

    final_batch['input_ids'] = torch.stack(augmented_inputs)
    final_batch['target'] = target
    final_batch['attention_mask'] = final_batch['attention_mask'].repeat(num_repeats, 1)
    final_batch['ref_input_ids'] = final_batch['ref_input_ids'].repeat(num_repeats, 1)

    # final_batch['label] has not the right shape :P. it was not necessary!!

    return final_batch


class BatchSamplerSimilarLength(Sampler):
  '''
    DATA FLOW:
      Dataset -> Sampler -> Collator -> DataLoader -> Model -> Training Loop
  '''
  def __init__(self, dataset, batch_size, indices=None, shuffle=False):
    self.batch_size = batch_size
    self.shuffle = shuffle
    # get the indices and length

    self.indices = [(i, len(s[0])) for i, s in enumerate(dataset)]
    # if indices are passed, then use only the ones passed (for ddp)
    if indices is not None:
       self.indices = torch.tensor(self.indices)[indices].tolist()

  def __iter__(self):
    if self.shuffle:
       random.shuffle(self.indices)

    pooled_indices = []
    # create pool of indices with similar lengths
    for i in range(0, len(self.indices), self.batch_size * 100):
      pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))
    self.pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    batches = [self.pooled_indices[i:i + self.batch_size] for i in
               range(0, len(self.pooled_indices), self.batch_size)]

    if self.shuffle:
        random.shuffle(batches)
    for batch in batches:
        yield batch

  def __len__(self):
    return len(self.indices) // self.batch_size

class GLUEBatchSamplerSimilarLength(Sampler):
  '''
    DATA FLOW:
      Dataset -> Sampler -> Collator -> DataLoader -> Model -> Training Loop
  '''
  def __init__(self, dataset, batch_size, indices=None, shuffle=False, task=None):
    self.batch_size = batch_size
    self.shuffle = shuffle
    # get the indices and length

    # i have already compute the total length of the sentences for bucketing
    self.indices = [(i, s["total_length"]) for i, s in enumerate(dataset)]

    # if indices are passed, then use only the ones passed (for ddp)
    if indices is not None:
       self.indices = torch.tensor(self.indices)[indices].tolist()

  def __iter__(self):
    if self.shuffle:
       random.shuffle(self.indices)

    pooled_indices = []
    # create pool of indices with similar lengths
    for i in range(0, len(self.indices), self.batch_size * 100):
      pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))
    self.pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    batches = [self.pooled_indices[i:i + self.batch_size] for i in
               range(0, len(self.pooled_indices), self.batch_size)]

    if self.shuffle:
        random.shuffle(batches)
    for batch in batches:
        yield batch

  def __len__(self):
    return len(self.indices) // self.batch_size


import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class DistributedEvalSampler(Sampler):
    r"""
    source code from : https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch

#https://discuss.pytorch.org/t/using-distributedsampler-in-combination-with-batch-sampler-to-make-sure-batches-have-sentences-of-similar-length/119824/3
class DistributedBatchSamplerSimilarLength(DistributedEvalSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, batch_size = 10, task:str = None) -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.batch_size = batch_size
        self.task = task

    def __iter__(self):
        indices = list(super().__iter__())
        batch_sampler = GLUEBatchSamplerSimilarLength(self.dataset, batch_size=self.batch_size, indices=indices, task=self.task)
        return iter(batch_sampler)

    def __len__(self) -> int:
        return self.num_samples//self.batch_size
  
def get_dimensions(lst):
    """
    Recursively find the dimensions of a nested list.
    
    Parameters:
    lst (list): The nested list for which dimensions are to be found.
    
    Returns:
    list: A list representing the dimensions of the input list.
    """
    if isinstance(lst, list):
        if len(lst) == 0:
            return [0]
        first_elem = lst[0]
        return [len(lst)] + get_dimensions(first_elem)
    else:
        return []

def get_interpretability_scores(model, device, data_loader, _distributed, target=None, head_mask=None):
  '''
    Notes:
      1. In the source code for LayerConductance.attribute() all the helper-functions
      that are called, activate the grad computation (requires_grad=True) before start
      the calculation.
      https://github.com/pytorch/captum/blob/ed3b1fa4b3d8afc0eff4179b1d1ef4b191f13cc1/captum/_utils/gradient.py#L589
  '''
  if _distributed:
    model = model.module

  interpretable_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')
  
  try:
    total_attributions = []
    for d in tqdm(data_loader, total=len(data_loader), desc=f"Process with rank-{device} calculating Attribution Scores", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
      input_ids = d["input_ids"].to(device)
      ref_input_ids = d["ref_input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      target = d["label"].to(device) if target == "golden" else target

      input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
      ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)

      layer_attn_mat = []
      for i in range(model.config.num_hidden_layers):
        lc = LayerConductance(squad_pos_forward_func, model.bert.encoder.layer[i])
        layer_attributions = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, target=target, additional_forward_args=(model, attention_mask, target, head_mask))
        layer_attn_mat.append(layer_attributions[1].detach().clone().cpu()) # convert into cpu memory and the delete from cuda to avoid memory leak

        # Delete intermediate tensors to free up memory
        del layer_attributions, lc
        torch.cuda.empty_cache()

      # stack all the layers -> layer x batch x head x tokens x tokens
      layer_attn_mat = torch.stack(layer_attn_mat)

      del input_ids, ref_input_ids, attention_mask, input_embeddings, ref_input_embeddings
      torch.cuda.empty_cache()

      # # stack all the attentions of the specific batch -> layer x batch x head x tokens x tokens
      total_attributions.append(layer_attn_mat.detach().clone().cpu())

      del layer_attn_mat
      torch.cuda.empty_cache()

    # total_attributions = torch.cat(attributions_list, dim=1)

  finally:
    # after we finish the interpretation we need to remove
    # interpretable embedding layer with the following command:
    remove_interpretable_embedding_layer(model, interpretable_embedding)

  return total_attributions

def get_predictions(model, device, data_loader, _distributed=False, head_mask=None, inference=False):
  '''
  Notes for MLM model:
    1. The MLM model finally produces a tensor (batch, tokens, vocab_size), that for every token
    produces a probability for each token in the vocab. Based on each fine tuning task, we add the
    appropriate head in the end and make the choise.
  '''
  if _distributed:
     model = model.module
     
  total_attentions = []
  preds = []
  labels = []

  desc = "Inference Samples" if inference else "Extracting Attention Weights"

  with torch.no_grad():
    for d in tqdm(data_loader, total=len(data_loader), desc=desc, ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):

      # we do not use 'label' as we do no care about the loss
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      token_type_ids = d["token_type_ids"].to(device)
      label = d["label"]
      
      pred, attentions = predict(
        model,
        inputs=input_ids,
        attention_mask=attention_mask, 
        head_mask=head_mask,
        token_type_ids=token_type_ids
      )
      
      pred = pred.cpu()

      preds.append(pred)
      labels.append(label)
      
      if inference:
         continue
      
      # stack all the layers -> layer x batch x head x tokens x tokens
      attentions_stack = torch.stack(tuple(i.detach().clone().cpu() for i in attentions)) # transfer data from cuda to cpu, therefore attentions_stack is on CPU
            
      del attentions
      torch.cuda.empty_cache()

      # stack all the attentions of the spescific batch -> layer x batch x head x tokens x tokens
      total_attentions.append(attentions_stack.detach().clone().cpu())
      # total_attentions = torch.cat((total_attentions, attentions_stack), dim=1)

      del attentions_stack
      torch.cuda.empty_cache()
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

  return total_attentions, preds, labels

def igspp_flatten_tensors(tensor_list):
    """
    Flatten the last two dimensions of each tensor in the list.
    Input tensors should have shape (12, batch, 12, tokens, tokens).
    The output tensors will have shape (12, batch, 12, flatten_tokens).
    """
    return [[layer.view(layer.size(0), layer.size(1), layer.size(2) * layer.size(3)) for layer in batch] for batch in tensor_list]


def flatten_tensors(tensor_list):
    """
    Flatten the last two dimensions of each tensor in the list.
    Input tensors should have shape (12, batch, 12, tokens, tokens).
    The output tensors will have shape (12, batch, 12, flatten_tokens).
    """
    return [tensor.view(12, -1, 12, tensor.size(3) * tensor.size(4)) for tensor in tensor_list]

def cor(attentions_list, attributions_list):
    """
    Computes the Spearman correlations for each batch in each tensor in the lists.
    The input lists should contain tensors of shape (12, batch, 12, flatten_tokens).
    The output is a tensor of shape (total_batches, 12, 12) where total_batches is the sum of all batches.
    """
    # Flatten the last two dimensions of each tensor
    attentions_list = flatten_tensors(attentions_list)
    attributions_list = flatten_tensors(attributions_list)
    
    # List to hold the correlation results for each batch
    all_correlation_results = []

    # Iterate over each tensor in the lists
    K = len(attentions_list)

    for i in tqdm(range(K), desc="Corr Calculation", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        
        attentions, attributions = attentions_list[i], attributions_list[i]

        # Ensure the tensors have the correct shape
        if attentions.shape[0] != 12 or attributions.shape[0] != 12:
            raise ValueError("Input tensors must have the first dimension of size 12")

        # Number of batches in the current tensor
        num_batches = attentions.shape[1]

        # Compute correlations for each batch
        for batch_idx in range(num_batches):
            # Initialize a tensor to hold the correlation results for this batch
            correlation_tensor = torch.zeros((12, 12))

            # Extract the batch-specific tensors
            attention_batch = attentions[:, batch_idx, :, :]
            attribution_batch = attributions[:, batch_idx, :, :]

            # Iterate over each pair of layers and heads
            for layer_idx in range(12):
                for head_idx in range(12):
                    # Extract the corresponding vectors for the current pair
                    attention_vector = attention_batch[layer_idx, head_idx, :].numpy()
                    attribution_vector = attribution_batch[layer_idx, head_idx, :].numpy()

                    # Compute the Spearman correlation between these vectors
                    correlation, _ = scipy.stats.spearmanr(attention_vector, attribution_vector)

                    # Store the correlation in the result tensor
                    correlation_tensor[layer_idx, head_idx] = correlation

            # Add the correlation results of this batch to the list
            all_correlation_results.append(correlation_tensor)

    # Stack all correlation results and compute the mean across the batch dimension
    stacked_correlations = torch.stack(all_correlation_results)
    mean_correlation_tensor = stacked_correlations.mean(dim=0).clone().detach()
  
    return mean_correlation_tensor

def igspp_cor(attentions_list, attributions_list, head_mask=None):
    """
    Computes the Spearman correlations for each batch in each tensor in the lists.
    The input lists should contain tensors of shape (12, batch, 12, flatten_tokens).
    The output is a tensor of shape (total_batches, 12, 12) where total_batches is the sum of all batches.
    """
    # Flatten the last two dimensions of each tensor
    
    attentions_list = igspp_flatten_tensors(attentions_list)
    attributions_list = igspp_flatten_tensors(attributions_list)
    # List to hold the correlation results for each layer and head
    
    # head_mask = np.load("/home/dbekris/src/pruning/IGSPF/corr/target_None/rte/samples_2000/head_mask_4.npy")
    # head_mask = torch.from_numpy(head_mask)
    if head_mask is None:
       head_mask = torch.ones((12,12))
       
    importance = torch.ones_like(head_mask)
    importance[head_mask==0.0] = float("Inf")
    importance_list = [[[] for i in range(importance.shape[1])] for j in range(importance.shape[0])]

    # Iterate over each layer (outer list)
    for _, (attentions_batch, attributions_batch) in enumerate(zip(attentions_list, attributions_list)):
        # print(importance_list)
        # Iterate over each tensor (inner list)
        for layer_idx, (attentions, attributions) in enumerate(zip(attentions_batch, attributions_batch)):

            # Ensure the tensors have the correct shape
            if attentions.shape[1] != attributions.shape[1]:
                raise ValueError("Number of heads must be the same in attention and attribution tensors for each layer")

            # Number of batches in the current tensor
            batch_size = attentions.shape[0]

            # List to hold the correlation results for this head across batches
            # print(attentions.shape)
            for h in range(attentions.shape[1]):
              attention_head = attentions[:, h, :]
              attribution_head = attributions[:, h, :]

              # Initialize a list to hold correlation results for this batch
              # Compute correlations for each batch
              for sample_idx in range(batch_size):
                  # Extract the batch-specific tensors
                  attention_vector = attention_head[sample_idx, :].numpy()
                  attribution_vector = attribution_head[sample_idx, :].numpy()

                  # Compute the Spearman correlation between these vectors
                  correlation, _ = scipy.stats.spearmanr(attention_vector, attribution_vector)

                  # Store the correlation in the result list
                  rest_heads = head_mask[layer_idx] != 0.0
                  indices = torch.nonzero(rest_heads)
                  # print(head_mask[layer_idx])
                  # print(indices)
                  indices = torch.flatten(indices)
                  # print(indices)
                  # print('gamw th mana soy', indices)
                  # print(indices, attentions.shape[1])
                  # print('gamw th mana ', indices[h])
                  # print('gamw th  ', len(importance_list[layer_idx]))
                  importance_list[layer_idx][indices[h]].append(correlation)


    for l in range(len(importance_list)):
       for h in range(len(importance_list[l])):
          head_list = importance_list[l][h]
          if head_list:
            importance[l][h] = np.mean(head_list)
    # print(importance)   
    return importance
    
def b_mi(attentions_list, attributions_list):
    """
    Computes the average mutual information for each batch in each tensor in the lists.
    The input lists should contain tensors of shape (12, batch, 12, flatten_tokens).
    The output is a tensor of shape (12, 12).
    """

    # Flatten the last two dimensions of each tensor
    attentions_list = flatten_tensors(attentions_list)
    attributions_list = flatten_tensors(attributions_list)

    # List to hold the mutual information results for each batch
    all_mi_results = []

    # Iterate over each tensor in the lists
    K = len(attentions_list)

    for i in tqdm(range(K), desc="MI Calculation", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        
        attentions, attributions = attentions_list[i], attributions_list[i]

        # Ensure the tensors have the correct shape
        if attentions.shape[0] != 12 or attributions.shape[0] != 12:
            raise ValueError("Input tensors must have the first dimension of size 12")

        # Number of batches in the current tensor
        num_batches = attentions.shape[1]

        # Compute mutual information for each batch
        for batch_idx in range(num_batches):
            # Initialize a tensor to hold the mutual information results for this batch
            mi_tensor = torch.zeros((12, 12))

            # Extract the batch-specific tensors
            attention_batch = attentions[:, batch_idx, :, :]
            attribution_batch = attributions[:, batch_idx, :, :]

            # Iterate over each pair of layers and heads
            for layer_idx in range(12):
                for head_idx in range(12):
                    # Extract the corresponding vectors for the current pair
                    attention_vector = attention_batch[layer_idx, head_idx, :].numpy()
                    attribution_vector = attribution_batch[layer_idx, head_idx, :].numpy()

                    # Compute the mutual information between these vectors
                    mi = mutual_info_regression(attention_vector.reshape(-1, 1), attribution_vector)
                    mi_tensor[layer_idx, head_idx] = mi[0]

            # Add the mutual information results of this batch to the list
            all_mi_results.append(mi_tensor)

    # Stack all mutual information results and compute the mean across the batch dimension
    stacked_mi = torch.stack(all_mi_results)
    mean_mi_tensor = stacked_mi.mean(dim=0).clone().detach()

    return mean_mi_tensor

def mi(attentions_list, attributions_list):

    attributions_list = flatten_tensors(attributions_list)
    attentions_list = flatten_tensors(attentions_list)
    
    attributions_list = [attr.squeeze() for attr in attributions_list]
    attentions_list = [att.squeeze() for att in attentions_list]

    K = len(attentions_list)
    num_rows, num_cols = 12, 12
    accumulated_mi = np.zeros((num_rows, num_cols))

    for k in tqdm(range(K), desc="MI Calculation", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        attentions = attentions_list[k]
        attributions = attributions_list[k]

        for i in range(num_rows):
            for j in range(num_cols):
                attention_slice = np.array(attentions[i][j]).reshape(-1, 1)
                attribution_slice = np.array(attributions[i][j])

                # Compute mutual information
                mi = mutual_info_regression(attention_slice, attribution_slice)
                accumulated_mi[i, j] += mi[0]

    # Average the mutual information matrix across all K elements
    average_mi = accumulated_mi / K
    return average_mi

def kl_div(attentions_list, attributions_list):

    attributions_list = flatten_tensors(attributions_list)
    attentions_list = flatten_tensors(attentions_list)

    attributions_list = [attr.squeeze() for attr in attributions_list]
    attentions_list = [att.squeeze() for att in attentions_list]

    K = len(attentions_list)
    num_rows, num_cols = 12, 12
    accumulated_kl_divergence = np.zeros((num_rows, num_cols))

    for k in tqdm(range(K), desc="KL Calculation", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        attentions = attentions_list[k]
        attributions = attributions_list[k]

        for i in range(num_rows):
            for j in range(num_cols):
                attention_slice = np.array(attentions[i][j])
                attribution_slice = np.array(attributions[i][j])

                # Compute KL Divergence
                kl_div = entropy(attention_slice, attribution_slice)
                accumulated_kl_divergence[i, j] += kl_div

    # Average the KL Divergence across all K elements
    average_kl_divergence = accumulated_kl_divergence / K
    return average_kl_divergence

import pickle 

def load_boundaries_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
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

# https://github.com/sai-prasanna/bert-experiments/blob/f02280f0c0fe2429f031331f2db81d044a265f00/src/glue_metrics.py
try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).float().mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]

        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "stsb":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli_two":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli_two_half":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans_mnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
