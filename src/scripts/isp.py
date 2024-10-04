import re
'''
    Prune the heads of the BERT model. 
    
    1st phase:
        Direct pruning
'''

from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
)
from lib import *
from datetime import datetime
from torch.utils.data import DataLoader

from distributed_attributions import dist_attribution

from run_glue_func import main

import logging 
logger = logging.getLogger(__name__)

def setup_logging(log_file):
    # Create a custom logger
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)

    # Remove all handlers associated with the root logger object to prevent logging to stdout
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logs = ""
    logs += "lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))) + "\n"
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logs += f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data) + "\n"
        else:
            logs += f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data) + "\n"
    
    logger.info(logs)

def print_list_of_lists(data):
    for sublist in data[0]:
        
        formatted_sublist = [f"{item:.3f}" if isinstance(item, float) else item for item in sublist]
        print(formatted_sublist)

def load_model(m_config):
    
    model_args = m_config["model_args"]
    data_args = m_config["data_args"]
    num_labels = m_config["num_labels"]

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        output_attentions=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes, 
        attn_implementation="eager"
    )

    return config, tokenizer, model

def get_largest_common_numbered_files(folder_path):
    # Patterns to match files with the names 'head_mask' and 'head_importance' followed by a number
    mask_pattern = re.compile(r'head_mask_(\d+)\.npy')
    importance_pattern = re.compile(r'head_importance_(\d+)\.npy')

    largest_num = -1
    largest_mask_file = None
    largest_importance_file = None

    # Dictionary to store matched numbers
    matched_numbers = {}

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        mask_match = mask_pattern.match(filename)
        importance_match = importance_pattern.match(filename)
        
        if mask_match:
            # Extract the number from the head_mask filename
            mask_number = int(mask_match.group(1))
            matched_numbers[mask_number] = matched_numbers.get(mask_number, 0) | 1
        
        if importance_match:
            # Extract the number from the head_importance filename
            importance_number = int(importance_match.group(1))
            matched_numbers[importance_number] = matched_numbers.get(importance_number, 0) | 2

    # Find the largest number that has both patterns
    for number, match_value in matched_numbers.items():
        if match_value == 3 and number > largest_num:
            largest_num = number
            largest_mask_file = f"head_mask_{number}.npy"
            largest_importance_file = f"head_importance_{number}.npy"

    return largest_num, largest_mask_file, largest_importance_file

# https://github.com/sai-prasanna/bert-experiments/blob/f02280f0c0fe2429f031331f2db81d044a265f00/src/find_head_masks.py#L63
# def mask_heads(args, model, eval_dataloader):
def mask_heads(**kwargs):
    """ 
        1. This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
        2.  Assumption for constant attention heads
    """
    def convert_to_head_dict(head_mask):
        
        heads_to_prune = {}

        for layer in range(len(head_mask)):
            heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
            heads_to_prune[layer] = heads_to_mask
        
        assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
        
        return heads_to_prune
    
    m_config = kwargs["m_config"]
    t_config = kwargs["t_config"]
    one_shot = kwargs["one_shot"]

    continue_pruning = kwargs["continue_pruning"]
    
    train_batch = kwargs["train_batch"]

    if continue_pruning:
        last_num, last_mask_file, last_importance_file = get_largest_common_numbered_files(kwargs["heads_output_dir"])

        new_head_mask = np.load(os.path.join(kwargs["heads_output_dir"], last_mask_file))
        new_head_mask = torch.from_numpy(new_head_mask)

        head_importance = np.load(os.path.join(kwargs["heads_output_dir"], last_importance_file))
        head_importance = torch.from_numpy(head_importance)
    
        original_score = 0.674

        i = last_num

    else:
        head_importance, preds, labels = compute_heads_importance(m_config, t_config, kwargs["world_size"], head_mask=None,  inference=False)


        preds = np.argmax(preds, axis=1) if kwargs["output_mode"] == "classification" else np.squeeze(preds)

        original_score = glue_compute_metrics(m_config["task"], preds, labels)[kwargs["metric_name"]]
        logger.info("Pruning: original score: %f, threshold: %f", original_score, original_score * kwargs["masking_threshold"])

        # head_importance = torch.from_numpy(head_importance)
        new_head_mask = torch.ones_like(head_importance)

        i = 0
    
    current_score = original_score

    while current_score >= original_score * kwargs["masking_threshold"] or one_shot:            

        # Get the indices of the surviving heads
        survived_heads = new_head_mask == 1
        survived_indices = torch.nonzero(survived_heads)

        # Flatten indices into 1D to correctly index new_head_mask
        flat_survived_indices = survived_indices[:, 0] * survived_heads.size(1) + survived_indices[:, 1]

        # Calculate the number of heads to mask
        num_to_mask = max(1, int(new_head_mask.flatten()[flat_survived_indices].numel() * kwargs["masking_amount"]))

        # print("percentaged score", original_score * kwargs["masking_threshold"])
        # print("current score", current_score)
        head_mask = new_head_mask.clone()  # save current head mask

        if kwargs["save_mask_all_iterations"] and (not continue_pruning or (i != last_num and continue_pruning)):
            np.save(os.path.join(kwargs["heads_output_dir"], f"head_mask_{i}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(kwargs["heads_output_dir"], f"head_importance_{i}.npy"), head_importance.detach().cpu().numpy())

        if one_shot and i == 1:
            break
        
        i += 1
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]
        
        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[head.item()] == float("Inf"):
                break
            layer_idx = head.item() // 12 # model.bert.config.num_attention_heads
            head_idx = head.item() % 12 # model.bert.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())

        if not selected_heads_to_mask:
            break

        logger.info("Heads to mask: %s", str(selected_heads_to_mask))
        
        #new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)
        
        # Check if there is a row witl zero elements and stop. Because there is an issue on the inference on gpu
        # has_zero_row = torch.any(torch.all(new_head_mask == 0, dim=1))
        # print(new_head_mask)
        # if has_zero_row:
        #     break
        
        # Prune model
        model_name_or_path = m_config["igspp_model_path"]

        m_config["model_args"].model_name_or_path = model_name_or_path
        config, tokenizer, model = load_model(m_config)
        heads_to_prune = convert_to_head_dict(new_head_mask)
        
        model.prune_heads(heads_to_prune)

        # Fine tune model and save it
        model_name_or_path = os.path.join(kwargs["output_dir"], f"head_mask_{i}/")
        main(config, tokenizer, model, model_name_or_path, m_config["data_args"].task_name, m_config['seed'], per_device_train_batch_size=train_batch, output_dir=model_name_or_path, save_strategy='no')
        torch.cuda.empty_cache()
        
        m_config["model_args"].model_name_or_path = model_name_or_path
        _, _, model = load_model(m_config)
        m_config['model'] = model

        # Compute metric and head importance again
        head_importance, preds, labels = compute_heads_importance(m_config, t_config, kwargs["world_size"], head_mask=new_head_mask)
        # head_importance = torch.from_numpy(head_importance)
        # print('head_importance', head_importance)

        preds = np.argmax(preds, axis=1) if kwargs["output_mode"] == "classification" else np.squeeze(preds)
        current_score = glue_compute_metrics(m_config["task"], preds, labels)[kwargs["metric_name"]]
        logger.info(
            "Masking: current score: %f, remaning heads %d (%.3f percents)",
            current_score,
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )
        
    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(kwargs["heads_output_dir"], "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask


def inference(device, m_config, t_config):

    preds = []
    labels = []

    model = m_config["model"]

    model = model.to(device)
    # implementation for bucketing and distributed sampling
    
    sampler = GLUEBatchSamplerSimilarLength(m_config["dataset"], shuffle=False, batch_size=m_config["batch_size"])
    
    collator = PruningCollator(t_config)
    data_loader = DataLoader(
        m_config["dataset"], 
        batch_sampler=sampler,
        # batch_size=1, 
        # collate_fn=collator 
        collate_fn=collator # because the load_configuration() does the encoding on preprocessing
    )  

    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader), desc="Inference Samples", ncols=75, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            
            # we do not use 'label' as we do no care about the loss
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            label = d["label"]

            pred, _ = predict(
                model,
                inputs=input_ids,
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )

            pred = pred.cpu()

            preds.append(pred)
            labels.append(label)

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    model = model.to("cpu")

    return preds, labels

# https://github.com/sai-prasanna/bert-experiments/blob/f02280f0c0fe2429f031331f2db81d044a265f00/src/find_head_masks.py#L63
def prune_heads(**kwargs):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    head_mask = kwargs["head_mask"]
    m_config = kwargs["m_config"]
    t_config = kwargs["t_config"]

    before_time = datetime.now()

    _, preds, labels = compute_heads_importance(m_config, t_config, kwargs["world_size"], head_mask=head_mask, inference=True)

    # Deserialize the parameteres
    model = m_config["model"]

    preds = np.argmax(preds, axis=1) if kwargs["output_mode"] == "classification" else np.squeeze(preds)
    score_masking = glue_compute_metrics(m_config["task"], preds, labels)[kwargs["metric_name"]]
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())

    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask
    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()

    # Assuming `model` is your PyTorch model
    def get_model_device(model):
        # Get the first parameter of the model
        param = next(model.parameters())
        # Get the device of the parameter
        device = param.device
        return device

    def check_cuda_device(model):
        device = get_model_device(model)
        if device.type == 'cuda':
            print(f"Model is on CUDA device with index {device.index}")
        else:
            print("Model is not on a CUDA device")

    # check_cuda_device(model)

    # heads_to_prune = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [11], 9: [], 10: [6], 11: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]}
    logger.info(f"{heads_to_prune}")

    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
  
    '''
        1. For now i am using the inference() function for sinmgle gpu implementation. 
        2. For multiprocessing, there is an issue when i assign the model in the cuda. Probably, the parameters
            sharing is not properly applied. TODO: Solve this issue in refactoring.
    '''
    # device = get_first_available_cuda_device()
    device = "cpu"
    preds, labels = inference(device, m_config, t_config)

    # _, preds, labels = compute_heads_importance(m_config, t_config, kwargs["world_size"], head_mask=None,  inference=True)

    preds = np.argmax(preds, axis=1) if kwargs["output_mode"] == "classification" else np.squeeze(preds)
    score_pruning = glue_compute_metrics(m_config["task"], preds, labels)[kwargs["metric_name"]]
    new_time = datetime.now() - before_time

    logger.info(
        "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
        original_num_params,
        pruned_num_params,
        pruned_num_params / original_num_params * 100,
    )
    logger.info("Pruning: score with masking: %.2f score with pruning: %.2f", score_masking, score_pruning)
    logger.info("Pruning: speed ratio (new timing / original timing): %.2f ", original_time / new_time)


# Fine tune: ~/src/transformers/examples/pytorch/text-classification
def count_heads(model):
    remaining_heads = {}
    for i, layer in enumerate(model.bert.encoder.layer):
        remaining_heads[i] = layer.attention.self.num_attention_heads
    return remaining_heads

def compute_heads_importance(m_config, t_config, world_size, head_mask=None, inference=False):

    final_metric, preds, labels = dist_attribution(m_config, t_config, world_size=world_size, head_mask=head_mask,  inference=inference)
    
    return final_metric, preds, labels

