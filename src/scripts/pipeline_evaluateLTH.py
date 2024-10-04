import multiprocessing as mp
import igspp_prune
import datasets.packaged_modules
import re
from variables import *
'''

fine_tune() # --> fine tune
mask_heads() --> call compute_head_importance
prune()

'''
import datasets
from typing import Union

from sklearn.model_selection import StratifiedShuffleSplit
from transformers import (
    default_data_collator,
    PretrainedConfig,
    AutoConfig, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    HfArgumentParser, 
    TrainingArguments
)
from typing import Optional
import sys
from dataclasses import dataclass, field

from datasets import load_dataset, concatenate_datasets

import argparse

from prune import *
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

base_path = EVALUATE_BASE_PATH
base_path2 = EVALUATE_BASE_PATH2
_samples=SAMPLES

max_seq_length=128 

batch_size = 64
seed = SEED


def is_empty_directory(path):
    # Check if the directory is empty
    if not os.listdir(path):
        return True
    return False

def get_sorted_directories(base_path):
    # Dictionary to hold task directories and their corresponding head_mask directories
    task_dict = {}

    # Walk through the base directory
    for root, dirs, files in os.walk(base_path):
        # Extract the task and head_mask information
        match = re.search(rf'/target_golden/one_shot_ISP/(?P<task>[^/]+)/balanced_samples_{_samples}/head_mask_(?P<head_mask>\d+)$', root)
        if match:
            task = match.group('task')
            head_mask = int(match.group('head_mask'))

            if task not in task_dict:
                task_dict[task] = []
            task_dict[task].append((head_mask, root))

    # Sort tasks alphabetically and head_mask numerically
    sorted_tasks = sorted(task_dict.keys())
    sorted_task_dict = {}
    for task in sorted_tasks:
        # Filter out empty directories
        non_empty_dirs = [(head_mask, directory) for head_mask, directory in task_dict[task] if not is_empty_directory(directory)]
        if non_empty_dirs:
            sorted_task_dict[task] = sorted(non_empty_dirs, key=lambda x: x[0])

    return sorted_task_dict

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

def load_configuration(model_name_or_path=None, task_name=None, max_seq_length=None):
    # Disable logging
    logger = logging.getLogger("prune")
    # logging.disable(logging.CRITICAL)
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    data_args = DataTrainingArguments(task_name=task_name, max_seq_length=max_seq_length)
    
    
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "nyu-mll/glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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


    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    def get_length(examples):
        ''' 
            Only tokenization here for DistributedBucketSampling
        '''
        result = {}

        sent1 = tokenizer.tokenize(examples[sentence1_key])
        result["total_length"] = len(sent1)

        if sentence2_key:
            sent2 = tokenizer.tokenize(examples[sentence2_key])
            result["total_length"] += len(sent2)

        return result

    # n = len(raw_datasets['train'])
    # length = samples if samples < n else n

    balanced_subset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"].map(
        get_length,
        desc="Running tokenization on samples for computing the total length before providing them on collator for bucket encoding",
    )
    
    t_config = {
        "tokenizer": tokenizer,
        "pad_to_max_length": data_args.pad_to_max_length,
        "max_seq_length": data_args.pad_to_max_length,
        "num_labels": num_labels,
        "is_regression": is_regression,
        "label_list": label_list,
        "overwrite_cache": data_args.overwrite_cache,
        "cls_id": tokenizer.cls_token_id,
        "sep_id": tokenizer.sep_token_id
    }

    return model, balanced_subset, t_config, model_args, data_args, num_labels

def pipe():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--output_mode",
            default="classification",
            type=str,
            required=False,
        )

    args = parser.parse_args()
    

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

    create_directories(base_path2)

    log_file = base_path2 + "evaluation_logfile.log"

    # useless, since in the tokenization, i am using the default length of the tokenizer
    # it is not used, so we use the default model input length, as the sentences of the dataset are short. 
    # IMDB has large sentences


    # Set seed before dataset loading for ensure reproducability
    set_seed(seed)

    import os

    # Setup logging

    logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Set the logging format
    filename=log_file,  # Specify the file to save the logs
    filemode='w'  # Set the file mode to write (overwrite the log file each time)
    )
        
    evaluation_dict = {}
    # with open(base_path + '/evaluation_scores.pkl', 'rb') as f:
    #     evaluation_dict = pickle.load(f)
    
    # The dirs are returned sorted based on the head mask number, so we can use a list
    sorted_directories = get_sorted_directories(base_path)

    for task, directories in sorted_directories.items():
        
        print(f"Task: {task}")
        
        evaluation_dict[task] = []

        for head_mask, model_name_or_path in directories:
            model, dataset, t_config, model_args, data_args, num_labels = load_configuration(model_name_or_path, task, max_seq_length)

            # Define encoding kwargs
            e_config = {
            "return_token_type_ids": True,
            "padding": True,
            "truncation": True, # it is already done in pre-processing
            "return_tensors": "pt"
            }

            t_config["e_config"] = e_config
            t_config["task"] = task
            
            # Define parameters for the rest of the algorithm
            m_config = {
                "model": model,
                "dataset": dataset,
                "task": task,
                "batch_size": batch_size,
                "seed": seed,
                "model_args": model_args,
                "data_args": data_args,
                "num_labels": num_labels,
            }

            kwargs = {
                "m_config": m_config,
                "t_config": t_config,
                "output_mode": args.output_mode,
                "save_mask_all_iterations": True,
                "metric_name": task_metric[task],
            }
                        
            print(f"Head Mask: {head_mask}, Directory: {model_name_or_path}")
        
            
            device = get_first_available_cuda_device()
            preds, labels = inference(device, m_config, t_config)
            

            # _, preds, labels = compute_heads_importance(m_config, t_config, kwargs["world_size"], head_mask=None,  inference=True)

            preds = np.argmax(preds, axis=1) if kwargs["output_mode"] == "classification" else np.squeeze(preds)
            score_pruning = glue_compute_metrics(m_config["task"], preds, labels)[kwargs["metric_name"]]
            
            evaluation_dict[task].append(score_pruning)
            
            logger.info(f"Task {task}, head_mask {head_mask}:")
            logger.info("Score with pruning: %.4f", score_pruning)

    # Save sorted boundaries dictionary to a pickle file
    with open(base_path2 + 'evaluation_scores.pkl', 'wb') as f:
        pickle.dump(evaluation_dict, f)

if __name__ == "__main__":

    pipe()