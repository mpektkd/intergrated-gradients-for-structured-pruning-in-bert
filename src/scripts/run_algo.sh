# export TASK_NAME=mnli
# export TASK_NAME=sst2
# export TASK_NAME=cola
# export TASK_NAME=mrpc

# python3 run_glue.py \
#   --model_name_or_path google-bert/bert-base-cased \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir /home/dbekris/src/tuning/$TASK_NAME/\
#   --overwrite_output_dir 


#!/bin/bash

source ./variables.sh

# Function to generate combinations
generate_combinations() {
    local list1=("${!1}")
    local list2=("${!2}")
    local combinations=()

    for item1 in "${list1[@]}"; do
        for item2 in "${list2[@]}"; do
            combinations+=("$item1|$item2")
        done
    done

    # Return the combinations
    echo "${combinations[@]}"
}
# Function to print usage information
usage() {
  echo "Usage: $0 --world_size <value> --train_batch <value> --subset <value> [--batch_size <value>] [--samples <value>] [--port <value>] [--seed <value>] [--continue_pruning <value>]"
  exit 1
}

# Check if at least one parameter is provided
if [ $# -lt 1 ]; then
  usage
fi

# Initialize default values (if any)
world_size=""
batch_size=""
samples=""

# Parse the command line arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --world_size)
      shift
      world_size="$1"
      ;;
    --batch_size)
      shift
      batch_size="$1"
      ;;
    --samples)
      shift
      samples="$1"
      ;;
    --port)
      shift
      port="$1"
      ;;
    --seed)
      shift
      seed="$1"
      ;;
      --train_batch)
      shift
      train_batch="$1"
      ;;
      --subset)
      shift
      subset="$1"
      ;;
      --continue_pruning)
      continue_pruning_param="--continue_pruning"
      ;;
    *)
      echo "Unknown parameter: $1"
      usage
      ;;
  esac
  shift
done

# Validate that required parameters are provided
if [ -z "$world_size" ]; then
  echo "Error: --world_size is required"
  usage
fi

if [ -z "$train_batch" ]; then
  echo "Error: --train_batch is required"
  usage
fi

# Prepare the parameters for the command
if [ -n "$batch_size" ]; then
  batch_size_param="--batch_size $batch_size"
else
  batch_size_param=""
fi

if [ -n "$samples" ]; then
  samples_param="--samples $samples"
else
  samples_param=""
fi

if [ -n "$port" ]; then
  port_param="--port $port"
else
  port_param=""
fi

if [ -n "$seed" ]; then
  seed_param="--seed $seed"
else
  seed_param=""
fi

if [ -n "$train_batch" ]; then
  train_batch_param="--train_batch $train_batch"
else
  train_batch_param=""
fi

if [ -n "$subset" ]; then
  subset_param="--subset $subset"
else
  subset_param=""
fi

# set3 # rte / seed_17, exoun treksei 5 iters --> continue_pruning
       # rte / seed_128, exoun treksei 6 iters --> continue_pruning

# tasks=("${tasks_set1[@]}")
# tasks=("${tasks_set2[@]}")
# tasks=("${tasks_set3[@]}")
tasks=("${tasks_set4[@]}")

# tasks=("${tasks_cased[@]}")

# tasks=("wnli")

targets=( golden )
# targets=( None )
# targets=("None" 0 1 )

    # tasks = ("cola") # TODO: there is no fine-tuned model for cola until now

# Tasks with 3 classes
# tasks=("mnli") # 3 cl
# targets=("None" 0 1 2)

# Regression Tasks
# # tasks=("stsb") # regr
base_path=$PIPELINE_BASE_PATH

# Generate combinations and store them in an array
combinations=($(generate_combinations tasks[@] targets[@]))

# Iterate over the combinations array
for combo in "${combinations[@]}"; do
    # Split the combo using '|' as delimiter
    IFS='|' read -r task target <<< "$combo"

    # echo "Fine-tuning model on $task"
    # python3 /home/dbekris/src/transformers/examples/pytorch/text-classification/run_glue.py \
    # --model_name_or_path google-bert/bert-base-cased \
    # --task_name $task \
    # --do_train \
    # --do_eval \
    # --max_seq_length 128 \
    # --per_device_train_batch_size 32 \
    # --learning_rate 2e-5 \
    # --num_train_epochs 3 \
    # --output_dir /home/dbekris/src/tuning/$task/ \
    # --seed 42 \
    # --overwrite_output_dir 

    echo "Pruning fine-tuned model on $task"
    
    # Criterion == "mi"
    # python3 $base_path/pipeline.py --one_shot --masking_amount 0.8 --criterion mi --target $target --task_name $task --world_size $world_size $subset_param $train_batch_param $batch_size_param $samples_param $port_param $seed_param
    # python3 $base_path/pipeline.py --masking_amount 0.1 --masking_threshold 0 --criterion mi --target $target --task_name $task --world_size $world_size $subset_param $train_batch_param $continue_pruning_param $batch_size_param $samples_param $port_param $seed_param
    # python3 $base_path/pipeline.py --igspp --masking_amount 0.1 --masking_threshold 0.9 --criterion mi --target $target --task_name $task --world_size $world_size $subset_param $train_batch_param $continue_pruning_param $batch_size_param $samples_param $port_param $seed_param

    # Criterion == "corr"
    python3 $base_path/pipeline.py --one_shot --masking_amount 0.8 --criterion corr --target $target --task_name $task --world_size $world_size $subset_param $train_batch_param $batch_size_param $samples_param $port_param $seed_param
    # python3 $base_path/pipeline.py --masking_amount 0.1 --masking_threshold -1 --criterion corr --target $target --task_name $task --world_size $world_size $subset_param $train_batch_param $continue_pruning_param $batch_size_param $samples_param $port_param $seed_param
    # python3 $base_path/pipeline.py --igspp --masking_amount 0.1 --masking_threshold 0 --criterion corr --target $target --task_name $task --world_size $world_size $subset_param $train_batch_param $continue_pruning_param $batch_size_param $samples_param $port_param $seed_param
    # python3 $base_path/pipeline.py --isp --masking_amount 0.1 --masking_threshold -1 --criterion corr --target $target --task_name $task --world_size $world_size $subset_param $train_batch_param $continue_pruning_param $batch_size_param $samples_param $port_param $seed_param

done


# Criterion == "mi"
# python3 ${base_path}/pipeline.py --one_shot --masking_amount 0.8 --criterion mi --target 2 --task_name mnli --world_size $world_size $batch_size_param $samples_param $port_param $seed_param
# python3 ${base_path}/pipeline.py --masking_amount 0.1 --masking_threshold 0 --criterion mi --target 2 --task_name mnli --world_size $world_size $batch_size_param $samples_param $port_param $seed_param
# python3 ${base_path}/pipeline.py --igspp --masking_amount 0.1 --masking_threshold 0.9 --criterion mi --target 2 --task_name mnli --world_size $world_size $batch_size_param $samples_param $port_param $seed_param

# Criterion == "corr"
# python3 ${base_path}/pipeline.py --one_shot --masking_amount 0.8 --criterion corr --target 2 --task_name mnli --world_size $world_size $batch_size_param $samples_param $port_param $seed_param
# python3 ${base_path}/pipeline.py --masking_amount 0.1 --masking_threshold 0 --criterion corr --target 2 --task_name mnli --world_size $world_size $batch_size_param $samples_param $port_param $seed_param
# python3 ${base_path}/pipeline.py --igspp --masking_amount 0.1 --masking_threshold 0.9 --criterion corr --target 2 --task_name mnli --world_size $world_size $batch_size_param $samples_param $port_param $seed_param




# tasks=("mnli" "qnli")

# Iterate through the array
# for task in "${tasks[@]}"
# do
  # echo "Fine-tuning on $task"
  # python3 /home/dbekris/src/transformers/examples/pytorch/text-classification/run_glue.py \
  # --model_name_or_path google-bert/bert-base-cased \
  # --task_name $task \
  # --do_train \
  # --do_eval \
  # --max_seq_length 128 \
  # --per_device_train_batch_size 32 \
  # --learning_rate 2e-5 \
  # --num_train_epochs 3 \
  # --output_dir /home/dbekris/src/tuning/$task/ \
  # --overwrite_output_dir 

  # echo "Pruning fine-tuned model on $task"

  # python3 ${base_path}/pipeline.py --one_shot --criterion mi --target 0 --task_name $task --world_size 2 
# done
  
  # 2 classes:                3 classes:            Regression(sim score)
  # cola (10.7K) RUN               mnli (432K) RUN               stsb (8.63K)
  # mrpc (5.8K)  RUN
  # qnli (116K)  RUN
  # qqp  (795K)  RUN
  # rte  (5.77K)
  # sst2  (70K)  RUN
  # wnli  (852)
