# TODO: Fill in the following variables
cache_dir=""
wandb_project=""
model_name=""
output_dir=''
export WANDB_PROJECT=$wandb_project
export WANDB_API_KEY=
export WANDB_USERNAME=
export WANDB_USER_EMAIL=
export HF_DATASETS_CACHE=$cache_dir
env_name=''

eval "$(conda shell.bash hook)"
conda activate $env_name # Activate the environment


exp_num=0
which_exp=${1:--1}
dryrun=false
debug=false
use_tpu=true
which_gpus=("0" "1" "2" "3")

wandb_dir=/tmp/wandb/$RANDOM
mkdir -p $wandb_dir
export WANDB_DIR=$wandb_dir

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi


output_dir="YOUR_OUTPUT_DIR"
models=("EleutherAI/pythia-70m" "EleutherAI/pythia-410m" "EleutherAI/pythia-1b" "EleutherAI/pythia-1.4b")
dataset_paths=("tatsu-lab/alpaca_farm")

for dataset_path in "${dataset_paths[@]}"; do
for model in "${models[@]}"; do
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi

    mnsimple=$(basename -- "$model")
    dsimple=$(basename -- "$dataset_path")
    output_dir="$output_dir/$wandb_project/$mnsimple-$dsimple-$RANDOM"
    
    echo "wandb_project: $wandb_project"
    echo "model: $model"
    echo "dataset_path: $dataset_path"
    echo "output_dir: $output_dir"

    which_gpu=${which_gpus[$exp_num % ${#which_gpus[@]}]}
    echo "which_gpu: $which_gpu"

    command="CUDA_VISIBLE_DEVICES=$which_gpu python $PWD/trainers/sft.py \
        --dataset_path \"$dataset_path\" \
        --pretrained_dir=\"$model\" \
        --output_dir=\"$output_dir\" \
        --num_train_epochs 10 \
    "
    
    if [[ $use_tpu = true ]]; then
        command+="--use_tpu "
    fi
    
    if [[ $which_exp -lt 0 ]]; then
        command+=" &"
    fi
    
    echo -e "$command\n"
    if [ $dryrun = false ]; then
        eval $command
        sleep 20
    fi
    
    exp_num=$((exp_num+1))
done
done