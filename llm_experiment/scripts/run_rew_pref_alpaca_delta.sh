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
start_port=29600
which_gpus=("0")

batch_sizes=(32)
num_gpus=(1)
accelerates=(true)

sft_checkpoint=""

model_names=(\
    $sft_checkpoint \
)

data_min='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength'
data_max='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_maxlength'
data_mode='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_modelength'
skew_data_merge='Asap7772/alpaca_skewexp_minlength_merged'
data_alpaca_farm='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data'
dataset_paths=($data_min $data_max $data_mode $skew_data_merge $data_alpaca_farm)

num_samples=(19000 10000 5000 2500)
label_noises=(0.0)
seeds=(24)
wds=(0.0)
gradient_checkpointings=(false)
lrs=(0.00001 0.000001)
which_exp=${1:--1}
dry_run=${2:-false}
debug=false
omp=4

config_file="$PWD/configs/accelerate/zero2-bf16.yaml"


for dataset_path in "${dataset_paths[@]}"; do
for model_name in "${model_names[@]}"; do
for num_sample in "${num_samples[@]}"; do
for label_noise in "${label_noises[@]}"; do
for lr in "${lrs[@]}"; do
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi

    port_num=$((start_port+exp_num))
    which_gpu=${which_gpus[$exp_num % ${#which_gpus[@]}]}
    num_gpu=${num_gpus[$exp_num % ${#num_gpus[@]}]}
    accelerate=${accelerates[$exp_num % ${#accelerates[@]}]}
    gradient_checkpointing=${gradient_checkpointings[$exp_num % ${#gradient_checkpointings[@]}]}
    wd=${wds[$exp_num % ${#wds[@]}]}
    batch_size=${batch_sizes[$exp_num % ${#batch_sizes[@]}]}
    seed=${seeds[$exp_num % ${#seeds[@]}]}
    
    if [[ $model_name == *"410m"* ]]; then
        gradient_accumulation_steps=32
    else
        gradient_accumulation_steps=1
    fi
    
    echo "Experiment $which_exp"
    description="Fine-tuning $model_name on $num_sample samples of reward preference data with LR $lr, Weight Decay $wd, and Batch Size $batch_size. Accelerate: $accelerate, Port $port_num, GPU: $which_gpu, Num GPUs: $num_gpu, Gradient Checkpointing: $gradient_checkpointing"
    echo -e "$description\n"

    if [[ accelerate -eq "true" ]]; then
        prefix="CUDA_VISIBLE_DEVICES=$which_gpu OMP_NUM_THREADS=$omp accelerate launch --num_processes $num_gpu --main_process_port $port_num --config_file $config_file"
    else 
        prefix="CUDA_VISIBLE_DEVICES=$which_gpu python"
    fi

    if [[ $debug == true ]]; then
        export WANDB_MODE="dryrun"
    fi

    command="$prefix trainers/reward_preference_trainer.py \
        --model_name $model_name \
        --lr $lr \
        --min_lr 0 \
        --weight_decay $wd \
        --num_samples $num_sample \
        --wandb_project $wandb_project \
        --batch_size $batch_size \
        --dataset_path \"$dataset_path\" \
        --description \"$description\" \
        --seed $seed \
        --label_noise $label_noise \
        --label_smoothing_weight $label_noise \
        --checkpoint_dir $checkpoint_dir \
        --gradient_accumulation_steps $gradient_accumulation_steps \
    "

    if [[ $gradient_checkpointing == true ]]; then
        command+=" --gradient_checkpointing"
    fi

    if [[ $which_exp -lt 0 ]]; then
        command+=" &"
    fi

    echo -e "$command\n"
    if [[ $dry_run == false ]]; then
        eval $command
        sleep 20
    fi
    exp_num=$((exp_num+1))
done
done
done
done
done
echo "Total Experiments: $exp_num"