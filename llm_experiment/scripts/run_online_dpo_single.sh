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

mix_ratios=(0.0 0.5 0.7 0.9)
length_types=('token')

inner_iteration_steps=(2)
batch_sizes=(64)
gradient_accumulation_steps_list=(32)

seeds=(42)
preference_thresholds=(0.0 0.1 0.5 1.0)
learning_rates=(1e-5)
temperatures=(0.1 0.5 0.05 0.01)
num_train_epochs=2
max_gen_batch_size=8
which_gpus=("0,1,2,3")

kl_penalty='kl'
use_score_scaling=false
use_score_norm=false
mini_batch_size=2
use_length_reward=false

# TODO: Fill in the following variables
rew_model_min=''
rew_model_max=''
rew_model_mode=''
rew_model_skew=''
rew_model_alpaca=''
rew_models=(
    $rew_model_min
    $rew_model_max
    $rew_model_mode
    $rew_model_skew
    $rew_model_alpaca
)

data_min='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength'
data_max='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_maxlength'
data_mode='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_modelength'
data_skew_data='Asap7772/alpaca_skewexp_minlength_merged'
data_alpaca_farm='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data'
preference_dataset_paths=($data_min $data_max $data_mode $data_skew_data $data_alpaca_farm)


if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

for seed in "${seeds[@]}"; do
for learning_rate in "${learning_rates[@]}"; do
for mix_ratio in "${mix_ratios[@]}"; do
for length_type in "${length_types[@]}"; do
for inner_iteration_step in "${inner_iteration_steps[@]}"; do
for batch_size in "${batch_sizes[@]}"; do
for preference_dataset_path in "${preference_dataset_paths[@]}"; do
for temperature in "${temperatures[@]}"; do
for preference_threshold in "${preference_thresholds[@]}"; do
    seed=$((RANDOM % 1000))
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi

    bs_rat=$(($batch_size/$mini_batch_size))
    
    run_name="online_dpo_${exp_num}_seed_${seed}_lr_${learning_rate}_beta_${temperature}_th_${preference_threshold}_mix_${mix_ratio}_len_${length_type}_inner_${inner_iteration_step}_bs_${bs_rat}"
    gradient_accumulation_steps=${gradient_accumulation_steps_list[$exp_num % ${#gradient_accumulation_steps_list[@]}]}
    rew_model_path=${rew_models[$exp_num % ${#rew_models[@]}]}
    
    which_gpus=${which_gpus[$exp_num % ${#which_gpus[@]}]}
    echo "Using GPUs: $which_gpus"
    export CUDA_VISIBLE_DEVICES=$which_gpus

    command="python $PWD/trainers/online_dpo.py \
        --wandb_project $wandb_project \
        --run_name $run_name \
        --mixing_ratio ${mix_ratio} \
        --inner_iteration_steps $inner_iteration_step \
        --batch_size $batch_size \
        --max_gen_batch_size $max_gen_batch_size \
        --mini_batch_size $mini_batch_size \
        --reward_model $rew_model_path \
        --length_type $length_type \
        --cache_dir $cache_dir \
        --preference_dataset_path $preference_dataset_path \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --seed $seed \
        --temperature $temperature \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --preference_threshold $preference_threshold \
    "

    if [[ $use_length_reward = true ]]; then
        command+="--use_length_reward "
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
done
done
done
done
done
done
done

echo "Total number of experiments: $exp_num"

