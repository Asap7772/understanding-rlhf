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

lrs=(1e-6)
ns=(4)
ks=(1)
kl_weights=(0.0)
batch_sizes=(64 128 256 512)
save_every_steps=(32 16 8 4)
inner_iteration_steps=(1)
gradient_accumulation_steps_list=(8)
length_types=('token')
which_gpus=("0,1,2,3")
seeds=(24 42 69)

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

exp_num=0
which_exp=${1:--1}
dryrun=false
debug=false

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

for lr in "${lrs[@]}"; do
for seed in "${seeds[@]}"; do
for length_type in "${length_types[@]}"; do
for n in "${ns[@]}"; do
for k in "${ks[@]}"; do
for rew_model_path in "${rew_models[@]}"; do
for kl_weight in "${kl_weights[@]}"; do
for inner_iteration_step in "${inner_iteration_steps[@]}"; do
for batch_size in "${batch_sizes[@]}"; do
    if [[ $n -lt $k || $n -eq k ]]; then
        continue
    fi

    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi
    
    gradient_accumulation_steps=${gradient_accumulation_steps_list[$exp_num % ${#gradient_accumulation_steps_list[@]}]}
    run_name=bofn_bs_${batch_size}_n_${n}_k_${k}_kl_${kl_weight}_inner_${inner_iteration_step}_seed_${seed}

    echo "Experiment: $exp_num"
    echo "Seed: $seed"
    echo "KL: $kl_weight"
    echo "Num Actions Per Prompt: $n"
    echo "Num Actions Keep: $k"
    echo "Rew Model Path: $rew_model_path"
    echo "Inner Iteration Step: $inner_iteration_step"
    echo "Batch Size: $batch_size"
    echo "Learning Rate: $lr"
    
    which_gpus=${which_gpus[$exp_num % ${#which_gpus[@]}]}
    echo "Using GPUs: $which_gpus"
    export CUDA_VISIBLE_DEVICES=$which_gpus
    save_every_step=${save_every_steps[$exp_num % ${#save_every_steps[@]}]}
    
    command="python $PWD/trainers/bofn.py \
        --wandb_project $wandb_project \
        --run_name $run_name  \
        --inner_iteration_steps $inner_iteration_step \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --mini_batch_size $mini_batch_size \
        --max_gen_batch_size $mini_batch_size \
        --num_actions_per_prompt $n \
        --num_actions_keep $k \
        --reward_model $rew_model_path \
        --kl_weight $kl_weight \
        --cache_dir $cache_dir \
        --use_score_scaling=$use_score_scaling \
        --use_score_norm=$use_score_norm \
        --learning_rate $lr \
        --seed $seed \
        --num_train_epochs 1 \
        --eval_every_steps 20 \
        --save_every_steps $save_every_step \
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
