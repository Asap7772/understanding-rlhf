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

lrs=(1e-5)
temperatures=(1.0 10.0)
mix_ratios=(0.0)
use_gold_rews=(false)
num_actions_per_prompts=(1)
inner_iteration_steps=(1)
batch_sizes=(64 128 256 512)
gradient_accumulation_steps_list=(8)
mini_batch_size=8
use_length_reward=false
length_types=(token)
add_baselines=(false)
baseline_type="moving_average"
seeds=(24 42)


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

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

for seed in "${seeds[@]}"; do
for temperature in "${temperatures[@]}"; do
for mix_ratio in "${mix_ratios[@]}"; do
for use_gold_rew in "${use_gold_rews[@]}"; do
for num_actions_per_prompt in "${num_actions_per_prompts[@]}"; do
for length_type in "${length_types[@]}"; do
for rew_model_path in "${rew_models[@]}"; do
for inner_iteration_step in "${inner_iteration_steps[@]}"; do
for batch_size in "${batch_sizes[@]}"; do
for add_baseline in "${add_baselines[@]}"; do
for lr in "${lrs[@]}"; do
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi
    echo "Experiment: $exp_num"
    echo "Seed: $seed"
    echo "Temperature: $temperature"
    echo "Mix Ratio: $mix_ratio"
    echo "Use Gold Reward: $use_gold_rew"
    echo "Num Actions Per Prompt: $num_actions_per_prompt"
    echo "Length Type: $length_type"
    echo "Rew Model Path: $rew_model_path"
    echo "Inner Iteration Step: $inner_iteration_step"
    echo "Batch Size: $batch_size"
    echo "Add Baseline: $add_baseline"
    echo "Baseline Type: $baseline_type"
    echo "Learning Rate: $lr"

    bs_rat=$(($batch_size/$mini_batch_size))
    run_name=rbc_gold${use_gold_rew}_mix${mix_ratio}_temp${temperature}_act${num_actions_per_prompt}_len${length_type}_inner${inner_iteration_step}
    gradient_accumulation_steps=${gradient_accumulation_steps_list[$exp_num % ${#gradient_accumulation_steps_list[@]}]}

    command="python $PWD/trainers/reweighted_bc.py \
        --wandb_project $wandb_project \
        --run_name $run_name \
        --mixing_ratio ${mix_ratio} \
        --inner_iteration_steps $inner_iteration_step \
        --batch_size $batch_size \
        --max_gen_batch_size $mini_batch_size \
        --mini_batch_size $mini_batch_size \
        --temperature ${temperature} \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --num_actions_per_prompt $num_actions_per_prompt \
        --reward_model $rew_model_path \
        --num_train_epochs 2 \
        --clip_weighting \
        --learning_rate $lr \
        --seed $seed \
    "

    if [[ $use_length_reward = true ]]; then
        command+="--use_length_reward "
    fi

    if [[ $use_gold_rew = true ]]; then
        command+="--use_gold_reward_model "
    fi

    if [[ $use_tpu = true ]]; then
        command+="--use_tpu "
    fi

    if [[ $add_baseline = true ]]; then
        command+="--add_baseline --baseline_type $baseline_type "
    fi

    if [[ $which_exp -lt 0 ]]; then
        command+=" &"
    fi
    echo -e "$command\n"
    if [ $dryrun = false ]; then
        eval $command
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
done
done
echo "Total number of experiments: $exp_num"