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

mix_ratios=(0.0)
vf_coefs=(0.1)
length_types=('token')
inner_iteration_steps=(1 2 4 8)
batch_sizes=(128)
gradient_accumulation_steps_list=(16)
adap_kl_ctrl_values=(true)
init_kl_coefs=(0.02)
target_kls=(6.0)
cliprange_values=(0.2)
clipranges=(0.2)
which_gpus=(0 1 2 3)
seeds=(21 42 69)
learning_rate=0.00001
num_train_epochs=4

kl_penalty='kl'
use_score_scaling=false
use_score_norm=false
mini_batch_size=8
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

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

for seed in "${seeds[@]}"; do
for mix_ratio in "${mix_ratios[@]}"; do
for target_kl in "${target_kls[@]}"; do
for vf_coef in "${vf_coefs[@]}"; do
for rew_model_path in "${rew_models[@]}"; do
for length_type in "${length_types[@]}"; do
for inner_iteration_step in "${inner_iteration_steps[@]}"; do
for batch_size in "${batch_sizes[@]}"; do
for init_kl_coef in "${init_kl_coefs[@]}"; do
for cliprange_value in "${cliprange_values[@]}"; do
for cliprange in "${clipranges[@]}"; do
for adap_kl_ctrl in "${adap_kl_ctrl_values[@]}"; do
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi

    bs_rat=$(($batch_size/$mini_batch_size))
    run_name=ppo_bs${batch_size}_mix${mix_ratio}_kl${target_kl}_vc${vf_coef}_len${length_type}_inner${inner_iteration_step}_seed_${seed}
    gradient_accumulation_steps=${gradient_accumulation_steps_list[$exp_num % ${#gradient_accumulation_steps_list[@]}]}
    
    which_gpus=${which_gpus[$exp_num % ${#which_gpus[@]}]}
    #export CUDA_VISIBLE_DEVICES=$which_gpus

    command="python $PWD/trainers/ppo.py \
        --wandb_project $wandb_project \
        --run_name $run_name \
        --mixing_ratio ${mix_ratio} \
        --inner_iteration_steps $inner_iteration_step \
        --batch_size $batch_size \
        --max_gen_batch_size $mini_batch_size \
        --mini_batch_size $mini_batch_size \
        --target_kl ${target_kl} \
        --target ${target_kl} \
        --vf_coef ${vf_coef} \
        --reward_model $rew_model_path \
        --length_type $length_type \
        --cache_dir $cache_dir \
        --adap_kl_ctrl=$adap_kl_ctrl \
        --init_kl_coef $init_kl_coef \
        --kl_penalty $kl_penalty \
        --cliprange $cliprange \
        --cliprange_value $cliprange_value \
        --use_score_scaling=$use_score_scaling \
        --use_score_norm=$use_score_norm \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --seed $seed \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
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
done
done
done

echo "Total number of experiments: $exp_num"
