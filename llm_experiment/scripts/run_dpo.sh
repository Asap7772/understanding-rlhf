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

wandb_project="01_30_dpo_ablation_all_datasets"
which_exp=${1:--1}
dryrun=false
debug=false
lrs=(1e-7 5e-7) 
betas=(0.05 0.1)
gradient_accumulation_steps=8
batch_size=16
mini_batch_size=2

data_min='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength'
data_max='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_maxlength'
data_mode='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_modelength'
skew_data_merge='Asap7772/alpaca_skewexp_minlength_merged'
alpaca_farm='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data'
preference_dataset_paths=($data_min $data_max $data_mode $skew_data_merge $alpaca_farm)
ipo_loss=false
# preference_dataset_paths=($data_min $data_max)

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

for preference_dataset_path in "${preference_dataset_paths[@]}"; do
for beta in "${betas[@]}"; do
for lr in "${lrs[@]}"; do
    dataset_basename=$(basename -- $preference_dataset_path)
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi

    run_name="dpo_${dataset_basename}_beta${beta}_lr${lr}_bs${batch_size}_gradacc${gradient_accumulation_steps}"
    echo "Running experiment $exp_num: $run_name"

    command="python $PWD/trainers/dpo.py \
        --wandb_project $wandb_project \
        --run_name $run_name \
        --inner_iteration_steps 1 \
        --batch_size $batch_size \
        --mini_batch_size $mini_batch_size \
        --pretrained_dir $model_name \
        --preference_dataset_path $preference_dataset_path \
        --temperature $beta \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --cache_dir $cache_dir \
        --learning_rate $lr \
        --output_dir $output_dir \
    "

    if [[ $ipo_loss = true ]]; then
        command+="--ipo_loss "
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
echo "Finished running $exp_num experiments"