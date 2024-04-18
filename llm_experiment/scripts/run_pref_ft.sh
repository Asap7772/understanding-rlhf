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

which_exp=${1:--1}
dryrun=false
debug=false

# TODO: Fill in the following variable
sft_model_path=""

gradient_accumulation_steps=8
batch_size=16
mini_batch_size=2

data_min='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength'
data_max='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_maxlength'
data_mode='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_modelength'
skew_data_merge='Asap7772/alpaca_skewexp_minlength_merged'
alpaca_farm='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data'
preference_dataset_paths=($data_min $data_max $data_mode $skew_data_merge $alpaca_farm)

lrs=(1e-5 1e-6 1e-7 5e-7 1e-8) 

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

for preference_dataset_path in "${preference_dataset_paths[@]}"; do
for lr in "${lrs[@]}"; do
    dataset_basename=$(basename -- $preference_dataset_path)
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi

    run_name="pref_ft_${dataset_basename}_lr${lr}_bs${batch_size}_gradacc${gradient_accumulation_steps}"

    command="python $PWD/trainers/sft_yplus.py \
        --wandb_project $wandb_project \
        --run_name $run_name \
        --inner_iteration_steps 1 \
        --batch_size $batch_size \
        --mini_batch_size $mini_batch_size \
        --pretrained_dir $sft_model_path \
        --preference_dataset_path $preference_dataset_path \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --cache_dir $cache_dir \
        --learning_rate $lr \
    "

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
echo "Finished running $exp_num experiments"