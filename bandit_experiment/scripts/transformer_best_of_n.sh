lr_list=(0.0001)
seeds=(1 2 3 4 5)
#train_states_list=(5 10 20 50 100 200)
train_states_list=(10)
algorithm="best_of_n"
reward_functions=(7)
epoch_list=(1)
N_values=(10)
num_inner_iterations=(100)
policy='categorical'


for reward_function in "${reward_functions[@]}"; do 
for lr in "${lr_list[@]}"; do
for train_state in "${train_states_list[@]}"; do
for epochs in "${epoch_list[@]}"; do
for N in "${N_values[@]}"; do
for num_iterations in "${num_inner_iterations[@]}"; do
for seed in "${seeds[@]}"; do
    filedir=transformer_final_negative_gradient_experiments_reward_function_${reward_function}
    project_name=transformer_final_negative_gradient_experiments_reward_function_${reward_function}

    command="python transformer_hyperparam_tuning.py --algorithm $algorithm \
        --reward_function $reward_function \
        --epochs $epochs \
        --N $N \
        --num_iterations_on_same_samples $num_iterations \
        --filedir $filedir \
        --lr $lr \
        --should_integrate_wandb \
        --project_name $project_name \
        --num_train_states $train_state \
        --seed $seed \
    "
    echo -e "$command\n"
    eval $command
done
done
done
done
done
done
done
