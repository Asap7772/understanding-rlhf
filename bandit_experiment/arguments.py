import argparse


def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-reward_function", 
        "--reward_function", 
        type=int, 
        default=1, 
    )

    ap.add_argument(
        "-should_integrate_wandb",
        "--should_integrate_wandb",
        action="store_true",
    )

    ap.add_argument(
        "-project_name",
        "--project_name",
        type=str,
        default="bandit"
    )

    ap.add_argument(
        "-epochs",
        "--epochs",
        type=int,
        default=100,
    )

    ap.add_argument(
        "-algorithm",
        "--algorithm",
        type=str,
        default="dpo",
        choices=["dpo", "reinforce", "best_of_n", "ipo", "ppo", "awr"],
    )

    ap.add_argument(
        "-filedir",
        "--filedir",
        type=str,
        default="hyperparam_tuning_results",
    )

    ap.add_argument(
        "-beta",
        "--beta",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "-lr",
        "--lr",
        type=float,
        default=None,
    )

    ap.add_argument(
        "-seed",
        "--seed",
        type=int,
        default=1,
    )

    ap.add_argument(
        "-policy",
        "--policy",
        type=str,
        default="tanh_gaussian",
        choices=[
            "gaussian", 
            "tanh_gaussian", 
            "tanh_cauchy", 
            "cauchy", 
            "categorical",
        ],
    )

    ap.add_argument(
        "-gradient_norm",
        "--gradient_norm",
        type=float,
        default=None,
    )

    ap.add_argument("-N", "--N", type=int, default=1)

    ap.add_argument(
        "-num_train_states",
        "--num_train_states",
        type=int,
        default=100,
    )

    ap.add_argument(
        "-num_eval_states",
        "--num_eval_states",
        type=int,
        default=100,
    )

    ap.add_argument(
        "-normalize_rewards_per_state",
        "--normalize_rewards_per_state",
        action="store_true",
    )

    ap.add_argument(
        "-negative_gradient_type",
        "--negative_gradient_type",
        type=str,
        default="no_negative_gradient",
        choices=[
            "no_negative_gradient",
            "worst_action",
            "average_over_worse_actions",
            "sum_over_worse_actions",
            "average_over_worse_actions_constrained",
            "sum_over_worse_actions_constrained",
            "sum_over_log_compliment_prob",
        ]
    )

    ap.add_argument(
        "-action_type",
        "--action_type",
        type=str,
        default="continuous",
        choices=["continuous", "discrete"],
    )

    ap.add_argument(
        "-num_actions",
        "--num_actions",
        type=int,
        default=100,
    )

    ap.add_argument(
        "-num_action_dim",
        "--num_action_dim",
        type=int,
        default=10,
    )

    ap.add_argument(
        "-num_iterations_on_same_samples",
        "--num_iterations_on_same_samples",
        type=int,
        default=None,
    )

    ap.add_argument(
        "-reward_threshold",
        "--reward_threshold",
        type=float,
        default=None,
    )

    ap.add_argument(
        "-max_new_tokens",
        "--max_new_tokens",
        type=int,
        default=9,
    )

    ap.add_argument(
        "-vocab_size",
        "--vocab_size",
        type=int,
        default=100,
    )

    ap.add_argument(
        "-model_type",
        "--model_type",
        type=str,
        default="gpt-nano",
    )

    ap.add_argument(
        "-cliprange",
        "--cliprange",
        type=float,
        default=0.2,
    )

    ap.add_argument(
        "-num_batches",
        "--num_batches",
        type=int,
        default=1,
    )

    script_arguments = vars(ap.parse_args())
    return script_arguments