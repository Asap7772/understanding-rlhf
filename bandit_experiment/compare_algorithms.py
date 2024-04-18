import argparse
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from colormap import IrisColormap

from bandit_agents import BanditAgent
from bandit_env import (
    ContextualBanditsEnv, 
    UniformContextDistribution,
)
from best_hyperparams import choose_hyperparameters
from reward_functions import choose_reward_function

from bandit_learner import BanditLearner
import utils
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")


def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-reward_function", 
        "--reward_function", 
        type=int, 
        default=1, 
        choices=[1, 2, 3, 4]
    )

    ap.add_argument(
        "-filedir",
        "--filedir",
        type=str,
        default="algorithm_comparisons",
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

    script_arguments = vars(ap.parse_args())
    return script_arguments


def extract_entropy_list(entropy_type, entropy_list):
    return [entropy_list[i][entropy_type] for i in range(len(entropy_list))]
    


def get_training_args(
    default_hyperparams,
    default_agent_dict,
    seed,
):
    training_args = deepcopy(default_hyperparams)
    utils.set_seed_everywhere(seed=seed)
    agent = BanditAgent(**default_agent_dict).float()
    training_args["agent"] = agent
    return training_args


def compare_algorithms(
    filedir,
    default_env_dict,
    default_agent_dict,
    default_learner_dict,
    best_of_n_hyperparams,
    reinforce_hyperparams,
    dpo_hyperparams,
    seed,
):  
    utils.print_dictionary(default_agent_dict, "Agent params")

    utils.set_seed_everywhere(seed=seed)
    bandit_env = ContextualBanditsEnv(**default_env_dict)
    default_learner_dict["env"] = bandit_env
    learner = BanditLearner(**default_learner_dict)

    assert os.path.isdir(filedir)

    fig = plt.figure(figsize=(25, 5))
    colors = IrisColormap().colors
    all_algorithms = ["best_of_n", "dpo", "reinforce"]
    for algorithm_index in range(len(all_algorithms)):
        algorithm = all_algorithms[algorithm_index]
        print(
            f"\nRunning {algorithm}",
            flush=True,
        )

        if algorithm == "best_of_n":
            training_args = get_training_args(
                default_hyperparams=best_of_n_hyperparams,
                default_agent_dict=default_agent_dict,
                seed=seed,
            )

            (
                all_reward_list, 
                iteration_list, 
                entropy_list,
                reward_model_queries,
                gradient_norms,
            ) = learner.best_of_n(**training_args)

        elif algorithm == "dpo":
            training_args = get_training_args(
                default_hyperparams=dpo_hyperparams,
                default_agent_dict=default_agent_dict,
                seed=seed,
            )

            (
                all_reward_list, 
                iteration_list, 
                entropy_list,
                reward_model_queries,
                gradient_norms,
            ) = learner.dpo(**training_args)

        elif algorithm == "reinforce":
            training_args = get_training_args(
                default_hyperparams=reinforce_hyperparams,
                default_agent_dict=default_agent_dict,
                seed=seed,
            )

            (
                all_reward_list, 
                iteration_list, 
                entropy_list,
                reward_model_queries,
                gradient_norms,
            ) = learner.on_policy(**training_args)

        else:
            raise ValueError(
                f"Given algorithm {algorithm} not supported."
            )
        
        reward_list = [all_reward_list[index] for index in iteration_list]

        plt.subplot(1, 3, 1)
        plt.plot(
            [i for i in range(len(all_reward_list))],
            all_reward_list,
            label=f"{algorithm}",
            color=colors[algorithm_index],
        )

        plt.subplot(1, 3, 2)
        plt.plot(
            reward_model_queries,
            reward_list,
            label=f"{algorithm}",
            color=colors[algorithm_index],
        )


        plt.subplot(1, 3, 3)
        plt.plot(
            extract_entropy_list(
                entropy_type="KL(pi_agent | pi_data)",
                entropy_list=entropy_list,
            ),
            all_reward_list,
            label=f"{algorithm}",
            color=colors[algorithm_index],
        )


    plt.subplot(1, 3, 1)
    plt.xlabel("Gradient step", fontsize=20)
    plt.ylabel("Evaluation reward", fontsize=20)

    plt.subplot(1, 3, 2)
    plt.xlabel("Reward model queries", fontsize=20)
    plt.ylabel("Evaluation reward", fontsize=20)

    plt.subplot(1, 3, 3)
    plt.xlabel("KL(pi_agent | pi_data)", fontsize=20)
    plt.ylabel("Evaluation reward", fontsize=20)


    fig.legend(
        [f"{algorithm}" for algorithm in all_algorithms],
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05), 
        fontsize=20, 
        ncol=4,
    )
    
    assert os.path.isdir(filedir)
    savepath = os.path.join(
        filedir,
        f"comparisons.png"
    )

    plt.savefig(savepath, bbox_inches='tight')
    plt.close(fig=fig)

def choose_action_config(args):
    if args["action_type"] == "continuous":
        _action_config = {
            "action_space_type": "continuous",
            "low": -1.0,
            "high": 1.0,
            "num_dimension": args["num_action_dim"],
        }
        _action_shape = [args["num_action_dim"]]
    elif args["action_type"] == "discrete":
        _action_config = {
            "action_space_type": "discrete",
            "num_actions": args["num_actions"],
        }
        _action_shape = [args["num_actions"]]
    else:
        raise ValueError("Given action type not supported.")
    
    return _action_config, _action_shape


def get_learner_dict(args):
    if args["action_type"] == "continuous":
        num_action_dim = args["num_action_dim"]
        pi_data = utils.MultidimensionalSquashedCauchy(
            loc=torch.from_numpy(np.array([-0.7 for i in range(num_action_dim)])), 
            scale=torch.from_numpy(np.array([0.4 for i in range(num_action_dim)])), 
        )
    elif args["action_type"] == "discrete":
        pi_data = utils.ContinuousSampledCategorical(
            continuous_pi_data=utils.SquashedCauchy(
                loc=torch.from_numpy(np.array([-0.7 for i in range(1)])), 
                scale=torch.from_numpy(np.array([0.4 for i in range(1)]))
            ),
            data_low=-1.0,
            data_high=1.0,
            num_bins=args["num_actions"],
            num_samples=10000,
        )
    else:
        raise ValueError("Given action type not supported.")

    _learner_dict = {
        "pi_data": pi_data,
        "num_train_states": 100,
        "num_eval_states": 100,
    }
    return _learner_dict
    

def main():
    args = get_arguments()
    reward_function = choose_reward_function(args=args)

    _action_config, _action_shape = choose_action_config(args=args)

    _context_distribution = UniformContextDistribution(
        low=0.0,
        high=1.0,
        num_dimension=100,
    )

    _observation_config = {
        "observation_space_type": "continuous",
        "num_dimension": 100,
        "low": 0.0,
        "high": 1.0,
        "context_distribution": _context_distribution
    }

    _env_dict = {
        "action_config": _action_config,
        "observation_config": _observation_config,
        "reward_function": reward_function,
    }

    _agent_dict = {
        "repr_dim": 100,
        "action_shape": _action_shape,
        "feature_dim": 50,
        "hidden_dim": 256,
        "log_std_bounds": [-20.0, 2.0],
        "policy_distribution": args["policy"],
    }

    _learner_dict = get_learner_dict(args=args)

    _best_of_n_hyperparams = {
        "N": choose_hyperparameters("best_of_n", args["reward_function"], "N"),
        "learning_rate": choose_hyperparameters("best_of_n", args["reward_function"], "learning_rate"),
        "num_epochs": 100,
        "max_iterations": 100,
        "delta": 0.1,
        "verbose": True,
        "evaluate_every": 1,
        "entropy_calculating_samples": 1,
        "gradient_norm": args["gradient_norm"],
        "negative_gradient_type": args["negative_gradient_type"],
        "should_clamp_actions": (args["action_type"] == "continuous"),
        "clamp_min": -0.99,
        "clamp_max": 0.99,
        "num_iterations_on_same_samples": args["num_iterations_on_same_samples"],
    }

    _reinforce_hyperparams = {
        "N": choose_hyperparameters("reinforce", args["reward_function"], "N"),
        "learning_rate": choose_hyperparameters("reinforce", args["reward_function"], "learning_rate"),
        "num_epochs": 100,
        "verbose": True,
        "beta": choose_hyperparameters("reinforce", args["reward_function"], "beta"),
        "remove_negative_gradients": False,
        "evaluate_every": 1,
        "entropy_calculating_samples": 1,
        "gradient_norm": args["gradient_norm"],
        "normalize_rewards_per_state": args["normalize_rewards_per_state"],
        "should_clamp_actions": (args["action_type"] == "continuous"),
        "clamp_min": -0.99,
        "clamp_max": 0.99,
        "num_iterations_on_same_samples": 1,
    }

    _dpo_hyperparams = {
        "N": choose_hyperparameters("dpo", args["reward_function"], "N"),
        "learning_rate": choose_hyperparameters("dpo", args["reward_function"], "learning_rate"),
        "num_epochs": 100,
        "max_iterations": 100,
        "delta": 0.1,
        "verbose": True,
        "beta": choose_hyperparameters("dpo", args["reward_function"], "beta"),
        "evaluate_every": 1,
        "entropy_calculating_samples": 1,
        "gradient_norm": args["gradient_norm"],
        "should_clamp_actions": (args["action_type"] == "continuous"),
        "clamp_min": -0.99,
        "clamp_max": 0.99,
        "num_iterations_on_same_samples": args["num_iterations_on_same_samples"],
    }

    filedir = os.path.join(
        args["filedir"],
        f"reward_function_{args['reward_function']}_action_type_{args['action_type']}"
    )
    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    compare_algorithms(
        filedir=filedir,
        default_env_dict=_env_dict,
        default_agent_dict=_agent_dict,
        default_learner_dict=_learner_dict,
        best_of_n_hyperparams=_best_of_n_hyperparams,
        reinforce_hyperparams=_reinforce_hyperparams,
        dpo_hyperparams=_dpo_hyperparams,
        seed=args["seed"],
    )


if __name__ == "__main__":
    main()