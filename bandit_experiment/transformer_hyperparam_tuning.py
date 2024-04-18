import os
os.environ['WANDB_START_METHOD'] = 'thread'

import numpy as np
import torch
from matplotlib import pyplot as plt
import wandb

from gpt import GPT
from bandit_env import TransformerEnv
from reward_functions import choose_reward_function
from arguments import get_arguments

from transformer_learner import TransformerLearner
import utils
from copy import deepcopy
import json

import warnings
warnings.filterwarnings("ignore")


def integrate_wandb(args):
    if args["should_integrate_wandb"]:
        algorithm = args["algorithm"]
        lr = args["lr"]
        beta = args["beta"]

        run_name = f"algorithm_{algorithm}_lr_{lr}_beta_{beta}"

        wandb.init(
            name=run_name,
            project=f"Bandit_{args['project_name']}",
        )
        wandb.config.update(args)


def extract_entropy_list(entropy_type, entropy_list):
    return [entropy_list[i][entropy_type] for i in range(len(entropy_list))]


def create_gpt_model(agent_dict):
    model_config = GPT.get_default_config()
    model_config.model_type = agent_dict["model_type"]
    model_config.vocab_size = agent_dict["vocab_size"]
    model_config.block_size = agent_dict["block_size"]
    model = GPT(model_config)
    model.float()

    return model


def run_algorithm(
    filedir,
    learner,
    learning_rates, 
    default_agent_dict,
    default_hyperparams,
    algorithm,
    seed,
):
    assert os.path.isdir(filedir)

    results = {}

    fig = plt.figure(figsize=(25, 15))
    for learning_rate in learning_rates:
        print(
            f"\nRunning {algorithm} with learning rate: {learning_rate}",
            flush=True,
        )
        training_args = deepcopy(default_hyperparams)
        training_args["learning_rate"] = learning_rate

        agent = create_gpt_model(default_agent_dict)
        training_args["agent"] = agent
        utils.set_seed_everywhere(seed=seed)

        if algorithm == "best_of_n":
            (
                all_reward_list, 
                iteration_list, 
                entropy_list,
                reward_model_queries,
                gradient_norms,
                loss_list,
            ) = learner.best_of_n(**training_args)

        elif algorithm == "awr":
            del training_args["negative_gradient_type"]
            (
                all_reward_list, 
                iteration_list, 
                entropy_list,
                reward_model_queries,
                gradient_norms,
                loss_list,
            ) = learner.awr(**training_args)

        elif algorithm == "dpo":
            (
                all_reward_list, 
                iteration_list, 
                entropy_list,
                reward_model_queries,
                gradient_norms,
                loss_list,
                margin_list,
            ) = learner.dpo(**training_args)

        elif algorithm == "ipo":
            (
                all_reward_list, 
                iteration_list, 
                entropy_list,
                reward_model_queries,
                gradient_norms,
                loss_list,
                margin_list,
            ) = learner.ipo(**training_args)

        elif algorithm in ["reinforce", "ppo"]:
            (
                all_reward_list, 
                iteration_list, 
                entropy_list,
                reward_model_queries,
                gradient_norms,
                loss_list,
            ) = learner.on_policy(**training_args)

        else:
            raise ValueError(
                f"Given algorithm {algorithm} not supported."
            )
        
        reward_list = [all_reward_list[index] for index in iteration_list]

        results[learning_rate] = {
            "all_reward_list": all_reward_list,
            "reward_list": reward_list,
            "iteration_list": iteration_list,
            "entropy_list": entropy_list,
            "reward_model_queries": reward_model_queries,
            "gradient_norms": gradient_norms,
            "loss": loss_list,
        }

        if algorithm == "dpo":
            results[learning_rate]["margin"]: margin_list

        plt.subplot(3, 3, 1)
        plt.plot(
            [i for i in range(len(all_reward_list))],
            all_reward_list,
            label=f"lr={learning_rate}",
        )

        plt.subplot(3, 3, 2)
        plt.plot(
            reward_model_queries,
            reward_list,
            label=f"lr={learning_rate}",
        )

        plt.subplot(3, 3, 3)
        plt.plot(
            [i for i in range(len(iteration_list))],
            reward_list,
            label=f"lr={learning_rate}",
        )

        plt.subplot(3, 3, 4)
        plt.plot(
            [i for i in range(len(all_reward_list))],
            extract_entropy_list(
                entropy_type="entropy",
                entropy_list=entropy_list,
            ),
            label=f"lr={learning_rate}",
        )

        plt.subplot(3, 3, 5)
        plt.plot(
            [i for i in range(len(all_reward_list))],
            extract_entropy_list(
                entropy_type="KL(pi_agent | pi_data)",
                entropy_list=entropy_list,
            ),
            label=f"lr={learning_rate}",
        )

        plt.subplot(3, 3, 6)
        plt.plot(
            extract_entropy_list(
                entropy_type="KL(pi_agent | pi_data)",
                entropy_list=entropy_list,
            ),
            all_reward_list,
            label=f"lr={learning_rate}",
        )

        plt.subplot(3, 3, 7)
        plt.plot(
            [i for i in range(len(loss_list))],
            loss_list,
            label=f"lr={learning_rate}",
        )

        plt.subplot(3, 3, 8)
        plt.plot(
            [i for i in range(len(gradient_norms))],
            gradient_norms,
            label=f"lr={learning_rate}",
        )

        if algorithm == "dpo":
            plt.subplot(3, 3, 9)
            plt.plot(
                [i for i in range(len(margin_list))],
                margin_list,
                label=f"lr={learning_rate}",
            )

        else:
            plt.subplot(3, 3, 9)
            plt.plot(
                [i for i in range(len(all_reward_list))],
                extract_entropy_list(
                    entropy_type="KL(pi_data | pi_agent)",
                    entropy_list=entropy_list,
                ),
                label=f"lr={learning_rate}",
            )


    plt.subplot(3, 3, 1)
    plt.xlabel("Gradient step", fontsize=20)
    plt.ylabel("Evaluation reward", fontsize=20)

    plt.subplot(3, 3, 2)
    plt.xlabel("Reward model queries", fontsize=20)
    plt.ylabel("Evaluation reward", fontsize=20)

    plt.subplot(3, 3, 3)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Evaluation reward", fontsize=20)

    plt.subplot(3, 3, 4)
    plt.xlabel("Gradient step", fontsize=20)
    plt.ylabel("H(pi_agent)", fontsize=20)

    plt.subplot(3, 3, 5)
    plt.xlabel("Gradient step", fontsize=20)
    plt.ylabel("KL(pi_agent | pi_data)", fontsize=20)

    plt.subplot(3, 3, 6)
    plt.xlabel("KL(pi_agent | pi_data)", fontsize=20)
    plt.ylabel("Evaluation reward", fontsize=20)

    plt.subplot(3, 3, 7)
    plt.xlabel("Gradient step", fontsize=20)
    plt.ylabel("Eval loss", fontsize=20)

    plt.subplot(3, 3, 8)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Gradient norm", fontsize=20)

    if algorithm == "dpo":
        plt.subplot(3, 3, 9)
        plt.xlabel("Gradient step", fontsize=20)
        plt.ylabel("Margin", fontsize=20)

    else:
        plt.subplot(3, 3, 9)
        plt.xlabel("Gradient step", fontsize=20)
        plt.ylabel("KL(pi_data | pi_agent)", fontsize=20)


    fig.legend(
        [f"lr={learning_rate}" for learning_rate in learning_rates],
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05), 
        fontsize=20, 
        ncol=4,
    )
    
    assert os.path.isdir(filedir)
    gradient_norm = default_hyperparams.get("gradient_norm")
    normalize = default_hyperparams.get("normalize_rewards_per_state", False)
    policy = default_agent_dict.get("policy_distribution")
    N = default_hyperparams.get("N")
    beta = default_hyperparams.get("beta")
    negative_gradient_type = default_hyperparams.get("negative_gradient_type")
    num_iterations_on_same_samples = default_hyperparams.get("num_iterations_on_same_samples")

    file_path_details = [
        f"algorithm={algorithm}",
        f"policy={policy}",
        f"lr={learning_rates[0]}",
        f"N={N}",
        f"beta={beta}",
        f"gradient_norm={gradient_norm}",
        f"normalize={normalize}",
        f"negative_gradient_type={negative_gradient_type}",
        f"num_iterations_{num_iterations_on_same_samples}",
        f"num_train_states_{len(learner.train_states)}",
        f"seed_{seed}",
    ]
    file_path_prefix = "_".join(file_path_details)
    
    plot_savepath = os.path.join(
        filedir,
        f"{file_path_prefix}.png"
    )

    plt.savefig(plot_savepath, bbox_inches='tight')
    plt.close(fig=fig)

    json_savepath = os.path.join(
        filedir,
        f"{file_path_prefix}.json"
    )

    with open(json_savepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4) 


def hyperparameter_tuning(
    args,
    filedir,
    default_env_dict,
    default_agent_dict,
    default_learner_dict,
    best_of_n_hyperparams,
    reinforce_hyperparams,
    dpo_hyperparams,
    ipo_hyperparams,
    algorithm,
    seed,
):  
    utils.print_dictionary(default_agent_dict, "Agent params")

    utils.set_seed_everywhere(seed=seed)
    transformer_env = TransformerEnv(**default_env_dict)
    default_learner_dict["env"] = transformer_env
    learner = TransformerLearner(**default_learner_dict)

    if algorithm == "best_of_n":
        utils.print_dictionary(
            best_of_n_hyperparams, 
            "Best-of-n hyperparams",
        )
        learning_rates = (
            [args["lr"]] if args["lr"] is not None 
            else [3e-5, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
        )

        run_algorithm(
            filedir=filedir,
            learner=learner,
            learning_rates=learning_rates,
            default_agent_dict=default_agent_dict,
            default_hyperparams=best_of_n_hyperparams,
            algorithm=algorithm,
            seed=seed,
        )

    elif algorithm == "awr":
        utils.print_dictionary(
            best_of_n_hyperparams, 
            "AWR hyperparams",
        )
        learning_rates = (
            [args["lr"]] if args["lr"] is not None 
            else [3e-5, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
        )

        run_algorithm(
            filedir=filedir,
            learner=learner,
            learning_rates=learning_rates,
            default_agent_dict=default_agent_dict,
            default_hyperparams=best_of_n_hyperparams,
            algorithm=algorithm,
            seed=seed,
        )

    elif algorithm == "dpo":
        utils.print_dictionary(
            dpo_hyperparams, 
            "DPO hyperparams",
        )

        learning_rates = (
            [args["lr"]] if args["lr"] is not None 
            else [1e-5, 3e-5, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
        )

        run_algorithm(
            filedir=filedir,
            learner=learner,
            learning_rates=learning_rates,
            default_agent_dict=default_agent_dict,
            default_hyperparams=dpo_hyperparams,
            algorithm=algorithm,
            seed=seed,
        )

    elif algorithm == "reinforce":
        utils.print_dictionary(
            reinforce_hyperparams, 
            "Reinforce hyperparams",
        )

        learning_rates = (
            [args["lr"]] if args["lr"] is not None 
            else [1e-5, 3e-5, 0.0001, 0.0003, 0.001, 0.003, 0.01]
        )

        run_algorithm(
            filedir=filedir,
            learner=learner,
            learning_rates=learning_rates,
            default_agent_dict=default_agent_dict,
            default_hyperparams=reinforce_hyperparams,
            algorithm=algorithm,
            seed=seed,
        )

    elif algorithm == "ppo":
        utils.print_dictionary(
            reinforce_hyperparams, 
            "PPO hyperparams",
        )

        learning_rates = (
            [args["lr"]] if args["lr"] is not None 
            else [1e-5, 3e-5, 0.0001, 0.0003, 0.001, 0.003, 0.01]
        )

        run_algorithm(
            filedir=filedir,
            learner=learner,
            learning_rates=learning_rates,
            default_agent_dict=default_agent_dict,
            default_hyperparams=reinforce_hyperparams,
            algorithm=algorithm,
            seed=seed,
        )

    elif algorithm == "ipo":
        utils.print_dictionary(
            ipo_hyperparams, 
            "IPO hyperparams",
        )

        learning_rates = (
            [args["lr"]] if args["lr"] is not None 
            else [1e-5, 3e-5, 0.0001, 0.0003, 0.001, 0.003, 0.01]
        )

        run_algorithm(
            filedir=filedir,
            learner=learner,
            learning_rates=learning_rates,
            default_agent_dict=default_agent_dict,
            default_hyperparams=ipo_hyperparams,
            algorithm=algorithm,
            seed=seed,
        )

    else:
        raise ValueError(f"Given algorithm {algorithm} not supported.")


def get_learner_dict(args):
    pi_data = utils.ContinuousSampledCategorical(
        continuous_pi_data=utils.SquashedCauchy(
            loc=torch.from_numpy(np.array([-0.7 for i in range(1)])), 
            scale=torch.from_numpy(np.array([0.4 for i in range(1)]))
        ),
        data_low=-1.0,
        data_high=1.0,
        num_bins=args["vocab_size"],
        num_samples=10000,
    )

    _learner_dict = {
        "pi_data": pi_data,
        "num_train_states": args["num_train_states"],
        "num_eval_states": args["num_eval_states"],
        "should_integrate_wandb": args["should_integrate_wandb"],
    }

    return _learner_dict

    
def main():
    args = get_arguments()
    integrate_wandb(args=args)
    reward_function = choose_reward_function(args=args)


    _env_dict = {
        "max_new_tokens": args["max_new_tokens"],
        "vocab_size": args["vocab_size"],
        "reward_function": reward_function,
    }

    _agent_dict = {
        "model_type": args["model_type"],
        "vocab_size": args["vocab_size"],
        "block_size": args["max_new_tokens"] + 1,
    }

    _learner_dict = get_learner_dict(args=args)

    _best_of_n_hyperparams = {
        "N": args["N"],
        "learning_rate": 1e-2,
        "num_epochs": args["epochs"],
        "max_iterations": 100,
        "delta": 0.1,
        "verbose": True,
        "beta": args["beta"],
        "entropy_calculating_samples": 1,
        "gradient_norm": args["gradient_norm"],
        "negative_gradient_type": args["negative_gradient_type"],
        "num_iterations_on_same_samples": args["num_iterations_on_same_samples"],
        "num_batches": args["num_batches"],
    }

    _reinforce_hyperparams = {
        "N": args["N"],
        "learning_rate": 1e-2,
        "num_epochs": args["epochs"],
        "max_iterations": 100,
        "delta": 0.0005,
        "verbose": True,
        "beta": args["beta"],
        "remove_negative_gradients": False,
        "entropy_calculating_samples": 1,
        "gradient_norm": args["gradient_norm"],
        "normalize_rewards_per_state": args["normalize_rewards_per_state"],
        "num_iterations_on_same_samples": args["num_iterations_on_same_samples"],
        "cliprange": args["cliprange"],
        "algorithm": args["algorithm"],
        "num_batches": args["num_batches"],
    }

    _dpo_hyperparams = {
        "N": args["N"],
        "learning_rate": 1e-2,
        "num_epochs": args["epochs"],
        "max_iterations": 100,
        "delta": 0.0005,
        "verbose": True,
        "beta": args["beta"],
        "entropy_calculating_samples": 1,
        "gradient_norm": args["gradient_norm"],
        "num_iterations_on_same_samples": args["num_iterations_on_same_samples"],
        "threshold": args["reward_threshold"],
        "return_margins": True,
    }

    _ipo_hyperparams = {
        "N": args["N"],
        "learning_rate": 1e-2,
        "num_epochs": args["epochs"],
        "max_iterations": 100,
        "delta": 0.0005,
        "verbose": True,
        "beta": args["beta"],
        "entropy_calculating_samples": 1,
        "gradient_norm": args["gradient_norm"],
        "num_iterations_on_same_samples": args["num_iterations_on_same_samples"],
        "threshold": args["reward_threshold"],
        "return_margins": True,
    }

    filedir = os.path.join(
        args["filedir"],
        f"reward_function_{args['reward_function']}_action_type_{args['action_type']}"
    )
    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    hyperparameter_tuning(
        args=args,
        filedir=filedir,
        default_env_dict=_env_dict,
        default_agent_dict=_agent_dict,
        default_learner_dict=_learner_dict,
        best_of_n_hyperparams=_best_of_n_hyperparams,
        reinforce_hyperparams=_reinforce_hyperparams,
        dpo_hyperparams=_dpo_hyperparams,
        ipo_hyperparams=_ipo_hyperparams,
        algorithm=args["algorithm"],
        seed=args["seed"],
    )


if __name__ == "__main__":
    main()
    
