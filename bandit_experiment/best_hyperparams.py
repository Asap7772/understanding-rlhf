def choose_hyperparameters(algorithm, reward_function, hyperparam):
    best_of_n_hyperparams = {
        1: {
            "N": 10,
            "learning_rate": 0.003,
        },
        2: {
            "N": 10,
            "learning_rate": 0.001,
        },
        3: {
            "N": 10,
            "learning_rate": 0.0003,
        },
    }

    dpo_hyperparams = {
        1: {
            "N": 10,
            "learning_rate": 0.0003,
            "beta": 0.01,
        },
        2: {
            "N": 10,
            "learning_rate": 0.001,
            "beta": 0.001,
        },
        3: {
            "N": 10,
            "learning_rate": 0.0003,
            "beta": 0.01,
        },
    }

    reinforce_hyperparams = {
        1: {
            "N": 10,
            "learning_rate": 0.001,
            "beta": 0.0,
        },
        2: {
            "N": 10,
            "learning_rate": 0.001,
            "beta": 0.0,
        },
        3: {
            "N": 10,
            "learning_rate": 0.001,
            "beta": 0.0,
        },
    }

    best_hyperparams = {
        "best_of_n": best_of_n_hyperparams,
        "dpo": dpo_hyperparams,
        "reinforce": reinforce_hyperparams,
    }

    return best_hyperparams[algorithm][reward_function][hyperparam]