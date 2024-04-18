# Preference Fine-Tuning of LLMs Should Leverage Suboptimal On-Policy Data
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

This is the official codebase of our paper ["Preference Fine-Tuning of LLMs Should Leverage Suboptimal On-Policy Data"]() by [Fahim Tajwar*](https://tajwarfahim.github.io/), [Anikait Singh*](https://asap7772.github.io/), [Archit Sharma](https://architsharma97.github.io/), [Rafael Rafailov](https://rmrafailov.github.io/), [Jeff Schneider](https://www.cs.cmu.edu/~schneide/), [Tengyang Xie](https://tengyangxie.github.io/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/) and [Aviral Kumar](https://aviralkumar2907.github.io/). For any questions/concerns related to this codebase, please reach out to [Fahim Tajwar](mailto:tajwarfahim932@gmail.com).


## Running experiments

For bandit experiments, make sure you are in the `bandit_experiment` directory. The `bandit_experiment/scripts` directory provides example commands to run our experiments.

For UltraFeedback DPO/Pref-FT experiments, `HALOs/project_scripts/run_halos_multi.sh` has the example commands to reproduce the experiments in our paper.

## Additional Datasets

We note the following additional datasets used in our LLM experiments:

1. [Relabelled AlpacaFarm](https://huggingface.co/datasets/Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data)
2. [Min Length](https://huggingface.co/datasets/Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength)
3. [Mode Length](https://huggingface.co/datasets/Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_modelength)
4. [Skew Length](https://huggingface.co/datasets/Asap7772/alpaca_skewexp_minlength_merged)

## Acknowledgements

We acknowledge the following codebases for our paper:

1. [HALOs](https://github.com/ContextualAI)
2. [TRL](https://github.com/huggingface/trl)
3. [DrQ-v2](https://arxiv.org/abs/2107.09645)

We thank the authors for providing us with easy-to-work-with codebases.
