import numpy as np
import torch
import math
import wandb

import utils
import dpo_utils
import metrics
from bandit_dataset import BanditDataset
from copy import deepcopy

class BanditLearner:
    def __init__(
        self,
        env,
        pi_data,
        num_train_states,
        num_eval_states,
        should_integrate_wandb=False,
    ):
        self.env = env
        self.pi_data = pi_data

        self.train_states = [env.reset() for _ in range(num_train_states)]
        self.eval_states = [env.reset() for _ in range(num_eval_states)]
        self.should_integrate_wandb = should_integrate_wandb

    def run_sft(
        self,
        agent,
        pi_data,
        is_categorical,
        num_epochs=100,
        learning_rate=0.01,
    ):
        if not is_categorical:
            raise NotImplementedError
        
        optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        for _ in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0
            for state in self.train_states:
                dist = agent.actor(torch.from_numpy(state).float())
                loss += torch.distributions.kl.kl_divergence(dist, pi_data)
            
            loss /= len(self.train_states)
            loss.backward()
            optimizer.step()

    def kl_divergence_and_entropy(self, agent, num_samples=1):
        entropies = []
        kl_pi_agent_pi_datas = []
        kl_pi_data_pi_agents = []
        agent.eval()

        for j in range(min(len(self.train_states), num_samples)):
            train_state = torch.from_numpy(self.train_states[j]).float()

            with torch.no_grad():
                empirical_entropy = metrics.entropy(
                    p=agent, 
                    p_is_agent=True, 
                    state=train_state,
                )
                entropies.append(empirical_entropy)

            
                kl_pi_data_pi_agent = metrics.kl_divergence(
                    p=self.pi_data, 
                    q=agent, 
                    p_is_agent=False,
                    q_is_agent=True,
                    state=train_state,
                )
                kl_pi_data_pi_agents.append(kl_pi_data_pi_agent)

                kl_pi_agent_pi_data = metrics.kl_divergence(
                    p=agent, 
                    q=self.pi_data, 
                    p_is_agent=True,
                    q_is_agent=False,
                    state=train_state,
                )
                kl_pi_agent_pi_datas.append(kl_pi_agent_pi_data)


        return {
            "KL(pi_data | pi_agent)": np.mean(kl_pi_data_pi_agents),
            "KL(pi_agent | pi_data)": np.mean(kl_pi_agent_pi_datas),
            "entropy": np.mean(entropies),
        }
        
    def gather_data_batch(
        self,
        given_states,
        N,
        action_sampler,
        is_agent,
        normalize,
        should_clamp_actions,
        clamp_min,
        clamp_max,
    ):
        states = []
        actions = []
        rewards = []

        if is_agent:
            action_sampler.eval()

        with torch.no_grad():
            for index in range(len(given_states)):
                state = given_states[index]
                clamp_function = lambda x: x if not should_clamp_actions else torch.clamp(x, clamp_min, clamp_max)

                if is_agent:
                    torch_state = torch.from_numpy(state).float()
                    state_actions = [
                        clamp_function(action_sampler.sample(torch_state).float()) for _ in range(N)
                    ]
                else:
                    state_actions = [
                        clamp_function(action_sampler.sample().float()) for _ in range(N)
                    ]

                sample_rewards = [self.env.reward(state, state_actions[i]) for i in range(N)]
                if normalize:
                    sample_rewards_mean = np.mean(sample_rewards)
                    sample_rewards_std = np.std(sample_rewards)
                    sample_rewards = (np.array(sample_rewards) - sample_rewards_mean)
                    sample_rewards = sample_rewards / (sample_rewards_std + 1e-12)
                    sample_rewards = sample_rewards.tolist()

                states.append([state] * N)
                actions += state_actions
                rewards += sample_rewards

        states = np.concatenate(states, axis=0)
        actions = torch.stack(actions)
        rewards = np.array(rewards)

        dataset = BanditDataset(
            states=states,
            actions=actions,
            rewards=rewards,
        )

        return dataset
        
    def evaluate(
        self, 
        agent, 
        algorithm, 
        N, 
        should_clamp_actions,
        clamp_min,
        clamp_max,
        beta=None,
        loss_type=None,
    ):
        total_reward = 0.0
        total_loss = 0.0
        total_count = 0.0
        agent.eval()

        dataset = self.gather_data_batch(
            given_states=self.eval_states,
            N=N,
            action_sampler=agent,
            is_agent=True,
            normalize=False,
            should_clamp_actions=should_clamp_actions,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )

        with torch.no_grad():
            if algorithm == "dpo":
                total_loss, total_count = dpo_utils.dpo_loss(
                    dataset=dataset,
                    num_states=len(self.eval_states),
                    N=N,
                    pi_ref=self.pi_data,
                    agent=agent,
                    beta=beta,
                    loss_type=loss_type,
                )

                total_margin = dpo_utils.dpo_reward_margin(
                    dataset=dataset,
                    num_states=len(self.eval_states),
                    N=N,
                    pi_ref=self.pi_data,
                    agent=agent,
                )

                for index in range(len(dataset)):
                    _, _, reward = dataset[index]
                    total_reward += reward

            else:
                for index in range(len(dataset)):
                    state, action, reward = dataset[index]
                    total_reward += reward
                    total_count += 1

                    if algorithm == "best_of_n":
                        total_loss -= agent.forward(state, action)

                    elif algorithm == "awr":
                        assert beta is not None

                        log_prob = agent.forward(state, action)
                        reward_weight = (reward / beta).exp()
                        total_loss -= log_prob * reward_weight

                    elif algorithm == "reinforce":
                        total_loss -= agent.forward(state, action) * reward

                    elif algorithm != "dpo":
                        raise ValueError(f"Given algorithm {algorithm} not supported.")

        average_reward = total_reward / len(dataset)
        average_loss = total_loss / total_count

        if torch.is_tensor(average_reward):
            average_reward = average_reward.item()

        if torch.is_tensor(average_loss):
            average_loss = average_loss.item()

        if algorithm == "dpo":
            average_margin = total_margin / total_count
            if torch.is_tensor(average_margin):
                average_margin = average_margin.item()

            return (
                average_reward, average_loss, average_margin
            )
        
        return (
            average_reward, 
            average_loss, 
        )

    def _off_policy(
        self,
        agent,
        N,
        learning_rate,
        num_epochs,
        max_iterations,
        beta,
        is_best_of_n,
        delta,
        verbose,
        evaluate_every,
        entropy_calculating_samples,
        gradient_norm,
        negative_gradient_type,
        should_clamp_actions,
        clamp_min,
        clamp_max,
        num_iterations_on_same_samples,
    ):
        if agent.policy_distribution == "categorical":
            self.run_sft(
                agent=agent,
                pi_data=self.pi_data,
                is_categorical=True,
            )

        assert negative_gradient_type in [
            "no_negative_gradient",
            "worst_action",
            "average_over_worse_actions",
            "sum_over_worse_actions",
            "average_over_worse_actions_constrained",
            "sum_over_worse_actions_constrained",
        ]
        if negative_gradient_type !=  "no_negative_gradient":
            assert is_best_of_n
            assert N > 1

        optimizer = torch.optim.Adam(
            agent.parameters(), 
            lr=learning_rate,
        )
        iteration_list = []
        reward_list = []
        entropy_list = []
        loss_list = []
        num_queries_to_reward_model = []
        gradient_norms = [0.0]
        num_iterations = 0
        num_reward_model_queries = 0

        # before any training
        agent.eval()
        average_reward, loss = self.evaluate(
            agent=agent,
            N=1,
            beta=beta,
            algorithm="best_of_n" if is_best_of_n else "awr",
            should_clamp_actions=should_clamp_actions,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        loss_list.append(loss)

        entropy = self.kl_divergence_and_entropy(
            agent=agent,
            num_samples=entropy_calculating_samples,
        )

        iteration_list.append(num_iterations)
        reward_list.append(average_reward)
        entropy_list.append(entropy)
        num_queries_to_reward_model.append(0)

        for epoch in range(num_epochs):
            prior_loss = float('inf')
            prior_reward = float('inf')
            agent.eval()

            dset = self.gather_data_batch(
                N=N,
                given_states=self.train_states,
                action_sampler=self.pi_data if epoch == 0 else agent,
                is_agent=(epoch > 0),
                normalize=False,
                should_clamp_actions=should_clamp_actions,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )

            while True:
                loss = 0.0
                agent.train()
                agent.zero_grad()
                count = 0.0

                for state_index in range(len(self.train_states)):
                    # algorithm = awr
                    if not is_best_of_n:
                        for action_index in range(N):
                            dset_index = state_index * N + action_index
                            state, action, reward = dset[dset_index]

                            log_prob = agent.forward(state, action)
                            reward_weight = (reward / beta).exp()
                            loss -= log_prob * reward_weight
                            count += 1

                    # algorithm = best-of-n
                    else:
                        state_rewards = []
                        for action_index in range(N):
                            dset_index = state_index * N + action_index
                            _, _, reward = dset[dset_index]
                            state_rewards.append(reward.item())

                        if negative_gradient_type == "no_negative_gradient":
                            best_action_index = np.argmax(state_rewards)
                            best_dset_index = state_index * N + best_action_index
                            state, action, reward = dset[best_dset_index]

                            loss -= agent.forward(state, action)
                            count += 1
                        
                        elif negative_gradient_type == "worst_action":
                            best_action_index = np.argmax(state_rewards)
                            best_dset_index = state_index * N + best_action_index

                            worst_action_index = np.argmin(state_rewards)
                            worst_dset_index = state_index * N + worst_action_index

                            state, action, reward = dset[best_dset_index]
                            loss -= agent.forward(state, action)

                            state, action, reward = dset[worst_dset_index]
                            loss += beta * agent.forward(state, action)
                            count += 1

                        elif negative_gradient_type in [
                            "average_over_worse_actions", 
                            "sum_over_worse_actions",
                            "average_over_worse_actions_constrained",
                            "sum_over_worse_actions_constrained",
                        ]:
                            best_action_index = np.argmax(state_rewards)
                            best_dset_index = state_index * N + best_action_index
                            state, action, _ = dset[best_dset_index]

                            loss -= agent.forward(state, action)
                            if negative_gradient_type.endswith("constrained"):
                                loss += self.pi_data.log_prob(action)

                            negative_gradient = 0.0

                            for action_index in range(N):
                                if action_index != best_action_index:
                                    dset_index = state_index * N + action_index
                                    state, action, _ = dset[dset_index]
                                    negative_gradient += beta * agent.forward(state, action)
                                    
                                    if negative_gradient_type.endswith("constrained"):
                                        negative_gradient -= beta * self.pi_data.log_prob(action)

                            if negative_gradient_type == "average_over_worse_actions":
                                negative_gradient = negative_gradient / (N - 1)

                            loss += negative_gradient
                            count += 1

                if is_best_of_n:
                    assert count == len(self.train_states)
                else:
                    assert count == len(dset)
                loss /= count

                loss.backward()
                if gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), gradient_norm)

                optimizer.step()

                num_iterations += 1

                average_reward, average_loss = self.evaluate(
                    agent=agent,
                    N=1,
                    beta=beta,
                    algorithm="best_of_n" if is_best_of_n else "awr",
                    should_clamp_actions=should_clamp_actions,
                    clamp_min=clamp_min,
                    clamp_max=clamp_max,
                )
                reward_list.append(average_reward)
                loss_list.append(average_loss)
                entropy = self.kl_divergence_and_entropy(
                    agent=agent,
                    num_samples=entropy_calculating_samples,
                )
                entropy_list.append(entropy)

                if self.should_integrate_wandb:
                    loc, scale = dpo_utils.get_dpo_agent_properties(
                        states=self.train_states,
                        agent=agent,
                    )
                    stats = {
                        "reward": average_reward,
                        "eval_loss": average_loss,
                        "train_loss": loss.item(),
                        "loc": loc,
                        "scale": scale,
                        "KL(pi_agent | pi_data)": entropy["KL(pi_agent | pi_data)"],
                        "entropy": entropy["entropy"],
                        "gradient_norm": utils.calculate_norm_gradient(model=agent),
                    }
                    wandb.log(stats)

                curr_iteration = num_iterations - iteration_list[-1]
                #utils.print_message(f"Iteration={curr_iteration}, loss={average_loss}", verbose=verbose)
                if num_iterations_on_same_samples is None:
                    if (
                        abs(prior_reward - average_reward) < delta 
                        or curr_iteration >= max_iterations
                    ):
                        break

                elif curr_iteration >= num_iterations_on_same_samples:
                    break

                prior_loss = average_loss
                prior_reward = average_reward

            num_reward_model_queries += len(dset)

            if (epoch + 1) % evaluate_every == 0:
                num_queries_to_reward_model.append(num_reward_model_queries)
                method = "B-of-N" if is_best_of_n else "AWR"

                utils.print_message(
                    f"{method} Epoch: {epoch + 1}, Reward: {average_reward}", 
                    verbose
                )

                iteration_list.append(num_iterations)
                gradient_norms.append(utils.calculate_norm_gradient(model=agent))

        return (
            reward_list, 
            iteration_list, 
            entropy_list, 
            num_queries_to_reward_model, 
            gradient_norms, 
            loss_list,
        )

    def best_of_n(
        self,
        agent,
        N,
        learning_rate,
        num_epochs,
        max_iterations,
        delta,
        verbose,
        beta,
        evaluate_every,
        entropy_calculating_samples,
        gradient_norm,
        negative_gradient_type,
        should_clamp_actions,
        clamp_min,
        clamp_max,
        num_iterations_on_same_samples,
    ):
        return self._off_policy(
            agent=agent,
            N=N,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            max_iterations=max_iterations,
            delta=delta,
            is_best_of_n=True,
            beta=beta,
            verbose=verbose,
            evaluate_every=evaluate_every,
            entropy_calculating_samples=entropy_calculating_samples,
            gradient_norm=gradient_norm,
            negative_gradient_type=negative_gradient_type,
            should_clamp_actions=should_clamp_actions,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            num_iterations_on_same_samples=num_iterations_on_same_samples,
        )
    
    def awr(
        self,
        agent,
        N,
        learning_rate,
        num_epochs,
        max_iterations,
        delta,
        beta,
        verbose,    
        evaluate_every,
        entropy_calculating_samples,
        gradient_norm,
        should_clamp_actions,
        clamp_min,
        clamp_max,
        num_iterations_on_same_samples,
    ):
        return self._off_policy(
            agent=agent,
            N=N,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            max_iterations=max_iterations,
            delta=delta,
            is_best_of_n=False,
            beta=beta,
            verbose=verbose,
            evaluate_every=evaluate_every,
            entropy_calculating_samples=entropy_calculating_samples,
            gradient_norm=gradient_norm,
            negative_gradient_type="no_negative_gradient",
            should_clamp_actions=should_clamp_actions,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            num_iterations_on_same_samples=num_iterations_on_same_samples,
        )
    
    def get_num_reward_model_queries_for_DPO(
        self,
        N,
    ):
        return math.ceil(0.5 * ( 1.0 + math.sqrt(1.0 + (8.0 * N))))
    
    def _negative_gradient_methods(
        self,
        agent,
        N,
        learning_rate,
        num_epochs,
        beta,
        delta,
        max_iterations,
        verbose,
        evaluate_every,
        entropy_calculating_samples,  
        gradient_norm,  
        should_clamp_actions,
        clamp_min,
        clamp_max,
        num_iterations_on_same_samples,
        threshold,
        return_margins,
        loss_type,
    ):
        if agent.policy_distribution == "categorical":
            self.run_sft(
                agent=agent,
                pi_data=self.pi_data,
                is_categorical=True,
            )

        optimizer = torch.optim.Adam(
            agent.parameters(), 
            lr=learning_rate,
        )
        iteration_list = []
        reward_list = []
        loss_list = []
        entropy_list = []
        margin_list = []
        num_queries_to_reward_model = []
        gradient_norms = [0.0]
        num_iterations = 0
        num_reward_model_queries = 0

        # before any training
        agent.eval()
        average_reward, loss, margin = self.evaluate(
            agent=agent,
            N=4,
            beta=beta,
            algorithm="dpo",
            should_clamp_actions=should_clamp_actions,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            loss_type=loss_type,
        )
        loss_list.append(loss)
        margin_list.append(margin)

        entropy = self.kl_divergence_and_entropy(
            agent=agent,
            num_samples=entropy_calculating_samples,
        )

        iteration_list.append(num_iterations)
        reward_list.append(average_reward)
        entropy_list.append(entropy)
        num_queries_to_reward_model.append(0)

        for epoch in range(num_epochs):
            prior_loss = float('inf')
            prior_reward = float('inf')
            agent.eval()

            dset = self.gather_data_batch(
                N=N,
                given_states=self.train_states,
                action_sampler=self.pi_data if epoch == 0 else agent,
                is_agent=(epoch > 0),
                normalize=False,
                should_clamp_actions=should_clamp_actions,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )

            dataset_reward = 0.0
            for i in range(len(dset)):
                _, _, reward = dset[i]
                if torch.is_tensor(reward):
                    reward = reward.item()
                dataset_reward += reward
            
            dataset_reward /= len(dset)
            if self.should_integrate_wandb:
                wandb.log(
                    {"average_reward_of_collected dataset": dataset_reward}
                )

            while True:
                agent.train()
                agent.zero_grad()
                loss, num_data = dpo_utils.dpo_loss(
                    dataset=dset,
                    num_states=len(self.train_states),
                    N=N,
                    pi_ref=self.pi_data,
                    agent=agent,
                    beta=beta,
                    threshold=threshold,
                    loss_type=loss_type,
                )
                loss /= num_data
                loss.backward()

                if gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), gradient_norm)

                optimizer.step()

                num_iterations += 1

                average_reward, average_loss, average_margin = self.evaluate(
                    agent=agent,
                    N=4,
                    beta=beta,
                    algorithm="dpo",
                    should_clamp_actions=should_clamp_actions,
                    clamp_min=clamp_min,
                    clamp_max=clamp_max,
                    loss_type=loss_type,
                )
                reward_list.append(average_reward)
                loss_list.append(average_loss)
                margin_list.append(average_margin)

                entropy = self.kl_divergence_and_entropy(
                    agent=agent,
                    num_samples=entropy_calculating_samples,
                )
                entropy_list.append(entropy)

                if self.should_integrate_wandb:
                    loc, scale = dpo_utils.get_dpo_agent_properties(
                        states=self.train_states,
                        agent=agent,
                    )
                    average_winner_log_prob, average_loser_log_prob = dpo_utils.dpo_action_logprob(
                        dataset=dset,
                        num_states=len(self.train_states),
                        N=N,
                        agent=agent,
                        ref_agent=self.pi_data,
                        is_transformer=False,
                    )
                    stats = {
                        "reward": average_reward,
                        "eval_loss": average_loss,
                        "train_loss": loss.item(),
                        "margin": average_margin,
                        "gradient_step": num_iterations - iteration_list[-1],
                        "loc": loc,
                        "scale": scale,
                        "KL(pi_agent | pi_data)": entropy["KL(pi_agent | pi_data)"],
                        "entropy": entropy["entropy"],
                        "gradient_norm": utils.calculate_norm_gradient(model=agent),
                        "E_train [log p(y_w|x)]": average_winner_log_prob,
                        "E_train [log p(y_l|x)]": average_loser_log_prob,
                    }
                    wandb.log(stats)

                curr_iteration = num_iterations - iteration_list[-1]
                #utils.print_message(f"Iteration={curr_iteration}, loss={average_loss}", verbose=verbose)
                if num_iterations_on_same_samples is None:
                    if (
                        abs(prior_reward - average_reward) < delta 
                        or curr_iteration >= max_iterations
                    ):
                        break

                elif curr_iteration >= num_iterations_on_same_samples:
                    break

                prior_loss = average_loss
                prior_reward = average_reward

            num_reward_model_queries = num_reward_model_queries + (len(self.train_states) * N)

            if (epoch + 1) % evaluate_every == 0:
                num_queries_to_reward_model.append(num_reward_model_queries)
                utils.print_message(
                    f"{loss_type} Epoch: {epoch + 1}, Reward: {average_reward}", 
                    verbose
                )
                iteration_list.append(num_iterations)
                gradient_norms.append(utils.calculate_norm_gradient(model=agent))

        if return_margins:
            return (
                reward_list, 
                iteration_list, 
                entropy_list, 
                num_queries_to_reward_model, 
                gradient_norms, 
                loss_list,
                margin_list,
            )
        else:
            return (
                reward_list, 
                iteration_list, 
                entropy_list, 
                num_queries_to_reward_model, 
                gradient_norms, 
                loss_list,
            )
        
    def dpo(
        self,
        agent,
        N,
        learning_rate,
        num_epochs,
        beta,
        delta,
        max_iterations,
        verbose,
        evaluate_every,
        entropy_calculating_samples,  
        gradient_norm,  
        should_clamp_actions,
        clamp_min,
        clamp_max,
        num_iterations_on_same_samples,
        threshold,
        return_margins,
    ):
        return self._negative_gradient_methods(
            agent=agent,
            N=N,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            beta=beta,
            delta=delta,
            max_iterations=max_iterations,
            verbose=verbose,
            evaluate_every=evaluate_every,
            entropy_calculating_samples=entropy_calculating_samples,  
            gradient_norm=gradient_norm,  
            should_clamp_actions=should_clamp_actions,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            num_iterations_on_same_samples=num_iterations_on_same_samples,
            threshold=threshold,
            return_margins=return_margins,
            loss_type="dpo",
        )

    def ipo(
        self,
        agent,
        N,
        learning_rate,
        num_epochs,
        beta,
        delta,
        max_iterations,
        verbose,
        evaluate_every,
        entropy_calculating_samples,  
        gradient_norm,  
        should_clamp_actions,
        clamp_min,
        clamp_max,
        num_iterations_on_same_samples,
        threshold,
        return_margins,
    ):
        return self._negative_gradient_methods(
            agent=agent,
            N=N,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            beta=beta,
            delta=delta,
            max_iterations=max_iterations,
            verbose=verbose,
            evaluate_every=evaluate_every,
            entropy_calculating_samples=entropy_calculating_samples,  
            gradient_norm=gradient_norm,  
            should_clamp_actions=should_clamp_actions,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            num_iterations_on_same_samples=num_iterations_on_same_samples,
            threshold=threshold,
            return_margins=return_margins,
            loss_type="ipo",
        )

    def on_policy(
        self,
        agent,
        N,
        learning_rate,
        num_epochs,
        verbose,
        beta,
        remove_negative_gradients,
        evaluate_every,
        entropy_calculating_samples,
        gradient_norm,
        normalize_rewards_per_state,
        should_clamp_actions,
        clamp_min,
        clamp_max,
        num_iterations_on_same_samples,
    ):
        if agent.policy_distribution == "categorical":
            self.run_sft(
                agent=agent,
                pi_data=self.pi_data,
                is_categorical=True,
            )

        optimizer = torch.optim.Adam(
            agent.parameters(), 
            lr=learning_rate,
        )

        reward_list = []
        loss_list = []
        iteration_list = []
        num_queries_to_reward_model = []
        entropy_list = []
        gradient_norms = []

        sum_reward = 0.0
        sum_reward_square = 0.0
        count = 0.0

        # stats before any training
        agent.eval()
        average_reward, loss = self.evaluate(
            agent=agent,
            algorithm="reinforce",
            N=1,
            beta=None,
            should_clamp_actions=should_clamp_actions,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        loss_list.append(loss)
        entropy = self.kl_divergence_and_entropy(
            agent=agent,
            num_samples=entropy_calculating_samples,
        )
        iteration_list.append(0)
        reward_list.append(average_reward)
        entropy_list.append(entropy)
        num_queries_to_reward_model.append(0)

        for epoch in range(num_epochs):
            agent.eval()
            dset = self.gather_data_batch(
                N=N,
                given_states=self.train_states,
                action_sampler=self.pi_data if epoch == 0 else agent,
                is_agent=(epoch > 0),
                normalize=normalize_rewards_per_state,
                should_clamp_actions=should_clamp_actions,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )

            agent.train()
            
            for iteration in range(num_iterations_on_same_samples):
                agent.zero_grad()
                loss = 0.0

                normalized_rewards = []
                for i in range(len(dset)):
                    # reward is already normalized
                    if normalize_rewards_per_state:
                        _, _, reward = dset[i]
                        normalized_rewards.append(reward)
                        count += 1
                    else:
                        _, _, reward = dset[i]
                        sum_reward += reward
                        sum_reward_square += (reward * reward)
                        count += 1
                        mean_reward = sum_reward / count
                        variance = (sum_reward_square - (sum_reward * sum_reward) / count)/ count

                        normalized_reward = (reward - mean_reward) / (math.sqrt(variance) + 1e-12)
                        normalized_rewards.append(normalized_reward)

                entropy_term = 0.0
                for i in range(len(dset)):
                    state, action, _ = dset[i]
                    log_prob = agent.forward(state, action)
                    entropy_term -= log_prob

                    normalized_reward = normalized_rewards[i]
                    if remove_negative_gradients:
                        normalized_reward = normalized_reward - np.min(normalized_rewards)

                    loss -= log_prob * normalized_reward

                loss /= len(dset)
                entropy_term /= len(dset)
                if beta != 0.0:
                    loss -= beta * entropy_term
                loss.backward()

                if gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), gradient_norm)

                optimizer.step()
                #utils.print_message(f"Iteration={iteration}", verbose=verbose)

                average_reward, average_loss = self.evaluate(
                    agent=agent,
                    algorithm="reinforce",
                    N=1,
                    beta=None,
                    should_clamp_actions=should_clamp_actions,
                    clamp_min=clamp_min,
                    clamp_max=clamp_max,
                )
                reward_list.append(average_reward)
                loss_list.append(average_loss)
                entropy = self.kl_divergence_and_entropy(
                    agent=agent,
                    num_samples=entropy_calculating_samples,
                )
                entropy_list.append(entropy)

                if self.should_integrate_wandb:
                    stats = {
                        "reward": average_reward,
                        "eval_loss": average_loss,
                        "train_loss": loss.item(),
                    }
                    wandb.log(stats)

            if (epoch + 1) % evaluate_every == 0:
                iteration_list.append((epoch + 1) * num_iterations_on_same_samples)
                num_queries_to_reward_model.append(count)
                gradient_norms.append(utils.calculate_norm_gradient(model=agent))

                utils.print_message(
                    f"Reinforce N={N} Positive reward {remove_negative_gradients} Epoch: {epoch + 1}, Reward: {average_reward}", 
                    verbose
                )

        return (
            reward_list, 
            iteration_list, 
            entropy_list, 
            num_queries_to_reward_model, 
            gradient_norms, 
            loss_list,
        )

        
    
        