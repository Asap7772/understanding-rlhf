import numpy as np
import torch
import math
import wandb

import utils
import dpo_utils
import transformer_metrics as metrics
from bandit_dataset import TransformerDataset
from copy import deepcopy

class TransformerLearner:
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
        num_epochs=100,
        learning_rate=0.01,
    ):
        optimizer = torch.optim.Adam(
            agent.parameters(), 
            lr=learning_rate,
        )
    
        for _ in range(num_epochs):
            prompt = self.env.reset()
            idx = torch.from_numpy(np.array([[prompt]]))
            loss = 0.0
            optimizer.zero_grad()
            
            for i in range(self.env.max_new_tokens):
                logits, _ = agent(idx)
                logits = logits[:, -1, :]
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)

                loss += torch.distributions.kl.kl_divergence(dist, pi_data)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            
            loss.backward()
            optimizer.step()

    def kl_divergence_and_entropy(self, agent, num_samples=1):
        entropies = []
        kl_pi_agent_pi_datas = []
        kl_pi_data_pi_agents = []
        agent.eval()

        for j in range(min(len(self.train_states), num_samples)):
            train_state = torch.from_numpy(np.array([[self.train_states[j]]]))

            with torch.no_grad():
                empirical_entropy = metrics.entropy(
                    model=agent,
                    state=train_state,
                    max_new_tokens=self.env.max_new_tokens,
                )
                if torch.is_tensor(empirical_entropy):
                    empirical_entropy = empirical_entropy.item()
                entropies.append(empirical_entropy)

            
                kl_pi_data_pi_agent = metrics.kl_divergence(
                    model=self.ref_model,
                    ref_model=agent,
                    state=train_state,
                    max_new_tokens=self.env.max_new_tokens,
                )
                if torch.is_tensor(kl_pi_data_pi_agent):
                    kl_pi_data_pi_agent = kl_pi_data_pi_agent.item()
                kl_pi_data_pi_agents.append(kl_pi_data_pi_agent)

                kl_pi_agent_pi_data = metrics.kl_divergence(
                    model=agent,
                    ref_model=self.ref_model,
                    state=train_state,
                    max_new_tokens=self.env.max_new_tokens,
                )
                if torch.is_tensor(kl_pi_agent_pi_data):
                    kl_pi_agent_pi_data = kl_pi_agent_pi_data.item()

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
        normalize,
    ):
        states = []
        actions = []
        rewards = []
        action_sampler.eval()

        with torch.no_grad():
            for index in range(len(given_states)):
                state = given_states[index]
                torch_state = torch.from_numpy(np.array([[state]]))
                state_actions = [
                    action_sampler.sample(torch_state, max_new_tokens=self.env.max_new_tokens).squeeze()
                    for _ in range(N)
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

        assert states.shape[0] == len(given_states) * N
        assert (
            actions.shape[0] == len(given_states) * N 
            and actions.shape[1] == (self.env.max_new_tokens + 1)
        )
        assert rewards.shape[0] == len(given_states) * N

        dataset = TransformerDataset(
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
        beta=None,
        loss_type=None,
        cliprange=None,
    ):
        total_reward = 0.0
        total_loss = 0.0
        total_count = 0.0
        agent.eval()

        dataset = self.gather_data_batch(
            given_states=self.eval_states,
            N=N,
            action_sampler=agent,
            normalize=False,
        )

        with torch.no_grad():
            if algorithm == "dpo":
                total_loss, total_count = dpo_utils.dpo_loss(
                    dataset=dataset,
                    num_states=len(self.eval_states),
                    N=N,
                    pi_ref=self.ref_model,
                    agent=agent,
                    beta=beta,
                    loss_type=loss_type,
                    is_transformer=True,
                )

                total_margin = dpo_utils.dpo_reward_margin(
                    dataset=dataset,
                    num_states=len(self.eval_states),
                    N=N,
                    pi_ref=self.ref_model,
                    agent=agent,
                    is_transformer=True,
                )

                for index in range(len(dataset)):
                    _, _, reward = dataset[index]
                    total_reward += reward

                average_winner_log_prob, average_loser_log_prob = dpo_utils.dpo_action_logprob(
                    dataset=dataset,
                    num_states=len(self.eval_states),
                    N=N,
                    agent=agent,
                    ref_agent=self.ref_model,
                    is_transformer=True,
                )

            else:
                for index in range(len(dataset)):
                    state, action, reward = dataset[index]
                    total_reward += reward
                    total_count += 1

                    if algorithm == "best_of_n":
                        total_loss -= agent.log_prob(action)

                    elif algorithm == "awr":
                        assert beta is not None

                        log_prob = agent.log_prob(action)
                        reward_weight = (reward / beta).exp()
                        total_loss -= log_prob * reward_weight

                    elif algorithm == "reinforce":
                        total_loss -= agent.log_prob(action) * reward

                    elif algorithm == "ppo":
                        agent_log_prob = agent.log_prob(action)
                        ref_log_prob = self.ref_model.log_prob(action)
                        ratio = torch.exp(agent_log_prob - ref_log_prob)

                        loss_1 = -reward * ratio
                        loss_2 = -reward * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
                        loss = torch.max(loss_1, loss_2)
                        total_loss += loss

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
                average_reward, average_loss, average_margin, average_winner_log_prob, average_loser_log_prob
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
        entropy_calculating_samples,
        gradient_norm,
        negative_gradient_type,
        num_iterations_on_same_samples,
        num_batches,
    ):
        self.run_sft(
            agent=agent,
            pi_data=self.pi_data,
        )

        if hasattr(self, 'ref_model'):
            del self.ref_model
        self.ref_model = deepcopy(agent)

        assert negative_gradient_type in [
            "no_negative_gradient",
            "worst_action",
            "average_over_worse_actions",
            "sum_over_worse_actions",
            "average_over_worse_actions_constrained",
            "sum_over_worse_actions_constrained",
            "sum_over_log_compliment_prob",
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
            assert len(self.train_states) % num_batches == 0
            batch_size = len(self.train_states) // num_batches

            dset = self.gather_data_batch(
                N=N,
                given_states=self.train_states,
                action_sampler=agent,
                normalize=not is_best_of_n,
            )

            data_generating_policy = deepcopy(agent)

            while True:
                shuffled_indices = np.random.permutation(len(self.train_states))

                for batch_index in range(num_batches):
                    loss = 0.0
                    count = 0.0
                    agent.train()
                    agent.zero_grad()

                    with torch.no_grad():
                        random_prompt = self.env.reset()
                        kl_pi_1_pi_2 = metrics.kl_divergence(
                            model=agent,
                            ref_model=data_generating_policy,
                            state=torch.from_numpy(np.array([[random_prompt]])),
                            max_new_tokens=self.env.max_new_tokens,
                            num_samples=1000,
                        )
                        if torch.is_tensor(kl_pi_1_pi_2):
                            kl_pi_1_pi_2 = kl_pi_1_pi_2.item()

                    for state_index in shuffled_indices[batch_index * batch_size : (batch_index + 1) * batch_size]:
                        # algorithm = awr
                        if not is_best_of_n:
                            for action_index in range(N):
                                dset_index = state_index * N + action_index
                                _, action, reward = dset[dset_index]

                                log_prob = agent.log_prob(action)
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

                                loss -= agent.log_prob(action)
                                count += 1
                            
                            elif negative_gradient_type == "worst_action":
                                best_action_index = np.argmax(state_rewards)
                                best_dset_index = state_index * N + best_action_index

                                worst_action_index = np.argmin(state_rewards)
                                worst_dset_index = state_index * N + worst_action_index

                                state, action, reward = dset[best_dset_index]
                                loss -= agent.log_prob(action)

                                state, action, reward = dset[worst_dset_index]
                                loss += beta * agent.log_prob(action)
                                count += 1

                            elif negative_gradient_type in [
                                "average_over_worse_actions", 
                                "sum_over_worse_actions",
                                "average_over_worse_actions_constrained",
                                "sum_over_worse_actions_constrained",
                                "sum_over_log_compliment_prob",
                            ]:
                                best_action_index = np.argmax(state_rewards)
                                best_dset_index = state_index * N + best_action_index
                                state, action, _ = dset[best_dset_index]

                                best_action_log_prob = agent.log_prob(action)
                                loss -= best_action_log_prob

                                best_action_prob = best_action_log_prob.exp().item()
                                if negative_gradient_type.endswith("constrained"):
                                    with torch.no_grad():
                                        loss += self.ref_model.log_prob(action)
                                

                                negative_gradient = 0.0
                                for action_index in range(N):
                                    if action_index != best_action_index:
                                        dset_index = state_index * N + action_index
                                        state, action, _ = dset[dset_index]
                                        action_log_prob = agent.log_prob(action)

                                        if negative_gradient_type == "sum_over_log_compliment_prob":
                                            loss -= beta * torch.log(1.0 - action_log_prob.exp())

                                        elif (action_log_prob.exp().item() / (best_action_prob + 1e-24)) > 0.01:
                                            negative_gradient += beta * action_log_prob

                                            if negative_gradient_type.endswith("constrained"):
                                                with torch.no_grad():
                                                    negative_gradient -= beta * self.ref_model.log_prob(action)

                                if negative_gradient_type == "average_over_worse_actions":
                                    negative_gradient = negative_gradient / (N - 1)

                                loss += negative_gradient
                                count += 1

                    if is_best_of_n:
                        assert count == batch_size
                    else:
                        assert count == batch_size * N
                    loss /= count

                    loss.backward()
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), gradient_norm)

                    optimizer.step()

                    average_reward, average_loss = self.evaluate(
                        agent=agent,
                        N=1,
                        beta=beta,
                        algorithm="best_of_n" if is_best_of_n else "awr",
                    )
                    reward_list.append(average_reward)
                    loss_list.append(average_loss)
                    entropy = self.kl_divergence_and_entropy(
                        agent=agent,
                        num_samples=entropy_calculating_samples,
                    )
                    entropy_list.append(entropy)

                    if self.should_integrate_wandb:
                        with torch.no_grad():
                            train_winner_log_prob, train_loser_log_prob = dpo_utils.dpo_action_logprob(
                                dataset=dset,
                                num_states=len(self.train_states),
                                N=N,
                                agent=agent,
                                ref_agent=self.ref_model,
                                is_transformer=True,
                            )
                        stats = {
                            "reward": average_reward,
                            "eval_loss": average_loss,
                            "train_loss": loss.item(),
                            "KL(pi_agent | pi_data)": entropy["KL(pi_agent | pi_data)"],
                            "entropy": entropy["entropy"],
                            "gradient_norm": utils.calculate_norm_gradient(model=agent),
                            "KL(pi_1 | pi_2)": kl_pi_1_pi_2,
                            "E_train[log p(y_w | x)]": train_winner_log_prob,
                            "E_train[log p(y_l | x)]": train_loser_log_prob,
                        }
                        wandb.log(stats)

                num_iterations += 1

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

            num_queries_to_reward_model.append(num_reward_model_queries)
            method = "B-of-N" if is_best_of_n else "AWR"

            utils.print_message(
                f"{method} Epoch: {epoch + 1}, Reward: {average_reward}", 
                verbose
            )

            iteration_list.append(num_iterations)
            gradient_norms.append(utils.calculate_norm_gradient(model=agent))

            del data_generating_policy

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
        entropy_calculating_samples,
        gradient_norm,
        negative_gradient_type,
        num_iterations_on_same_samples,
        beta,
        num_batches,
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
            entropy_calculating_samples=entropy_calculating_samples,
            gradient_norm=gradient_norm,
            negative_gradient_type=negative_gradient_type,
            num_iterations_on_same_samples=num_iterations_on_same_samples,
            num_batches=num_batches,
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
        entropy_calculating_samples,
        gradient_norm,
        num_iterations_on_same_samples,
        num_batches,
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
            entropy_calculating_samples=entropy_calculating_samples,
            gradient_norm=gradient_norm,
            negative_gradient_type="no_negative_gradient",
            num_iterations_on_same_samples=num_iterations_on_same_samples,
            num_batches=num_batches,
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
        entropy_calculating_samples,  
        gradient_norm,  
        num_iterations_on_same_samples,
        threshold,
        return_margins,
        loss_type,
    ):
        self.run_sft(
            agent=agent,
            pi_data=self.pi_data,
        )

        if hasattr(self, 'ref_model'):
            del self.ref_model
        self.ref_model = deepcopy(agent)

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
        average_reward, loss, margin, eval_winner_log_prob, eval_loser_log_prob = self.evaluate(
            agent=agent,
            N=4,
            beta=beta,
            algorithm="dpo",
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
                action_sampler=agent,
                normalize=False,
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
                    pi_ref=self.ref_model,
                    agent=agent,
                    beta=beta,
                    threshold=threshold,
                    loss_type=loss_type,
                    is_transformer=True,
                )
                loss /= num_data
                loss.backward()

                if gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), gradient_norm)

                optimizer.step()

                num_iterations += 1

                average_reward, average_loss, average_margin, eval_winner_log_prob, eval_loser_log_prob = self.evaluate(
                    agent=agent,
                    N=4,
                    beta=beta,
                    algorithm="dpo",
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
                    with torch.no_grad():
                        train_winner_log_prob, train_loser_log_prob = dpo_utils.dpo_action_logprob(
                            dataset=dset,
                            num_states=len(self.train_states),
                            N=N,
                            agent=agent,
                            ref_agent=self.ref_model,
                            is_transformer=True,
                        )
                    stats = {
                        "reward": average_reward,
                        "eval_loss": average_loss,
                        "train_loss": loss.item(),
                        "margin": average_margin,
                        "gradient_step": num_iterations - iteration_list[-1],
                        "KL(pi_agent | pi_data)": entropy["KL(pi_agent | pi_data)"],
                        "entropy": entropy["entropy"],
                        "gradient_norm": utils.calculate_norm_gradient(model=agent),
                        "E_train[log p(y_w | x)]": train_winner_log_prob,
                        "E_train[log p(y_l | x)]": train_loser_log_prob,
                        "E_test[log p(y_w | x)]": eval_winner_log_prob,
                        "E_test[log p(y_l | x)]": eval_loser_log_prob,
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
        entropy_calculating_samples,  
        gradient_norm,  
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
            entropy_calculating_samples=entropy_calculating_samples,  
            gradient_norm=gradient_norm,  
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
        entropy_calculating_samples,  
        gradient_norm,  
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
            entropy_calculating_samples=entropy_calculating_samples,  
            gradient_norm=gradient_norm,  
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
        max_iterations,
        delta,
        cliprange,
        num_epochs,
        verbose,
        beta,
        remove_negative_gradients,
        entropy_calculating_samples,
        gradient_norm,
        normalize_rewards_per_state,
        num_iterations_on_same_samples,
        algorithm,
        num_batches=1,
    ):
        assert algorithm in ["reinforce", "ppo"]
        self.run_sft(
            agent=agent,
            pi_data=self.pi_data,
        )

        if hasattr(self, 'ref_model'):
            del self.ref_model
        self.ref_model = deepcopy(agent)

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
            beta=None,
            algorithm=algorithm,
            cliprange=cliprange,
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

        sum_reward = 0.0
        sum_reward_square = 0.0
        count = 0.0

        for epoch in range(num_epochs):
            prior_loss = float('inf')
            prior_reward = float('inf')
            agent.eval()
            assert len(self.train_states) % num_batches == 0
            batch_size = len(self.train_states) // num_batches

            dset = self.gather_data_batch(
                N=N,
                given_states=self.train_states,
                action_sampler=agent,
                normalize=normalize_rewards_per_state,
            )

            normalized_rewards = []
            for i in range(len(dset)):
                # reward is already normalized
                if normalize_rewards_per_state:
                    _, _, reward = dset[i]
                    normalized_rewards.append(reward)
                    count += 1
                # normalize using running mean
                else:
                    _, _, reward = dset[i]
                    sum_reward += reward
                    sum_reward_square += (reward * reward)
                    count += 1
                    mean_reward = sum_reward / count
                    variance = (sum_reward_square - (sum_reward * sum_reward) / count)/ count

                    normalized_reward = (reward - mean_reward) / (math.sqrt(variance) + 1e-12)
                    normalized_rewards.append(normalized_reward)

            data_generating_policy = deepcopy(agent)

            while True:
                shuffled_indices = np.random.permutation(len(self.train_states))

                for batch_index in range(num_batches):
                    loss = 0.0
                    count = 0.0
                    agent.train()
                    agent.zero_grad()

                    with torch.no_grad():
                        random_prompt = self.env.reset()
                        kl_pi_1_pi_2 = metrics.kl_divergence(
                            model=agent,
                            ref_model=data_generating_policy,
                            state=torch.from_numpy(np.array([[random_prompt]])),
                            max_new_tokens=self.env.max_new_tokens,
                            num_samples=1000,
                        )
                        if torch.is_tensor(kl_pi_1_pi_2):
                            kl_pi_1_pi_2 = kl_pi_1_pi_2.item()

                    for state_index in shuffled_indices[batch_index * batch_size : (batch_index + 1) * batch_size]:
                        for action_index in range(N):
                            dset_index = state_index * N + action_index
                            _, action, _ = dset[dset_index]

                            reward = normalized_rewards[dset_index]
                            if remove_negative_gradients:
                                reward = reward - np.min(normalized_rewards)

                            agent_log_prob = agent.log_prob(action, sum_log_probs=False)
                            with torch.no_grad():
                                ref_log_prob = self.ref_model.log_prob(action, sum_log_probs=False)
                            kl_penalty = agent_log_prob - ref_log_prob

                            if algorithm == "ppo":
                                with torch.no_grad():
                                    data_generating_policy_logprobs = data_generating_policy.log_prob(
                                        action,
                                        sum_log_probs=False,
                                    )
                                
                                ratio = torch.exp(agent_log_prob - data_generating_policy_logprobs)
                                loss_1 = -reward * ratio
                                loss_2 = -reward * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
                                pg_loss = torch.max(loss_1, loss_2)
                                loss += torch.sum(pg_loss + beta * kl_penalty) 

                            elif algorithm == "reinforce":
                                loss -= torch.sum(agent_log_prob * reward - beta * kl_penalty)

                            else:
                                raise ValueError()

                    loss = loss / (batch_size * N)

                    loss.backward()
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), gradient_norm)

                    optimizer.step()

                    average_reward, average_loss = self.evaluate(
                        agent=agent,
                        N=1,
                        beta=beta,
                        algorithm=algorithm,
                        cliprange=cliprange,
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
                            "KL(pi_agent | pi_data)": entropy["KL(pi_agent | pi_data)"],
                            "entropy": entropy["entropy"],
                            "gradient_norm": utils.calculate_norm_gradient(model=agent),
                            "KL(pi_1 | pi_2)": kl_pi_1_pi_2,
                        }
                        wandb.log(stats)

                num_iterations += 1

                curr_iteration = num_iterations - iteration_list[-1]
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
            iteration_list.append(num_iterations)
            num_queries_to_reward_model.append(num_reward_model_queries)
            gradient_norms.append(utils.calculate_norm_gradient(model=agent))

            utils.print_message(
                f"{algorithm} Epoch: {epoch + 1}, Reward: {average_reward}", 
                verbose
            )

            del data_generating_policy

        return (
            reward_list, 
            iteration_list, 
            entropy_list, 
            num_queries_to_reward_model, 
            gradient_norms, 
            loss_list,
        )
