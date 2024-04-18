import torch


def dpo_decide_winner(dataset, first_index, second_index):
        assert first_index != second_index
        _, _, reward_1 = dataset[first_index]
        _, _, reward_2 = dataset[second_index]

        if reward_1.item() > reward_2.item():
            return first_index, second_index
        else:
            return second_index, first_index
        

def calculate_dpo_logit(
    dataset, 
    first_index, 
    second_index, 
    pi_ref, 
    agent,
    threshold=None,
    is_transformer=False,
):
        
    winner_index, loser_index = dpo_decide_winner(
        dataset=dataset,
        first_index=first_index,
        second_index=second_index,
    )

    winner_state, winner_action, winner_reward = dataset[winner_index]
    loser_state, loser_action, loser_reward = dataset[loser_index]

    if threshold is not None and abs(winner_reward.item() - loser_reward.item()) < threshold:
        return 0.0
    
    if is_transformer:
        winner_action_agent_logprob = agent.log_prob(winner_action)
    else:
        winner_action_agent_logprob = agent.log_prob(winner_state, winner_action)

    with torch.no_grad():
        winner_action_pi_ref_logprob = pi_ref.log_prob(winner_action)

    winner_logit = winner_action_agent_logprob - winner_action_pi_ref_logprob

    if is_transformer:
        loser_action_agent_logprob = agent.log_prob(loser_action)
    else:
        loser_action_agent_logprob = agent.log_prob(loser_state, loser_action)
    
    with torch.no_grad():
        loser_action_pi_ref_logprob = pi_ref.log_prob(loser_action)
    
    loser_logit = loser_action_agent_logprob - loser_action_pi_ref_logprob

    logit = winner_logit - loser_logit
    return logit


@torch.no_grad()
def dpo_action_logprob(
    dataset,
    num_states,
    N,
    agent,
    ref_agent,
    is_transformer=False,
):
    sum_winner_log_prob = 0.0
    winner_count = 0.0

    sum_loser_log_prob = 0.0
    loser_count = 0.0

    for index in range(num_states):
        indices = [(index * N) + i for i in range(N)]
        all_rewards = []
        for k in indices:
            _, _, reward = dataset[k]
            if torch.is_tensor(reward):
                reward = reward.item()
            all_rewards.append((k, reward))

        all_rewards = sorted(all_rewards, key=lambda x: x[1])
        winner_index = all_rewards[-1][0]

        winner_state, winner_action, _ = dataset[winner_index]
        if is_transformer:
            agent_logprob = agent.log_prob(winner_action) - ref_agent.log_prob(winner_action)
        else:
            agent_logprob = agent.log_prob(winner_state, winner_action) - ref_agent.log_prob(winner_action)
            
        sum_winner_log_prob += agent_logprob
        winner_count += 1

        for k in indices:
            if k != winner_index:
                state, action, _ = dataset[k]

                if is_transformer:
                    agent_logprob = agent.log_prob(action) - ref_agent.log_prob(action)
                else:
                    agent_logprob = agent.log_prob(state, action) - ref_agent.log_prob(action)

                sum_loser_log_prob += agent_logprob
                loser_count += 1
    
    average_winner_log_prob = sum_winner_log_prob / winner_count
    average_loser_log_prob = sum_loser_log_prob / loser_count

    if torch.is_tensor(average_winner_log_prob):
        average_winner_log_prob = average_winner_log_prob.item()
    if torch.is_tensor(average_loser_log_prob):
        average_loser_log_prob = average_loser_log_prob.item()

    return average_winner_log_prob, average_loser_log_prob

        
def dpo_loss(
    dataset,
    num_states,
    N,
    pi_ref,
    agent,
    beta,
    threshold=None,
    loss_type="dpo",
    is_transformer=False,
):
    total_loss = 0.0
    num_data = 0.0

    for index in range(num_states):
        indices = [(index * N) + i for i in range(N)]
        all_rewards = []
        for k in indices:
            _, _, reward = dataset[k]
            if torch.is_tensor(reward):
                reward = reward.item()
            all_rewards.append((k, reward))

        all_rewards = sorted(all_rewards, key=lambda x: x[1])
        
        for i in range(N - 1):
            for j in range(i + 1, N):
                first_index = all_rewards[i][0]
                second_index = all_rewards[j][0]

                logit = calculate_dpo_logit(
                    dataset=dataset,
                    first_index=first_index,
                    second_index=second_index,
                    pi_ref=pi_ref,
                    agent=agent,
                    threshold=threshold,
                    is_transformer=is_transformer,
                )
                if torch.is_tensor(logit):
                    if loss_type == "dpo":
                        total_loss -= torch.nn.functional.logsigmoid(logit * beta)

                    elif loss_type == "ipo":
                        total_loss += (logit - beta) ** 2
                        
                    else:
                        raise NotImplementedError
                    
                    num_data += 1

    return total_loss, num_data


def dpo_reward_margin(
    dataset,
    num_states,
    N,
    pi_ref,
    agent,
    is_transformer=False,
):
    total_reward_margin = 0.0
    total_count = 0.0

    with torch.no_grad():
        for index in range(num_states):
            for i in range(N - 1):
                for j in range(i + 1, N):
                    first_index = (index * N) + i
                    second_index = (index * N) + j
                    reward_margin = calculate_dpo_logit(
                        dataset=dataset,
                        first_index=first_index,
                        second_index=second_index,
                        pi_ref=pi_ref,
                        agent=agent,
                        threshold=None,
                        is_transformer=is_transformer,
                    )
                    total_reward_margin += reward_margin
                    total_count += 1
    
    return total_reward_margin / total_count


def get_dpo_agent_properties(
    states,
    agent,
):
    agent.eval()
    if agent.policy_distribution not in [
        "tanh_gaussian", 
        "tanh_cauchy",
        "gaussian", 
        "cauchy",
    ]:
        return None, None

    with torch.no_grad():
        loc = 0.0
        scale = 0.0
        for i in range(len(states)):
            state = torch.from_numpy(states[i]).float()
            mu, std_pred = agent.actor.policy_parameters(state)
            loc += mu
            scale += std_pred

    loc /= len(states)
    scale /= len(states)
    return torch.norm(loc).item(), torch.norm(scale).item()

     