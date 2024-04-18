import numpy as np
import torch


def kl_divergence(p, q, p_is_agent, q_is_agent, state, num_samples=10000):
    with torch.no_grad():
        if p_is_agent and p.policy_distribution == "categorical":
            return torch.distributions.kl.kl_divergence(p.actor(state), q).item()
        elif q_is_agent and q.policy_distribution == "categorical":
            return torch.distributions.kl.kl_divergence(p, q.actor(state)).item()
    
    with torch.no_grad():
        if p_is_agent:
            p.eval()
            samples = [p.sample(state).float() for _ in range(num_samples)]
            samples = [torch.clamp(sample, -0.99, 0.99) for sample in samples]
            log_p = np.array([p.log_prob(state, samples[i]).item() for i in range(num_samples)])
        else:
            samples = [p.sample().float() for _ in range(num_samples)]
            samples = [torch.clamp(sample, -0.99, 0.99) for sample in samples]
            log_p = np.array([p.log_prob(samples[i]).item() for i in range(num_samples)])

        if q_is_agent:
            q.eval()
            log_q = np.array([q.log_prob(state, samples[i]).item() for i in range(num_samples)])
        else:
            log_q = np.array([q.log_prob(samples[i]).item() for i in range(num_samples)])

    log_p = np.ma.array(log_p, mask=np.isnan(log_p)) 
    log_q = np.ma.array(log_q, mask=np.isnan(log_q)) 

    return np.mean(log_p - log_q)


def entropy(p, p_is_agent, state, num_samples=10000):
    if p_is_agent and p.policy_distribution == "categorical":
        return p.actor(state).entropy().item()
    
    with torch.no_grad():
        if p_is_agent:
            p.eval()
            samples = [p.sample(state).float() for _ in range(num_samples)]
            samples = [torch.clamp(sample, -0.99, 0.99) for sample in samples]
            log_p = np.array([p.log_prob(state, samples[i]).item() for i in range(num_samples)])

        else:
            samples = [p.sample().float() for _ in range(num_samples)]
            samples = [torch.clamp(sample, -0.99, 0.99) for sample in samples]
            log_p = np.array([p.log_prob(samples[i]).item() for i in range(num_samples)])

        log_p = np.ma.array(log_p, mask=np.isnan(log_p)) 

    # print(log_p)
    
    return -np.mean(log_p)