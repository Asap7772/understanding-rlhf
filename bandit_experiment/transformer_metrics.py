def entropy(model, state, max_new_tokens, num_samples=100):
    sum_log_probs = 0.0

    for _ in range(num_samples):
        generations = model.sample(state, max_new_tokens=max_new_tokens)
        log_probs = model.log_prob(idx=generations, sum_log_probs=False)
        sum_log_probs += log_probs

    return -sum_log_probs.mean(dim=-1).sum(dim=0, keepdim=True)


def kl_divergence(model, ref_model, state, max_new_tokens, num_samples=100):
    sum_value = 0.0

    for _ in range(num_samples):
        generations = model.sample(state, max_new_tokens=max_new_tokens)
        model_log_probs = model.log_prob(idx=generations, sum_log_probs=False)
        ref_model_log_probs = ref_model.log_prob(idx=generations, sum_log_probs=False)

        sum_value += model_log_probs - ref_model_log_probs
    
    return sum_value.mean(dim=-1).sum(dim=0, keepdim=True)


