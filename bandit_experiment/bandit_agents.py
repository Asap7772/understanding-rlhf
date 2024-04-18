import torch
import torch.nn as nn
import utils
    

class Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sample_action(self, state):
        raise NotImplementedError

    def forward(self, state, action):
        raise NotImplementedError
    

class NonCategoricalActor(torch.nn.Module):
    def __init__(
        self,
        repr_dim,
        action_shape,
        feature_dim,
        hidden_dim,
        log_std_bounds,
        policy_distribution,
    ):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        action_dim = action_shape[0] * 2

        self.policy = nn.Sequential(
            # convert image/state to a normalized vector
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            # policy layers
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(utils.weight_init)

        if policy_distribution not in [
            "tanh_gaussian", 
            "tanh_cauchy",
            "gaussian",
            "cauchy",
        ]:
            raise ValueError(f"Given policy distribution {policy_distribution} not supported.")
        
        self.policy_distribution = policy_distribution

    def forward(self, obs):
        mu, std_pred = self.policy_parameters(obs=obs)
        #print("Mu: ", mu.item(), "Std :", std_pred.item())

        if self.policy_distribution == "tanh_gaussian":
            dist = utils.SquashedNormal(
                loc=mu, 
                scale=std_pred,
            )
        elif self.policy_distribution == "tanh_cauchy":
            dist = utils.SquashedCauchy(
                loc=mu, 
                scale=std_pred,
            )
        elif self.policy_distribution == "cauchy":
            dist = torch.distributions.Cauchy(
                loc=mu, 
                scale=std_pred,
            )
        elif self.policy_distribution == "gaussian":
            dist = torch.distributions.Normal(
                loc=mu,
                scale=std_pred,
            )
        else:
            raise ValueError(
                f"Given policy distribution {self.policy_distribution} not supported."
            )

        return dist
    
    def policy_parameters(self, obs):
        mu, log_std = self.policy(obs).chunk(2, dim=-1)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clip(log_std, log_std_min, log_std_max)
        std_pred = log_std.exp()

        return mu, std_pred
    

class CategoricalActor(torch.nn.Module):
    def __init__(
        self,
        repr_dim,
        action_shape,
        feature_dim,
        hidden_dim,
    ):
        super().__init__()

        action_dim = action_shape[0]
        self.policy = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            # policy layers
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        categorical_logits = self.policy(obs)
        probs = torch.nn.functional.softmax(categorical_logits)
        dist = torch.distributions.Categorical(probs=probs)
        return dist

class BanditAgent(Agent):
    def __init__(
        self,
        repr_dim,
        feature_dim,
        hidden_dim,
        action_shape,
        log_std_bounds,
        policy_distribution,
    ):
        super().__init__()
        self.policy_distribution = policy_distribution

        if policy_distribution in [
            "tanh_gaussian", 
            "tanh_cauchy",
            "gaussian", 
            "cauchy",
        ]:
            self.actor = NonCategoricalActor(
                repr_dim=repr_dim,
                action_shape=action_shape,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                log_std_bounds=log_std_bounds,
                policy_distribution=policy_distribution,
            )

        elif policy_distribution == "categorical":
            self.actor = CategoricalActor(
                repr_dim=repr_dim,
                action_shape=action_shape,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
            )
        
        else:
            raise ValueError(
                f"Given policy distribution {policy_distribution} is not supported."
            )

    def forward(self, state, action):
        return self.log_prob(state=state, action=action)
    
    def sample(self, state):
        dist = self.actor(state)
        return dist.sample()
    
    def log_prob(self, state, action):
        dist = self.actor(state)
        log_probs = dist.log_prob(action)
        return torch.sum(log_probs)
