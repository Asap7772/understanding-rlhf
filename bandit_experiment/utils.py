import torch.nn as nn
import math 
from torch import distributions as pyd
import torch.nn.functional as F
import torch
import numpy as np
import random


def set_seed_everywhere(seed):
    print(f"Setting seed to: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_dictionary(dictionary, dictionary_name=None):
    if dictionary_name is not None:
        print(f"\nPrinting {dictionary_name}", flush=True)
    for key in dictionary:
        print(f"{key}: {dictionary[key]}", flush=True)
    print("\n", flush=True)


def print_message(message, verbose):
    if verbose:
        print(message, flush=True)


def calculate_norm_gradient(model):
    total_norm = 0.0 
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            
    total_norm = total_norm ** 0.5
    return total_norm


def weight_init(m):
    if isinstance(m, nn.Linear):
        # TODO: Changed initialization to xavier_uniform_
        nn.init.xavier_uniform_(m.weight.data, gain=1e-2)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return y.atanh()

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
    

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class SquashedCauchy(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Cauchy(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class MultidimensionalSquashedCauchy(SquashedCauchy):
    def __init__(self, loc, scale):
        super().__init__(loc=loc, scale=scale)

    def log_prob(self, sample):
        return torch.sum(super().log_prob(sample))
    

class ContinuousSampledCategorical(torch.distributions.Categorical):
    def __init__(
        self,
        continuous_pi_data,
        data_low,
        data_high,
        num_bins,
        num_samples,
    ):
        samples = [continuous_pi_data.sample().item() for i in range(num_samples)]
        hist, _ = np.histogram(
            samples, 
            range=(data_low, data_high), 
            bins=num_bins,
        )
        probs = torch.from_numpy(hist) / num_samples
        self.bin_probs = probs
        super().__init__(probs=probs)