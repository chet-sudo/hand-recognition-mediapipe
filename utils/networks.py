"""Small neural network modules for RL agents."""
from __future__ import annotations

import math
from typing import Tuple
import torch
import torch.nn as nn


def init_layer(layer: nn.Linear) -> None:
    """Xavier initialization for linear layers to keep scales reasonable."""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


class MLP(nn.Module):
    """Basic multilayer perceptron for value or policy networks."""

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=(128, 128), activation=nn.Tanh):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            linear = nn.Linear(prev, h)
            init_layer(linear)
            layers += [linear, activation()]
            prev = h
        out = nn.Linear(prev, output_dim)
        init_layer(out)
        layers.append(out)
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous control with state-dependent stddev.

    The network outputs mean actions; the log standard deviation is a learned
    parameter vector with shape (action_dim,) to keep the implementation
    simple and easy to inspect.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.body = MLP(obs_dim, action_dim, hidden_sizes, activation=nn.Tanh)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.body(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(-1)
