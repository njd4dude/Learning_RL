"""The policy network: state in, action probabilities out."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    """MLP policy with one (CartPole) or two (Pixelcopter) hidden layers.

    Layer names match the course notebook's two `Policy` classes, so
    state dicts and pickled models from either are interchangeable with this one.
    """

    def __init__(self, s_size: int, a_size: int, h_size: int, n_hidden_layers: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        if n_hidden_layers == 1:
            self.fc2 = nn.Linear(h_size, a_size)
        elif n_hidden_layers == 2:
            self.fc2 = nn.Linear(h_size, h_size * 2)
            self.fc3 = nn.Linear(h_size * 2, a_size)
        else:
            raise ValueError(f"n_hidden_layers must be 1 or 2, got {n_hidden_layers}")

    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        if hasattr(self, "fc3"):
            x = self.fc3(F.relu(x))
        return F.softmax(x, dim=1)

    def act(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        """Sample an action from the policy; return it and its log-probability.

        Sampling (not argmax) is what makes REINFORCE explore. The notebook's
        first, deliberately broken version used argmax here.
        """
        device = next(self.parameters()).device
        state = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def build_policy(hyperparameters: dict, device: str = "cpu") -> Policy:
    return Policy(
        hyperparameters["state_space"],
        hyperparameters["action_space"],
        hyperparameters["h_size"],
        hyperparameters.get("n_hidden_layers", 1),
    ).to(device)
