# models/ppo_agent.py
import os
import math
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam


# --- ACTIONS (même mapping que ton main.py) ---
ACTIONS = {
    0: np.array([0.0, 0.0, 0.0]),   # rien
    1: np.array([-1.0, 0.0, 0.0]),  # gauche
    2: np.array([1.0, 0.0, 0.0]),   # droite
    3: np.array([0.0, 1.0, 0.0]),   # accélérer
    4: np.array([0.0, 0.0, 1.0]),   # freiner
    5: np.array([-1.0, 1.0, 0.0]),  # gauche + accélérer
    6: np.array([1.0, 1.0, 0.0]),   # droite + accélérer
    7: np.array([-1.0, 0.0, 1.0]),  # gauche + frein
    8: np.array([1.0, 0.0, 1.0])    # droite + frein
}
N_ACTIONS = len(ACTIONS)


# --- Actor-Critic network ---
class ActorCritic(nn.Module):
    def __init__(self, obs_shape: Tuple[int], n_actions: int):
        super().__init__()
        # obs: (C, H, W) — now supports C=4 (RGB + speed channel)
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
        )

        self.policy_logits = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = x / 255.0
        features = self.conv(x)
        features = self.fc(features)
        logits = self.policy_logits(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value


# --- PPO Agent ---
class PPOAgent:
    def __init__(
        self,
        obs_shape: Tuple[int],
        n_actions: int = N_ACTIONS,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(obs_shape, n_actions).to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def act(self, obs: np.ndarray):
        """Return action_id (int), logprob (float), value (float)"""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,C,H,W)
        with torch.no_grad():
            logits, value = self.net(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def get_values_and_logprobs(self, obs_batch: torch.Tensor, act_batch: torch.Tensor):
        logits, values = self.net(obs_batch)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(act_batch)
        entropy = dist.entropy().mean()
        return values, logprobs, entropy

    def compute_gae(self, rewards, dones, values, last_value):
        """
        Compute GAE advantages and returns.
        Inputs are numpy arrays:
          rewards: (T,)
          dones: (T,) mask (1.0 if done else 0.0)
          values: (T,) or (T+1,) depending on caller (we expect values length == T)
          last_value: scalar bootstrap value for the value at T
        Returns:
          advantages (np.array shape (T,)), returns (np.array shape (T,))
        """
        T = len(rewards)
        advs = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            next_val = values[t + 1] if (t + 1 < len(values)) else last_value
            delta = rewards[t] + self.gamma * next_val * nonterminal - values[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            advs[t] = lastgaelam
        returns = advs + values
        return advs, returns

    def update(self, batch_obs, batch_actions, batch_logprobs, batch_returns, batch_advantages,
               epochs=4, minibatch_size=64):
        """Perform PPO update on collected rollout buffer."""
        obs = torch.tensor(batch_obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch_returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch_advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        last_loss = 0.0
        for epoch in range(epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)
            for start in range(0, n, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_oldlog = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                values, logprobs, entropy = self.get_values_and_logprobs(mb_obs, mb_actions)
                ratio = torch.exp(logprobs - mb_oldlog)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - mb_returns) ** 2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                last_loss = loss.item()
        return last_loss

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
