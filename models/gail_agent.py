# models/gail_agent.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical

from models.ppo_agent import PPOAgent, ACTIONS, N_ACTIONS


# ----------------------------------------------------
# Discriminateur (CNN + MLP)
# ----------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()      # prob(expert)
        )

    def forward(self, obs, action_oh):
        x = obs / 255.0
        features = self.conv(x)
        x = torch.cat([features, action_oh], dim=1)
        return self.fc(x)


# ----------------------------------------------------
# GAIL Agent (Policy = PPO + Discriminator reward)
# ----------------------------------------------------
class GAILAgent:
    def __init__(self, obs_shape, lr_policy=3e-4, lr_disc=3e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Policy PPO
        self.policy = PPOAgent(obs_shape)
        self.policy.net = self.policy.net.to(self.device)

        # Discriminator
        self.disc = Discriminator(obs_shape, N_ACTIONS).to(self.device)
        self.opt_disc = Adam(self.disc.parameters(), lr=lr_disc)

    # -----------------------------
    # Reward from Discriminator
    # -----------------------------
    def compute_gail_reward(self, obs, actions):
        actions_onehot = torch.nn.functional.one_hot(actions, N_ACTIONS).float()
        d_out = self.disc(obs, actions_onehot)
        reward = -torch.log(1 - d_out + 1e-8)
        return reward.squeeze(-1)

    # -----------------------------
    # Update Discriminator
    # -----------------------------
    def update_discriminator(self, expert_obs, expert_actions, policy_obs, policy_actions, epochs=3):
        expert_obs = torch.tensor(expert_obs, dtype=torch.float32, device=self.device)
        expert_actions = torch.tensor(expert_actions, dtype=torch.long, device=self.device)

        policy_obs = torch.tensor(policy_obs, dtype=torch.float32, device=self.device)
        policy_actions = torch.tensor(policy_actions, dtype=torch.long, device=self.device)

        for _ in range(epochs):
            exp_oh = torch.nn.functional.one_hot(expert_actions, N_ACTIONS).float()
            pol_oh = torch.nn.functional.one_hot(policy_actions, N_ACTIONS).float()

            d_exp = self.disc(expert_obs, exp_oh)
            d_pol = self.disc(policy_obs, pol_oh)

            loss = -torch.log(d_exp + 1e-8).mean() - torch.log(1 - d_pol + 1e-8).mean()

            self.opt_disc.zero_grad()
            loss.backward()
            self.opt_disc.step()

            print(f"[GAIL] Discriminator loss: {loss.item():.4f}")

    # -----------------------------
    # Update policy using PPO
    # -----------------------------
    def update_policy(self, obs, actions, logprobs_old, values, rewards):
        loss = self.policy.update(
            batch_obs=obs,
            batch_actions=actions,
            batch_logprobs=logprobs_old,
            batch_returns=rewards,
            batch_advantages=rewards - values,
            epochs=4,
            minibatch_size=64
        )
        print(f"[GAIL] PPO policy loss: {loss:.4f}")

    # -----------------------------
    # Save policy
    # -----------------------------
    def save(self, path):
        dirpath = os.path.dirname(path)
        if dirpath != "":
            os.makedirs(dirpath, exist_ok=True)

        torch.save({
            "policy": self.policy.net.state_dict(),
            "discriminator": self.disc.state_dict()
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.net.load_state_dict(data["policy"])
        self.disc.load_state_dict(data["discriminator"])
