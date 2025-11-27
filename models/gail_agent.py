# models/gail_agent.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F

from models.ppo_agent import PPOAgent, ACTIONS, N_ACTIONS


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
            nn.Sigmoid()
        )

    def forward(self, obs, action_oh):
        x = obs / 255.0
        features = self.conv(x)
        x = torch.cat([features, action_oh], dim=1)
        return self.fc(x)


class GAILAgent:
    def __init__(self, obs_shape, lr_policy=3e-4, lr_disc=3e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = PPOAgent(obs_shape)
        self.policy.net = self.policy.net.to(self.device)
        self.disc = Discriminator(obs_shape, N_ACTIONS).to(self.device)
        self.opt_disc = Adam(self.disc.parameters(), lr=lr_disc)

    def compute_gail_reward(self, obs, actions, speed=None, on_track=None, speed_scale=100.0, speed_coef=1.5):
        """
        obs: tensor (B, C, H, W)
        actions: tensor (B,)
        speed: tensor (B,) raw speed values (will be normalized)
        on_track: tensor (B,) mask (1.0 on track, 0 off)
        """
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        actions_onehot = F.one_hot(actions, N_ACTIONS).float().to(self.device)

        d_out = self.disc(obs, actions_onehot)  # (B,1)
        reward = -torch.log(1 - d_out + 1e-8).squeeze(-1)  # (B,)

        if speed is not None:
            speed = speed.to(self.device).float()
            speed_norm = speed / float(speed_scale)
            if on_track is not None:
                on_track = on_track.to(self.device).float()
                speed_norm = speed_norm * on_track
            reward = reward + float(speed_coef) * speed_norm

        return reward  # (B,)

    def update_discriminator(self, expert_obs, expert_actions, expert_speed, expert_ontrack,
                             policy_obs, policy_actions, policy_speed, policy_ontrack, epochs=3):
        """
        Convert numpy arrays to tensors and train discriminator.
        """
        expert_obs_t = torch.tensor(np.asarray(expert_obs), dtype=torch.float32, device=self.device)
        expert_actions_t = torch.tensor(np.asarray(expert_actions), dtype=torch.long, device=self.device)
        policy_obs_t = torch.tensor(np.asarray(policy_obs), dtype=torch.float32, device=self.device)
        policy_actions_t = torch.tensor(np.asarray(policy_actions), dtype=torch.long, device=self.device)

        last_loss = None
        for _ in range(epochs):
            exp_oh = F.one_hot(expert_actions_t, N_ACTIONS).float()
            pol_oh = F.one_hot(policy_actions_t, N_ACTIONS).float()

            d_exp = self.disc(expert_obs_t, exp_oh)
            d_pol = self.disc(policy_obs_t, pol_oh)

            loss = -torch.log(d_exp + 1e-8).mean() - torch.log(1 - d_pol + 1e-8).mean()

            self.opt_disc.zero_grad()
            loss.backward()
            self.opt_disc.step()

            last_loss = loss.item()
            print(f"[GAIL] Discriminator loss: {last_loss:.4f}")

        return last_loss

    def update_policy(self, obs, actions, logprobs_old, values, returns, advantages, epochs=4, minibatch_size=64):
        """
        obs: np.array (B, C, H, W)
        actions: np.array (B,)
        logprobs_old: np.array (B,)
        values: np.array (B,)  (not used directly by PPO.update besides logging)
        returns: np.array (B,)
        advantages: np.array (B,)
        """
        # forward to PPOAgent.update
        loss = self.policy.update(
            batch_obs=np.array(obs, dtype=np.float32),
            batch_actions=np.array(actions, dtype=np.int64),
            batch_logprobs=np.array(logprobs_old, dtype=np.float32),
            batch_returns=np.array(returns, dtype=np.float32),
            batch_advantages=np.array(advantages, dtype=np.float32),
            epochs=epochs,
            minibatch_size=minibatch_size
        )
        print(f"[GAIL] PPO policy loss: {loss:.4f}")
        return loss

    def save(self, path):
        dirpath = os.path.dirname(path)
        if dirpath != "":
            os.makedirs(dirpath, exist_ok=True)
        torch.save({
            "policy": self.policy.net.state_dict(),
            "discriminator": self.disc.state_dict()
        }, path)
        print(f"[GAIL] Modèle sauvegardé dans : {path}")

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.net.load_state_dict(data["policy"])
        self.disc.load_state_dict(data["discriminator"])
        print(f"[GAIL] Modèle chargé depuis : {path}")
