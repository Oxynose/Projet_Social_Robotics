import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStack
from gymnasium.vector import SyncVectorEnv

# --- Discrete actions ---
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
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(nn.Linear(conv_out, 512), nn.ReLU())
        self.policy_logits = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        f = self.conv(x)
        f = self.fc(f)
        return self.policy_logits(f), self.value_head(f).squeeze(-1)


# --- PPO agent ---
class PPOAgent:
    def __init__(self,
                 obs_shape=(4,84,84),
                 n_actions=N_ACTIONS,
                 lr=2.5e-4,
                 gamma=0.99,
                 lam=0.95,
                 clip_coef=0.2,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 num_envs=8,
                 rollout_steps=1024,
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(obs_shape, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps

    # --- act for a single observation ---
    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    # --- compute advantages ---
    def compute_gae(self, rewards, dones, values, last_value):
        advs = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            next_val = values[t+1] if t+1 < len(values) else last_value
            delta = rewards[t] + self.gamma * next_val * nonterminal - values[t]
            advs[t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        returns = advs + values
        return advs, returns

    # --- update PPO ---
    def update(self, batch_obs, batch_actions, batch_logprobs, batch_returns, batch_advantages,
               epochs=4, minibatch_size=64):
        obs = torch.tensor(batch_obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch_returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch_advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        for epoch in range(epochs):
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            for start in range(0, n, minibatch_size):
                mb_idx = idxs[start:start+minibatch_size]
                mb_obs, mb_actions, mb_oldlog, mb_returns, mb_adv = \
                    obs[mb_idx], actions[mb_idx], old_logprobs[mb_idx], returns[mb_idx], advantages[mb_idx]

                logits, values = self.net(mb_obs)
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logprobs - mb_oldlog)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - mb_returns) ** 2).mean()
                loss = policy_loss + self.vf_coef*value_loss - self.ent_coef*entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    # --- save / load model ---
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
