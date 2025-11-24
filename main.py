# main.py
import argparse
import os
import sys
import time
from collections import deque

import gymnasium as gym
import numpy as np
import pygame
import torch

from models.ppo_agent import PPOAgent, ACTIONS, N_ACTIONS

# Key bindings (AZERTY)
STEER_LEFT = pygame.K_q
STEER_RIGHT = pygame.K_d
ACCELERATE = pygame.K_z
BRAKE = pygame.K_s
QUIT = pygame.K_ESCAPE

MODEL_PATH = "ppo_carracing.pth"


def get_discrete_action(keys):
    left = keys[STEER_LEFT]
    right = keys[STEER_RIGHT]
    gas = keys[ACCELERATE]
    brake = keys[BRAKE]

    if left and gas:
        return 5
    if right and gas:
        return 6
    if left and brake:
        return 7
    if right and brake:
        return 8
    if left:
        return 1
    if right:
        return 2
    if gas:
        return 3
    if brake:
        return 4
    return 0


def play_with_keyboard():
    pygame.init()
    pygame.display.set_caption("CarRacing Discrete Controls (ZQSD)")

    env = gym.make("CarRacing-v2", render_mode="human")
    obs, info = env.reset(seed=0)

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action_id = get_discrete_action(keys)
        action = ACTIONS[action_id]

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    pygame.quit()
    return


def play_agent(model_path: str, render_mode="human", deterministic=True, sleep=0.0):
    env = gym.make("CarRacing-v2", render_mode=render_mode)
    obs, info = env.reset()
    # observation shape (H,W,C); ppo_agent expects (C,H,W)
    obs_shape = (3, obs.shape[0], obs.shape[1])

    agent = PPOAgent(obs_shape=obs_shape)
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train first.")
        return
    agent.load(model_path)
    print("Model loaded. Playing...")

    done = False
    total_reward = 0.0
    while True:
        # convert observation to CHW float32
        obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
        action_id, _, _ = agent.act(obs_chw)
        action = ACTIONS[action_id]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if sleep:
            time.sleep(sleep)
        if terminated or truncated:
            print("Episode finished, total_reward:", total_reward)
            obs, info = env.reset()
            total_reward = 0.0


def train_ppo(
    total_timesteps: int = 200_000,
    rollout_steps: int = 2048,
    update_epochs: int = 8,
    minibatch_size: int = 64,
    save_path: str = MODEL_PATH,
):
    """
    Training loop (single env). Collect rollouts of length `rollout_steps`,
    compute GAE, then call PPO.update().
    """
    env = gym.make("CarRacing-v2", render_mode=None)  # no rendering during training
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    agent = PPOAgent(obs_shape=obs_shape)

    device = agent.device
    print("Training on device:", device)

    # storage
    obs_buffer = []
    actions_buffer = []
    logprobs_buffer = []
    rewards_buffer = []
    dones_buffer = []
    values_buffer = []

    timestep = 0
    episode_rewards = deque(maxlen=100)
    ep_reward = 0.0
    ep_len = 0
    start_time = time.time()

    while timestep < total_timesteps:
        # collect rollout_steps
        obs, _ = env.reset() if timestep == 0 else (obs, info)
        for step in range(rollout_steps):
            # transform obs to CHW
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            action_id, logp, value = agent.act(obs_chw)

            next_obs, reward, terminated, truncated, info = env.step(ACTIONS[action_id])
            done = bool(terminated or truncated)

            obs_buffer.append(obs_chw)
            actions_buffer.append(action_id)
            logprobs_buffer.append(logp)
            rewards_buffer.append(float(reward))
            dones_buffer.append(float(done))
            values_buffer.append(float(value))

            ep_reward += reward
            ep_len += 1
            timestep += 1

            obs = next_obs

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_len = 0
                obs, info = env.reset()

            if timestep >= total_timesteps:
                break

        # get last value bootstrap
        # convert last observation to CHW and get value estimate
        obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
        with torch.no_grad():
            net = agent.net
            obs_t = torch.tensor(obs_chw, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_value = net(obs_t)
            last_value = float(last_value.item())

        # compute advantages and returns
        rewards_np = np.array(rewards_buffer, dtype=np.float32)
        dones_np = np.array(dones_buffer, dtype=np.float32)
        values_np = np.array(values_buffer + [last_value], dtype=np.float32)  # values[t] and last bootstrap

        advantages, returns = agent.compute_gae(rewards_np, dones_np, values_np[:-1], last_value)

        # flatten and call update
        batch_obs = np.array(obs_buffer, dtype=np.float32)
        batch_actions = np.array(actions_buffer, dtype=np.int64)
        batch_logprobs = np.array(logprobs_buffer, dtype=np.float32)
        batch_returns = returns
        batch_advantages = advantages

        agent.update(
            batch_obs=batch_obs,
            batch_actions=batch_actions,
            batch_logprobs=batch_logprobs,
            batch_returns=batch_returns,
            batch_advantages=batch_advantages,
            epochs=update_epochs,
            minibatch_size=minibatch_size,
        )

        # clear buffers
        obs_buffer.clear()
        actions_buffer.clear()
        logprobs_buffer.clear()
        rewards_buffer.clear()
        dones_buffer.clear()
        values_buffer.clear()

        # save model
        agent.save(save_path)

        # logging
        elapsed = time.time() - start_time
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        print(f"Timestep: {timestep}/{total_timesteps} | Elapsed: {int(elapsed)}s | AvgReward(100): {avg_reward:.2f}")

    env.close()
    print("Training complete. Model saved to", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Entrainer le PPO")
    parser.add_argument("--play", action="store_true", help="Charger le modèle et jouer (render)")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Chemin du fichier modèle")
    args = parser.parse_args()

    if args.train:
        # Ex : 200k timesteps par défaut. Ajuste selon ta machine.
        train_ppo(total_timesteps=200_000, rollout_steps=2048, update_epochs=8, minibatch_size=64, save_path=args.model)
    elif args.play:
        play_agent(args.model, render_mode="human", deterministic=True, sleep=0.0)
    else:
        # comportement interactif classique avec Pygame
        play_with_keyboard()
