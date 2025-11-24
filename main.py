# main.py
import argparse
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import pygame
import torch

from models.ppo_agent import PPOAgent, ACTIONS

# Key bindings (AZERTY)
STEER_LEFT = pygame.K_q
STEER_RIGHT = pygame.K_d
ACCELERATE = pygame.K_z
BRAKE = pygame.K_s
QUIT_KEY = pygame.K_ESCAPE

MODEL_PATH_PPO = "ppo_carracing.pth"


# ------------------------------------------
#  Discrete keyboard → continuous actions
# ------------------------------------------
def get_discrete_action(keys):
    left = keys[STEER_LEFT]
    right = keys[STEER_RIGHT]
    gas = keys[ACCELERATE]
    brake = keys[BRAKE]

    if left and gas: return 5
    if right and gas: return 6
    if left and brake: return 7
    if right and brake: return 8
    if left: return 1
    if right: return 2
    if gas: return 3
    if brake: return 4
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
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == QUIT_KEY):
                running = False

        keys = pygame.key.get_pressed()
        action_id = get_discrete_action(keys)
        action = ACTIONS[action_id]

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    pygame.quit()


# ---------------------------
# PPO — PLAY
# ---------------------------
def play_ppo(model_path):
    env = gym.make("CarRacing-v2", render_mode="human")
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])

    agent = PPOAgent(obs_shape=obs_shape)

    if not os.path.exists(model_path):
        print(f"Model '{model_path}' not found.")
        return

    agent.load(model_path)
    print("PPO model loaded. Playing...")

    total_reward = 0.0

    while True:
        obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
        action_id, _, _ = agent.act(obs_chw)
        action = ACTIONS[action_id]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print("Episode finished | reward:", total_reward)
            total_reward = 0.0
            obs, info = env.reset()


# ---------------------------
# PPO — TRAIN
# ---------------------------
def train_ppo(total_timesteps=200_000, rollout_steps=2048, update_epochs=8, minibatch_size=64, save_path=MODEL_PATH_PPO):

    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()

    obs_shape = (3, obs.shape[0], obs.shape[1])
    agent = PPOAgent(obs_shape=obs_shape)
    device = agent.device

    # Rollout buffers
    obs_buffer = []
    actions_buffer = []
    logprobs_buffer = []
    rewards_buffer = []
    dones_buffer = []
    values_buffer = []

    timestep = 0
    episode_rewards = deque(maxlen=100)
    ep_reward = 0
    start_time = time.time()

    while timestep < total_timesteps:

        obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)

        for step in range(rollout_steps):

            action_id, logp, value = agent.act(obs_chw)
            action = ACTIONS[action_id]

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # store transition
            obs_buffer.append(obs_chw)
            actions_buffer.append(action_id)
            logprobs_buffer.append(logp)
            rewards_buffer.append(reward)
            dones_buffer.append(float(done))
            values_buffer.append(value)

            ep_reward += reward
            timestep += 1

            obs = next_obs
            obs_chw = np.transpose(next_obs, (2, 0, 1)).astype(np.float32)

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0
                obs, info = env.reset()

            if timestep >= total_timesteps:
                break

        # ------------- compute last value -------------
        obs_tensor = torch.tensor(obs_chw, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, last_value = agent.net(obs_tensor)
            last_value = last_value.item()

        rewards_np = np.array(rewards_buffer, dtype=np.float32)
        dones_np = np.array(dones_buffer, dtype=np.float32)
        values_np = np.array(values_buffer + [last_value], dtype=np.float32)

        advantages, returns = agent.compute_gae(rewards_np, dones_np, values_np[:-1], last_value)

        # ------------- PPO update -------------
        agent.update(
            batch_obs=np.array(obs_buffer, dtype=np.float32),
            batch_actions=np.array(actions_buffer, dtype=np.int64),
            batch_logprobs=np.array(logprobs_buffer, dtype=np.float32),
            batch_returns=returns,
            batch_advantages=advantages,
            epochs=update_epochs,
            minibatch_size=minibatch_size
        )

        # clear rollout
        obs_buffer.clear()
        actions_buffer.clear()
        logprobs_buffer.clear()
        rewards_buffer.clear()
        dones_buffer.clear()
        values_buffer.clear()

        agent.save(save_path)

        avg_r = np.mean(episode_rewards) if episode_rewards else 0
        print(f"Timestep {timestep}/{total_timesteps} | AvgReward(100): {avg_r:.2f}")

    env.close()
    print("PPO training complete. Saved to:", save_path)


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--model", type=str, default=None)

    args = parser.parse_args()

    model_path = args.model or MODEL_PATH_PPO

    if args.train:
        train_ppo(save_path=model_path)

    elif args.play:
        play_ppo(model_path)

    else:
        play_with_keyboard()
