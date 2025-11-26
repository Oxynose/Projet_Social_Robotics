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
from models.gail_agent import GAILAgent

# Key bindings (AZERTY)
STEER_LEFT = pygame.K_q
STEER_RIGHT = pygame.K_d
ACCELERATE = pygame.K_z
BRAKE = pygame.K_s
QUIT_KEY = pygame.K_ESCAPE

MODEL_PATH_PPO = "ppo_carracing.pth"
MODEL_PATH_GAIL = "gail_carracing.pth"

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

# ---------------------------
# PLAY KEYBOARD
# ---------------------------
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
def train_ppo(total_timesteps=100000, rollout_steps=2048, update_epochs=8, minibatch_size=32, save_path=MODEL_PATH_PPO):
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()

    obs_shape = (3, obs.shape[0], obs.shape[1])
    agent = PPOAgent(obs_shape=obs_shape)
    device = agent.device

    # Rollout buffers
    obs_buffer, actions_buffer, logprobs_buffer, rewards_buffer, dones_buffer, values_buffer = [], [], [], [], [], []

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

# ==========================================================
#                      GAIL TRAIN
# ==========================================================
def train_gail(save_path=MODEL_PATH_GAIL, max_demos=30, gail_epochs=50, discriminator_steps=5, max_fake_steps=2048):
    # Collecte des démonstrations
    env = gym.make("CarRacing-v2", render_mode="human")
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    agent = GAILAgent(obs_shape)

    demonstrations_obs, demonstrations_actions, demonstrations_speed = [], [], []

    print("\n===== COLLECTE DES DÉMONSTRATIONS HUMAINES =====")
    pygame.init()
    clock = pygame.time.Clock()
    demos = 0
    ep_obs, ep_act, ep_speed = [], [], []

    while demos < max_demos:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        keys = pygame.key.get_pressed()
        action_id = get_discrete_action(keys)
        action = ACTIONS[action_id]

        obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
        speed = info.get("speed", 0.0)  # vitesse actuelle
        ep_obs.append(obs_chw)
        ep_act.append(action_id)
        ep_speed.append(speed)

        obs, reward, term, trunc, info = env.step(action)
        total_tiles = info.get("all_tiles", 0)
        visited_tiles = info.get("tiles", 0)
        if total_tiles > 0:
            percent = visited_tiles / total_tiles * 100
            print(f"\rProgression de l'épisode : {percent:.1f}%   ", end="")

        if term or trunc:
            print()
            if visited_tiles >= total_tiles:
                demonstrations_obs.extend(ep_obs)
                demonstrations_actions.extend(ep_act)
                demonstrations_speed.extend(ep_speed)
                demos += 1
                print(f"Démonstration {demos}/{max_demos} enregistrée (parcours complet).")
            else:
                print("Épisode incomplet, démonstration ignorée.")
            ep_obs, ep_act, ep_speed = [], [], []
            obs, info = env.reset()

    env.close()
    pygame.quit()

    expert_obs = np.array(demonstrations_obs, dtype=np.float32)
    expert_actions = np.array(demonstrations_actions, dtype=np.int64)
    expert_speed = np.array(demonstrations_speed, dtype=np.float32)

    # ------------ ENTRAÎNEMENT GAIL ------------
    print("\n===== ENTRAÎNEMENT GAIL EN COURS... =====")
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()

    for epoch in range(1, gail_epochs + 1):
        fake_obs, fake_actions, fake_speed, logps, values = [], [], [], [], []

        step_count = 0
        while step_count < max_fake_steps:
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            a, lp, v = agent.policy.act(obs_chw)
            speed = info.get("speed", 0.0)

            fake_obs.append(obs_chw)
            fake_actions.append(a)
            fake_speed.append(speed)
            logps.append(lp)
            values.append(v)

            obs, _, term, trunc, info = env.step(ACTIONS[a])
            step_count += 1

            if term or trunc:
                obs, info = env.reset()

            if step_count % 100 == 0 or step_count == max_fake_steps:
                print(f"\r[Epoch {epoch}] Generating fake trajectories: step {step_count}/{max_fake_steps}", end="")

        print()

        # Conversion en numpy / tensor
        t_obs = torch.tensor(fake_obs, dtype=torch.float32, device=agent.device)
        t_act = torch.tensor(fake_actions, dtype=torch.long, device=agent.device)
        t_speed = torch.tensor(fake_speed, dtype=torch.float32, device=agent.device)
        gail_rewards = agent.compute_gail_reward(t_obs, t_act, speed=t_speed).detach().cpu().numpy()

        logps = np.array(logps)
        values = np.array(values)

        # Mise à jour du discriminateur
        for step in range(discriminator_steps):
            d_loss = agent.update_discriminator(expert_obs, expert_actions, expert_speed,
                                               fake_obs, fake_actions, fake_speed)
            print(f"[Epoch {epoch}] Discriminator step {step+1}/{discriminator_steps}, loss: {d_loss:.4f}")

        # Mise à jour de la politique PPO
        agent.update_policy(obs=fake_obs, actions=fake_actions,
                            logprobs_old=logps, values=values, rewards=gail_rewards)

        # Sauvegarde
        agent.save(save_path)

    env.close()
    print("\nEntraînement GAIL terminé. Modèle sauvegardé dans :", save_path)

# ---------------------------
# GAIL — PLAY
# ---------------------------
def play_gail(model_path=MODEL_PATH_GAIL):
    env = gym.make("CarRacing-v2", render_mode="human")
    obs, info = env.reset()

    obs_shape = (3, obs.shape[0], obs.shape[1])
    agent = GAILAgent(obs_shape)

    if not os.path.exists(model_path):
        print(f"Model '{model_path}' introuvable.")
        return

    agent.load(model_path)
    print("Modèle GAIL chargé. Jeu en cours...")

    total_reward = 0

    while True:
        obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
        action_id, _, _ = agent.policy.act(obs_chw)
        obs, r, term, trunc, info = env.step(ACTIONS[action_id])
        total_reward += r

        if term or trunc:
            print("Épisode terminé – reward =", total_reward)
            total_reward = 0
            obs, info = env.reset()

# ==========================================================
#                          CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo", action="store_true")
    parser.add_argument("--gail", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--demos", type=int, default=20, help="Nombre de démonstrations complètes pour GAIL")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'époques GAIL")

    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        model_path = MODEL_PATH_PPO if args.ppo else MODEL_PATH_GAIL

    # PPO
    if args.ppo and args.train:
        train_ppo(save_path=model_path)
    elif args.ppo and args.play:
        play_ppo(model_path)

    # GAIL
    elif args.gail and args.train:
        train_gail(save_path=model_path, max_demos=args.demos, gail_epochs=args.epochs)
    elif args.gail and args.play:
        play_gail(model_path)

    # Keyboard
    else:
        play_with_keyboard()
