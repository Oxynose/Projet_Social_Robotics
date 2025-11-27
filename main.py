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


# utility: build speed channel (normalized)
def build_speed_channel(speed, H=96, W=96, speed_scale=100.0):
    # normalize
    s = float(speed) / float(speed_scale)
    # clip to [0,1]
    s = max(0.0, min(1.0, s))
    return np.full((1, H, W), s, dtype=np.float32)


def play_with_keyboard():
    pygame.init()
    pygame.display.set_caption("CarRacing Discrete Controls (ZQSD)")

    env = gym.make("CarRacing-v2", render_mode="human")
    obs_raw, info = env.reset(seed=0)

    # obs_raw: H,W,C -> convert to C,H,W
    obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
    speed_channel = build_speed_channel(info.get("speed", 0.0))
    obs = np.concatenate([obs_img, speed_channel], axis=0)

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

        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        next_obs_img = np.transpose(next_obs_raw, (2, 0, 1)).astype(np.float32)
        next_speed_channel = build_speed_channel(info.get("speed", 0.0))
        obs = np.concatenate([next_obs_img, next_speed_channel], axis=0)

        if terminated or truncated:
            next_raw, info = env.reset()
            next_img = np.transpose(next_raw, (2, 0, 1)).astype(np.float32)
            next_speed_channel = build_speed_channel(info.get("speed", 0.0))
            obs = np.concatenate([next_img, next_speed_channel], axis=0)

    env.close()
    pygame.quit()


def play_ppo(model_path):
    env = gym.make("CarRacing-v2", render_mode="human")
    obs_raw, info = env.reset()
    obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
    speed_channel = build_speed_channel(info.get("speed", 0.0))
    obs = np.concatenate([obs_img, speed_channel], axis=0)

    obs_shape = (4, obs.shape[1], obs.shape[2])
    agent = PPOAgent(obs_shape=obs_shape)

    if not os.path.exists(model_path):
        print(f"Model '{model_path}' not found.")
        return

    agent.load(model_path)
    print("PPO model loaded. Playing...")

    total_reward = 0.0

    while True:
        action_id, _, _ = agent.act(obs)
        action = ACTIONS[action_id]

        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        next_img = np.transpose(next_obs_raw, (2, 0, 1)).astype(np.float32)
        next_speed_channel = build_speed_channel(info.get("speed", 0.0))
        obs = np.concatenate([next_img, next_speed_channel], axis=0)

        total_reward += reward
        if terminated or truncated:
            print("Episode finished | reward:", total_reward)
            total_reward = 0.0
            obs_raw, info = env.reset()
            obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
            speed_channel = build_speed_channel(info.get("speed", 0.0))
            obs = np.concatenate([obs_img, speed_channel], axis=0)


def train_ppo(total_timesteps=100000, rollout_steps=2048, update_epochs=8, minibatch_size=32, save_path=MODEL_PATH_PPO):
    env = gym.make("CarRacing-v2", render_mode=None)
    obs_raw, info = env.reset()
    obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
    speed_channel = build_speed_channel(info.get("speed", 0.0))
    obs = np.concatenate([obs_img, speed_channel], axis=0)

    obs_shape = (4, obs.shape[1], obs.shape[2])
    agent = PPOAgent(obs_shape=obs_shape)
    device = agent.device

    obs_buffer, actions_buffer, logprobs_buffer, rewards_buffer, dones_buffer, values_buffer = [], [], [], [], [], []

    timestep = 0
    episode_rewards = deque(maxlen=100)
    ep_reward = 0
    start_time = time.time()

    while timestep < total_timesteps:
        obs_chw = obs  # already C,H,W

        for step in range(rollout_steps):
            action_id, logp, value = agent.act(obs_chw)
            action = ACTIONS[action_id]

            next_raw, reward, terminated, truncated, info = env.step(action)
            next_img = np.transpose(next_raw, (2, 0, 1)).astype(np.float32)
            next_speed_channel = build_speed_channel(info.get("speed", 0.0))
            next_obs = np.concatenate([next_img, next_speed_channel], axis=0)

            done = terminated or truncated

            obs_buffer.append(obs_chw)
            actions_buffer.append(action_id)
            logprobs_buffer.append(logp)
            rewards_buffer.append(reward)
            dones_buffer.append(float(done))
            values_buffer.append(value)

            ep_reward += reward
            timestep += 1

            obs = next_obs
            obs_chw = obs

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0
                obs_raw, info = env.reset()
                obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
                speed_channel = build_speed_channel(info.get("speed", 0.0))
                obs = np.concatenate([obs_img, speed_channel], axis=0)
                obs_chw = obs

            if timestep >= total_timesteps:
                break

        # compute last value for bootstrap
        obs_tensor = torch.tensor(obs_chw, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, last_value = agent.net(obs_tensor)
            last_value = last_value.item()

        rewards_np = np.array(rewards_buffer, dtype=np.float32)
        dones_np = np.array(dones_buffer, dtype=np.float32)
        values_np = np.array(values_buffer + [last_value], dtype=np.float32)

        advantages, returns = agent.compute_gae(rewards_np, dones_np, values_np[:-1], last_value)

        agent.update(
            batch_obs=np.array(obs_buffer, dtype=np.float32),
            batch_actions=np.array(actions_buffer, dtype=np.int64),
            batch_logprobs=np.array(logprobs_buffer, dtype=np.float32),
            batch_returns=returns,
            batch_advantages=advantages,
            epochs=update_epochs,
            minibatch_size=minibatch_size
        )

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


def train_gail(save_path=MODEL_PATH_GAIL, max_demos=20, gail_epochs=50, discriminator_steps=5, max_fake_steps=2048,
               speed_scale=100.0, speed_coef=1.5):
    # collect demonstrations
    env = gym.make("CarRacing-v2", render_mode="human")
    obs_raw, info = env.reset()
    obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
    speed_channel = build_speed_channel(info.get("speed", 0.0), speed_scale=speed_scale)
    obs = np.concatenate([obs_img, speed_channel], axis=0)

    obs_shape = (4, obs.shape[1], obs.shape[2])
    agent = GAILAgent(obs_shape=obs_shape if False else obs_shape)  # keep signature compatible

    demonstrations_obs, demonstrations_actions, demonstrations_speed, demonstrations_ontrack = [], [], [], []

    print("\n===== COLLECTE DES DÉMONSTRATIONS HUMAINES =====")
    pygame.init()
    clock = pygame.time.Clock()
    demos = 0
    ep_obs, ep_act, ep_speed, ep_ontrack = [], [], [], []

    while demos < max_demos:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        keys = pygame.key.get_pressed()
        action_id = get_discrete_action(keys)
        action = ACTIONS[action_id]

        obs_chw = obs  # current obs (C,H,W)
        speed = info.get("speed", 0.0)
        on_track = 1.0 if info.get("tiles", 0) > 0 else 1.0

        ep_obs.append(obs_chw)
        ep_act.append(action_id)
        ep_speed.append(float(speed))
        ep_ontrack.append(float(on_track))

        next_raw, reward, term, trunc, info = env.step(action)
        next_img = np.transpose(next_raw, (2, 0, 1)).astype(np.float32)
        next_speed_channel = build_speed_channel(info.get("speed", 0.0), speed_scale=speed_scale)
        obs = np.concatenate([next_img, next_speed_channel], axis=0)

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
                demonstrations_ontrack.extend(ep_ontrack)
                demos += 1
                print(f"Démonstration {demos}/{max_demos} enregistrée (parcours complet).")
            else:
                print("Épisode incomplet, démonstration ignorée.")
            ep_obs, ep_act, ep_speed, ep_ontrack = [], [], [], []
            obs_raw, info = env.reset()
            obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
            speed_channel = build_speed_channel(info.get("speed", 0.0), speed_scale=speed_scale)
            obs = np.concatenate([obs_img, speed_channel], axis=0)

    env.close()
    pygame.quit()

    expert_obs = np.array(demonstrations_obs, dtype=np.float32)
    expert_actions = np.array(demonstrations_actions, dtype=np.int64)
    expert_speed = np.array(demonstrations_speed, dtype=np.float32)
    expert_ontrack = np.array(demonstrations_ontrack, dtype=np.float32)

    # training loop
    print("\n===== ENTRAÎNEMENT GAIL EN COURS... =====")
    env = gym.make("CarRacing-v2", render_mode=None)
    obs_raw, info = env.reset()
    obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
    speed_channel = build_speed_channel(info.get("speed", 0.0), speed_scale=speed_scale)
    obs = np.concatenate([obs_img, speed_channel], axis=0)

    for epoch in range(1, gail_epochs + 1):
        fake_obs, fake_actions, fake_speed, fake_ontrack, logps, values, dones = [], [], [], [], [], [], []

        step_count = 0
        while step_count < max_fake_steps:
            obs_chw = obs
            a, lp, v = agent.policy.act(obs_chw)
            speed = float(info.get("speed", 0.0))
            on_track = 1.0 if info.get("tiles", 0) > 0 else 1.0

            fake_obs.append(obs_chw)
            fake_actions.append(a)
            fake_speed.append(speed)
            fake_ontrack.append(on_track)
            logps.append(lp)
            values.append(v)

            next_raw, _, term, trunc, info = env.step(ACTIONS[a])
            next_img = np.transpose(next_raw, (2, 0, 1)).astype(np.float32)
            next_speed_channel = build_speed_channel(info.get("speed", 0.0), speed_scale=speed_scale)
            obs = np.concatenate([next_img, next_speed_channel], axis=0)

            done = float(term or trunc)
            dones.append(done)

            step_count += 1
            if term or trunc:
                obs_raw, info = env.reset()
                obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
                speed_channel = build_speed_channel(info.get("speed", 0.0), speed_scale=speed_scale)
                obs = np.concatenate([obs_img, speed_channel], axis=0)

            if step_count % 200 == 0 or step_count == max_fake_steps:
                print(f"\r[Epoch {epoch}] Generated {step_count}/{max_fake_steps} fake steps", end="")

        print()

        fake_obs_np = np.array(fake_obs, dtype=np.float32)
        fake_actions_np = np.array(fake_actions, dtype=np.int64)
        fake_speed_np = np.array(fake_speed, dtype=np.float32)
        fake_ontrack_np = np.array(fake_ontrack, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)

        # compute GAIL rewards
        t_obs = torch.from_numpy(fake_obs_np).to(agent.policy.device)
        t_act = torch.tensor(fake_actions_np, dtype=torch.long, device=agent.policy.device)
        t_speed = torch.tensor(fake_speed_np, dtype=torch.float32, device=agent.policy.device)
        t_ontrack = torch.tensor(fake_ontrack_np, dtype=torch.float32, device=agent.policy.device)

        gail_rewards_t = agent.compute_gail_reward(t_obs, t_act, speed=t_speed, on_track=t_ontrack,
                                                   speed_scale=speed_scale, speed_coef=speed_coef)
        gail_rewards = gail_rewards_t.detach().cpu().numpy()

        # compute last value for bootstrapping
        last_obs_chw = obs
        with torch.no_grad():
            obs_tensor = torch.tensor(last_obs_chw, dtype=torch.float32, device=agent.policy.device).unsqueeze(0)
            _, last_value_t = agent.policy.net(obs_tensor)
            last_value = float(last_value_t.item())

        values_np = np.array(values + [last_value], dtype=np.float32)

        # GAE: compute advantages & returns using PPOAgent.compute_gae
        advantages, returns = agent.policy.compute_gae(gail_rewards, dones_np, values_np[:-1], last_value)

        # update discriminator
        for step in range(discriminator_steps):
            d_loss = agent.update_discriminator(
                expert_obs, expert_actions, expert_speed, expert_ontrack,
                fake_obs_np, fake_actions_np, fake_speed_np, fake_ontrack_np,
                epochs=1
            )
            print(f"[Epoch {epoch}] Discriminator step {step+1}/{discriminator_steps}, loss: {d_loss:.4f}")

        # update policy
        agent.update_policy(
            obs=fake_obs_np,
            actions=fake_actions_np,
            logprobs_old=np.array(logps, dtype=np.float32),
            values=values_np[:-1],
            returns=returns,
            advantages=advantages,
            epochs=4,
            minibatch_size=64
        )

        agent.save(save_path)

    env.close()
    print("\nEntraînement GAIL terminé. Modèle sauvegardé dans :", save_path)


def play_gail(model_path=MODEL_PATH_GAIL):
    env = gym.make("CarRacing-v2", render_mode="human")
    obs_raw, info = env.reset()
    obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
    speed_channel = build_speed_channel(info.get("speed", 0.0))
    obs = np.concatenate([obs_img, speed_channel], axis=0)

    obs_shape = (4, obs.shape[1], obs.shape[2])
    agent = GAILAgent(obs_shape=obs_shape)

    if not os.path.exists(model_path):
        print(f"Model '{model_path}' introuvable.")
        return

    agent.load(model_path)
    print("Modèle GAIL chargé. Jeu en cours...")

    total_reward = 0.0
    while True:
        action_id, _, _ = agent.policy.act(obs)
        obs_raw, r, term, trunc, info = env.step(ACTIONS[action_id])
        obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
        speed_channel = build_speed_channel(info.get("speed", 0.0))
        obs = np.concatenate([obs_img, speed_channel], axis=0)
        total_reward += r
        if term or trunc:
            print("Épisode terminé – reward =", total_reward)
            total_reward = 0.0
            obs_raw, info = env.reset()
            obs_img = np.transpose(obs_raw, (2, 0, 1)).astype(np.float32)
            speed_channel = build_speed_channel(info.get("speed", 0.0))
            obs = np.concatenate([obs_img, speed_channel], axis=0)


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
