# evaluate_model.py
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from wrappers.wrapper import DiscreteToContinuousWrapper
import numpy as np

MODEL_PATH = "best_models/ppo/best_model.zip"  # ou sac
env = DiscreteToContinuousWrapper(gym.make("CarRacing-v3", render_mode="human"))

# Choisir le modèle : PPO ou SAC
model = PPO.load(MODEL_PATH, env=env)

obs, info = env.reset(seed=0)
done = False
total_reward = 0

while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print("Episode terminé, reward total :", total_reward)
        obs, info = env.reset()
        total_reward = 0
