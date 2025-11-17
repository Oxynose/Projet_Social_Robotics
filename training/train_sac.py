# train_sac.py
import gymnasium as gym
from stable_baselines3 import SAC
from wrappers.wrapper import DiscreteToContinuousWrapper
from callbacks.callbacks import SaveOnBestTrainingRewardCallback

env = DiscreteToContinuousWrapper(gym.make("CarRacing-v3", render_mode="rgb_array"))
save_path = "best_models/sac"

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, save_path=save_path)

model = SAC("CnnPolicy", env, verbose=1, tensorboard_log="./sac_tensorboard/")
model.learn(total_timesteps=500_000, callback=callback)

model.save(f"{save_path}/final_model")
env.close()
