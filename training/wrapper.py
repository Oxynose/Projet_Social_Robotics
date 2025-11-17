# wrappers/wrapper.py
import gymnasium as gym
import numpy as np

class DiscreteToContinuousWrapper(gym.ActionWrapper):
    """
    Convertit un espace d'action Discret en action continue pour CarRacing-v3.
    """
    def __init__(self, env):
        super().__init__(env)
        # Actions possibles (id -> vecteur)
        self.actions = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([-1.0, 0.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0]),
            4: np.array([0.0, 0.0, 1.0]),
            5: np.array([-1.0, 1.0, 0.0]),
            6: np.array([1.0, 1.0, 0.0]),
            7: np.array([-1.0, 0.0, 1.0]),
            8: np.array([1.0, 0.0, 1.0])
        }
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action_id):
        return self.actions[action_id]
