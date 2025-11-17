# callbacks/callbacks.py
from stable_baselines3.common.callbacks import BaseCallback
import os

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Sauvegarde le modèle si le reward moyen sur 100 épisodes augmente.
    """
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -float('inf')
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # calcule le reward moyen
            x = self.training_env.get_attr('episode_rewards')
            if len(x[0]) >= 100:
                mean_reward = sum(x[0][-100:]) / 100
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    model_path = os.path.join(self.save_path, 'best_model')
                    self.model.save(model_path)
                    if self.verbose > 0:
                        print(f"Nouvelle meilleure moyenne : {mean_reward}, modèle sauvegardé !")
        return True
