import standard_cell_layout
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
import json
from standard_cell_layout.envs.parseStdcell import get_stdcell_Graph
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, save_path, check_freq, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.current_episode_reward += reward

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            self.episode_count += 1

            # 每check_freq个episodes后计算平均奖励
            if self.episode_count % self.check_freq == 0:
                mean_reward = np.mean(self.episode_rewards[-self.check_freq:])
                if mean_reward >= self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
                    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                    print(f"The better model Q: {mean_reward}")

        return True

cell_name = "SDFQND2"

env = gym.make("standard_cell_layout/StdCellPlaceEnv-v0")

best_save_path = "./model/best_Q/third/ppo_best_model_" + cell_name
last_save_path = "./model/last/third/ppo_last_model_" + cell_name

callback = SaveOnBestTrainingRewardCallback(best_save_path, 1)

# 训练
model = PPO("MlpPolicy", env, verbose=1)
# model = PPO.load(best_save_path, env)
model.learn(total_timesteps=7000000, callback=callback)
model.save(last_save_path)