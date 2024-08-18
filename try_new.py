# gymのインポート
import gymnasium as gym
# pandasのインポート
import pandas as pd
# matplotlibのインポート
import matplotlib.pyplot  as plt
import pygame
import sys
from collections import deque
import random
import numpy as np
import gymnasium as gym
import joblib
import math
from env_new_4 import MyEnv
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

import wandb
from stable_baselines3 import PPO


if __name__ == "__main__":

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="fight-project",
        name='ppo_learner_model_2',

        # # track hyperparameters and run metadata
        # config={
        # "epochs": 10,
        # }
    )

    env = MyEnv(render_mode="human")
    # model_path = "ppo_learner_model_3"
    # enemy_model = PPO.load(model_path, env=env)
    # env = MyEnv(render_mode="human",enemy_model= enemy_model)
    # モデルの準備
    model = PPO('MlpPolicy', env, verbose=1)
    # model = PPO.load("ppo_learner_model_3", env=env)
    model.learn(total_timesteps=300000)
    # model.learn(total_timesteps=10000)
    model.save(f"ppo_learner_model_5")
    # model.save(f"ppo_learner_model_second")
    # model.save(f"ppo_learner_model_second_1")

    # ④ wandbのrunを終了
    env.close()
    wandb.finish()
