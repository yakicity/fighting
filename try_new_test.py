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

    env = MyEnv(render_mode="human")
    # model = PPO.load("ppo_learner_model")
    model = PPO.load("ppo_learner_model_5")
    num_episodes = 1

    player_info = ["cpu","random"]
    """メインループ"""
    for episode in range(1,num_episodes+1):
        if episode % 2 == 0:
            meind = 1
        else:
            meind = 2
        state,_ = env.reset() # observerを初期化し、前処理済みの初期状態を返す
        done = False # エピソードの終了フラグ
        reward_per_episode = 0 # 1エピソード当たりの報酬の総和
        t = 0
        while (not done): # エピソードが終了しない間はずっと処理を行う

            t += 1
            action, _ = model.predict(state, deterministic=True)
            if player_info[1] == "random":
                # enemy_action = np.random.uniform(-1,1)
                # enemy_action = np.random.choice([0,1,2,3,4,5,6], 1)[0]
                enemy_action = np.random.choice([2,3,4,6], 1)[0]
            # elif player_info[1] == "cpu":
                # ここにstateのmeとenamyいれかえたstate[0]をもとめる処理角
                # enemy_state = [state[0][0],state[0][1],state[0][4],state[0][5],state[0][2],state[0][3],state[0][0],state[0][8],state[0][7],state[0][10],state[0][9]]
                # enemy_state = np.array(enemy_state, dtype=np.float32)
                # enemy_action = algo_enemy.exploit(enemy_state)[0]

            state, reward, done, _,_ = env.step_eval([action,enemy_action])

            env.render()
            # if t >= 100:
            #     break

    # ④ wandbのrunを終了
    env.close()

