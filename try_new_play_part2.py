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
from imitation.algorithms import bc

import wandb
from stable_baselines3 import PPO
from imitation.data.types import Trajectory

from collections import namedtuple
import pickle

if __name__ == "__main__":
    pygame.init()
    env = MyEnv(render_mode="human")
    reconstructed_policy = bc.reconstruct_policy("0_bc_policy/policy_bc_2.pt")
    env = MyEnv(render_mode="human",enemy_model=reconstructed_policy)
    num_episodes = 20

    transitions = []  # Transitionを保存するリスト

    player_info = ["play","random"]
    """メインループ"""
    for episode in range(1,num_episodes+1):
        if episode % 2 == 0:
            meind = 1
        else:
            meind = 2
        state,_ = env.reset() # observerを初期化し、前処理済みの初期状態を返す

        prev_state = state  # 前の状態を保存
        done = False # エピソードの終了フラグ
        reward_per_episode = 0 # 1エピソード当たりの報酬の総和
        t = 0

        actions = []
        infos = []
        dones = []
        observations = [state]

        while (not done): # エピソードが終了しない間はずっと処理を行う

            t += 1
            (next_state, reward, done, _,_),action = env.step_play_transition()
            actions.append(action)
            print(action)
            observations.append(next_state)
            infos.append({})
            dones.append(done)
            env.render()
            prev_state = next_state  # 状態を更新
            # if t >= 100:
            #     break
        ts = Trajectory(obs=np.array(observations), acts=np.array(actions), infos=np.array(infos),terminal=True)
        transitions.append(ts)

    env.close()

    with open("0_expartdata/transitions_final_4.pickle", mode="wb") as f:
        pickle.dump(transitions, f)

    print("Transitions saved successfully!")