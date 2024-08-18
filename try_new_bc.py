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

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import pickle
from collections import namedtuple
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.data.types import Trajectory,Transitions
from imitation.util.util import save_policy
if __name__ == "__main__":


# 複数回の試行回数でdemoを追加したい時

    with open("0_expartdata/transitions_final_1.pickle", "rb") as f:
        trajectories_1 = pickle.load(f)
    transitions_1 = rollout.flatten_trajectories(trajectories_1)
    with open("0_expartdata/transitions_final_2.pickle", "rb") as f:
        trajectories_2 = pickle.load(f)
    transitions_2 = rollout.flatten_trajectories(trajectories_2)
    with open("0_expartdata/transitions_final_3.pickle", "rb") as f:
        trajectories_3 = pickle.load(f)
    transitions_3= rollout.flatten_trajectories(trajectories_3)
    with open("transitions_final_4.pickle", "rb") as f:
        trajectories_4 = pickle.load(f)
    transitions_4= rollout.flatten_trajectories(trajectories_4)
    with open("transitions_final_5.pickle", "rb") as f:
        trajectories_5 = pickle.load(f)
    transitions_5= rollout.flatten_trajectories(trajectories_5)
    transitions = transitions_1 + transitions_2 + transitions_3

    print(len(transitions))
    print(transitions[0])
    print(len([t for t in transitions if t['acts'] == 0]))
    print(len([t for t in transitions if t['acts'] == 1]))
    print(len([t for t in transitions if t['acts'] == 2]))
    print(len([t for t in transitions if t['acts'] == 3]))
    print(len([t for t in transitions if t['acts'] == 4]))
    print(len([t for t in transitions if t['acts'] == 5]))
    print(len([t for t in transitions if t['acts'] == 6]))
    print('-====================================')



    filtered_transitions_6 = [t for t in transitions if t['acts'] == 6]

    ideal_len_ts = len(filtered_transitions_6)
    print(ideal_len_ts)
    balanced_transitions_list = []
    for i in range(7):
        now_filtered_transitions = [t for t in transitions if t['acts'] == i]
        if i == 5:
            continue
        # if len(now_filtered_transitions) > ideal_len_ts:
        #     sample_num = min(ideal_len_ts * 2, len(now_filtered_transitions))
        #     now_filtered_transitions = random.sample(now_filtered_transitions, sample_num)
        balanced_transitions_list += now_filtered_transitions

    # フィルタリングで得たサンプル
    print(len([t for t in balanced_transitions_list if t['acts'] == 0]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 1]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 2]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 3]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 4]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 5]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 6]))

    # # 他のアクションとバランスを取るために、一部のNO_OPサンプルをリサンプリング
    # sampled_noop_transitions = random.sample(noop_transitions, ideal_noop_transitions_len)

    # # フィルタリングとリサンプリングしたサンプルを合わせる
    # balanced_transitions_list = filtered_transitions + sampled_noop_transitions

    # print(balanced_transitions_list[0:3])
    # # `balanced_transitions_list`を`Transitions`オブジェクトに変換
    balanced_transitions = Transitions(
        obs=np.array([t["obs"] for t in balanced_transitions_list]),
        acts=np.array([t["acts"] for t in balanced_transitions_list]),
        infos=np.array([t["infos"] for t in balanced_transitions_list], dtype=object),
        next_obs=np.array([t["next_obs"] for t in balanced_transitions_list]),
        dones=np.array([t["dones"] for t in balanced_transitions_list])
        # terminal=True
    )
    # balanced_transitions = rollout.flatten_trajectories(balanced_transitions)

    # print(balanced_transitions)
    print(len(balanced_transitions))


    SEED = 42
    FAST = True
    rng = np.random.default_rng()
    env = MyEnv(render_mode="human")

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=balanced_transitions,
        # demonstrations=transitions,
        rng=rng,
    )
    bc_trainer.train(n_epochs=150)
    save_policy(bc_trainer.policy, "0_bc_policy/policy_bc_2.pt")
    # bc_trainer.policy.save('simple_policy_v0')
    # bc_trainer.save_policy

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
            action, _ = bc_trainer.policy.predict(state, deterministic=True)
            if player_info[1] == "random":
                # enemy_action = np.random.uniform(-1,1)
                # enemy_action = np.random.choice([0,1,2,3,4,5,6], 1)[0]
                enemy_action = np.random.choice([2,3,6], 1)[0]
            # elif player_info[1] == "cpu":
                # ここにstateのmeとenamyいれかえたstate[0]をもとめる処理角
                # enemy_state = [state[0][0],state[0][1],state[0][4],state[0][5],state[0][2],state[0][3],state[0][0],state[0][8],state[0][7],state[0][10],state[0][9]]
                # enemy_state = np.array(enemy_state, dtype=np.float32)
                # enemy_action = algo_enemy.exploit(enemy_state)[0]
            print(action)

            state, reward, done, _,_ = env.step_eval([action.item(),enemy_action])

            env.render()
            # if t >= 100:
            #     break

    # ④ wandbのrunを終了
    env.close()


