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
# ニューラルネットワークモデルのインポート
from sklearn.neural_network import MLPRegressor
# ACAgentクラスの作成
from sklearn.exceptions import NotFittedError
import joblib

from env import MyEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import math
import time
from datetime import timedelta


# Observerクラスの作成
class Observer(object):
    def __init__(self, env): # 初期化メソッド
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def render(self): # 状態などを可視化するメソッド
        self.env.render()

    def reset(self): # 環境を初期化して初期状態を返すメソッド
        return self.preprocess(self.env.reset()[0])

    def step(self, action): # 行動を渡して前処理した状態と報酬などを返すメソッド
        # print(action)
        state, reward, done, _, info = self.env.step(action)
        print(state)
        return self.preprocess(state), reward, done, info
    def preprocess(self, state):
        return state.reshape((1, 11))

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, new_state,done):
        data = (state, action, reward, new_state,done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        idx = np.random.choice(np.arange(len(self.buffer)), size=self.batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

def calculate_log_pi(log_stds, noises, actions):
    """ 確率論的な行動の確率密度を返す． """
    # ガウス分布 `N(0, stds * I)` における `noises * stds` の確率密度の対数(= \log \pi(u|a))を計算する．
    # (torch.distributions.Normalを使うと無駄な計算が生じるので，下記では直接計算しています．)
    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    # tanh による確率密度の変化を修正する．
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    return log_pis

def reparameterize(means, log_stds):
    """ Reparameterization Trickを用いて，確率論的な行動とその確率密度を返す． """
    # 標準偏差．
    stds = log_stds.exp()
    # 標準ガウス分布から，ノイズをサンプリングする．
    noises = torch.randn_like(means)
    # Reparameterization Trickを用いて，N(means, stds)からのサンプルを計算する．
    us = means + noises * stds
    # tanhを適用し，確率論的な行動を計算する．
    actions = torch.tanh(us)

    # 確率論的な行動の確率密度の対数を計算する．
    log_pis = calculate_log_pi(log_stds, noises, actions)

    return actions, log_pis

def atanh(x):
    """ tanh の逆関数． """
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def evaluate_log_pi(means, log_stds, actions):
    """ 平均(mean)，標準偏差の対数(log_stds)でパラメータ化した方策における，行動(actions)の確率密度の対数を計算する． """
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

class PPOActor(nn.Module):

    def __init__(self, state_shape, action_shape):
        super().__init__()
        # 状態を受け取り, ガウス分布の平均（決定論的な行動）を出力するネットワーク
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0]),
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_log_pi(self.net(states), self.log_stds, actions)

class PPOCritic(nn.Module):

    def __init__(self, state_shape):
        super().__init__()
        # 状態を受け取り、状態価値を出力するネットワーク
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, states):
        return self.net(states)

def calculate_advantage(values, rewards, dones, next_values, gamma=0.995, lambd=0.997):
    """ GAEを用いて，状態価値のターゲットとGAEを計算する． """

    # TD誤差を計算する．
    deltas = rewards + gamma * next_values * (1 - dones) - values

    # GAEを初期化する．
    advantages = torch.empty_like(rewards)

    # 終端ステップを計算する．
    advantages[-1] = deltas[-1]

    # 終端ステップの1つ前から，順番にGAEを計算していく．
    for t in reversed(range(rewards.size(0) - 1)):
        advantages[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * advantages[t + 1]

    # 状態価値のターゲットをλ-収益として計算する．
    targets = advantages + values

    # GAEを標準化する．
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return targets, advantages

class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device=torch.device('cuda')):

        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)

        # 次にデータを挿入するインデックス．
        self._p = 0
        # バッファのサイズ．
        self.buffer_size = buffer_size

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        # リングバッファにする
        self._p = (self._p + 1) % self.buffer_size

    def get(self):
        return self.states, self.actions, self.rewards, self.dones, self.log_pis, self.next_states

    def clear(self):
        self.states = torch.empty_like(self.states)
        self.actions = torch.empty_like(self.actions)
        self.rewards = torch.empty_like(self.rewards)
        self.dones = torch.empty_like(self.dones)
        self.log_pis = torch.empty_like(self.log_pis)
        self.next_states = torch.empty_like(self.next_states)
        self._p = 0

class PPO:

    def __init__(self, state_shape, action_shape, device=torch.device('cuda'), seed=0,
                 batch_size=64, gamma=0.995, lr_actor=3e-4, lr_critic=3e-4, buffer_size=2048,
                 horizon=2048, num_updates=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=0.5,
                 max_action=1.,):

        # シードを設定する．
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # データ保存用のバッファ．
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        # Actor-Criticのネットワークを構築する．
        self.actor = PPOActor(
            state_shape=state_shape,
            action_shape=action_shape,
        ).to(device)
        self.critic = PPOCritic(
            state_shape=state_shape,
        ).to(device)

        # オプティマイザ．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # その他パラメータ．
        self.learning_steps = 0
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.horizon = horizon
        self.buffer_size = buffer_size
        self.num_updates = num_updates
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.max_action = max_action

    def is_update(self, steps):
        # ロールアウト1回分のデータが溜まったら学習する．
        return steps % self.horizon == 0

    def explore(self, state):
        # 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す．
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state) # ともにtorch.Size([1, 1])
        return action.cpu().numpy()[0] * self.max_action, log_pi.item()

    def exploit(self, state):
        # 決定論的な行動を返す．
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0] * self.max_action

    def get_value(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            value = self.critic(state)
        return value.cpu().numpy()[0]

    def step(self, env, state, t, steps):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)

        # ゲームオーバーではなく，最大ステップ数に到達したことでエピソードが終了した場合は，
        # 本来であればその先も試行が継続するはず．よって，終了シグナルをFalseにする．
        # NOTE: ゲームオーバーによってエピソード終了した場合には， done_masked=True が適切．
        # しかし，以下の実装では，"たまたま"最大ステップ数でゲームオーバーとなった場合には，
        # done_masked=False になってしまう．
        # その場合は稀で，多くの実装ではその誤差を無視しているので，今回も無視する．
        if t == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        # バッファにデータを追加する．
        self.buffer.append(state, action, reward, done_masked, log_pi, next_state)

        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1

        states, actions, rewards, dones, log_pis, next_states = self.buffer.get()
        actions /= self.max_action

        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        targets, advantages = calculate_advantage(values, rewards, dones, next_values, self.gamma, self.lambd)

        # バッファ内のデータを num_updates回ずつ使って，ネットワークを更新する．
        for _ in range(self.num_updates):
            # インデックスをシャッフルする．
            indices = np.arange(self.buffer_size)
            np.random.shuffle(indices)

            # ミニバッチに分けて学習する．
            for start in range(0, self.buffer_size, self.batch_size):
                idxes = indices[start:start+self.batch_size]
                self.update_critic(states[idxes], targets[idxes])
                self.update_actor(states[idxes], actions[idxes], log_pis[idxes], advantages[idxes])

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        # 学習を安定させるヒューリスティックとして，勾配のノルムをクリッピングする．
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, advantages):
        # 現在の方策における actions の確率密度の対数を計算する．
        # (log_pis_old はデータを収集したときの方策における actions の確率密度の対数です．)
        log_pis = self.actor.evaluate_log_pi(states, actions)

        mean_entropy = -log_pis.mean()
        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean() - self.coef_ent * mean_entropy

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        # 学習を安定させるヒューリスティックとして，勾配のノルムをクリッピングする．
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

# 平均収益を保存するための辞書．
returns = {'step': [], 'return': []}

def evaluate_policy(steps, eval_episodes=3):
    """ 複数エピソード環境を動かし，平均収益を記録する． """
    _returns = []
    for _ in range(eval_episodes):
        state = env_test.reset()
        done = False
        episode_return = 0.0

        while (not done):
            action = algo.exploit(state)
            state, reward, done, _ = env_test.step(action)
            episode_return += reward

        _returns.append(episode_return)

    mean_return = np.mean(_returns)
    returns['step'].append(steps)
    returns['return'].append(mean_return)

    print(f'Num steps: {steps:<6}   '
            f'Return: {mean_return:<5.1f}   '
            f'Time elapsed: {str(timedelta(seconds=int(time() - start_time)))}')

if __name__ == "__main__":

    rewards = []

    num_episodes = 1000
    gamma = 0.9
    buffer_length = 512
    batch_size = 128

    num_average_epidodes = 1
    # buffer_length = 128
    # batch_size = 16

    env = MyEnv(render_mode="human")
    env_test = MyEnv(render_mode="human")
    # env = MyEnv()
    observer = Observer(env) # Observer作成
    observer_test = Observer(env_test) # Observer作成
    # agent = ACAgent(actions = [0,1,2,3,4,5,6,7])
    actions = np.array([0,1,2,3,4,5,6,7])
    observation_space = np.arange(11)

    # replayBuffer = ReplayBuffer(buffer_length,batch_size)
    max_steps = 10000 # エピソードの最大ステップ数
    state = observer.reset() # observerを初期化し、前処理済みの初期状態を返す

    # ---------------
    # agent.load_models()
    # agent.initialize() # モデルを初期化
    # ---------------

    SEED = 0
    # データ収集を行うステップ数．
    NUM_STEPS = 5 * 10 ** 4
    # 評価の間のステップ数(インターバル)．
    EVAL_INTERVAL = 10 ** 3

    algo = PPO(
        state_shape=actions.shape,
        action_shape=observation_space.shape,
        seed=SEED,
    )

    # 学習開始の時間
    start_time = time()
    # エピソードのステップ数
    t = 0

    # 環境を初期化する
    state = observer.reset()

    """メインループ"""
    for steps in range(1, NUM_STEPS+1):
        # 環境(env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
        # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．
        state, t = algo.step(observer, state, t, steps)

        # アルゴリズムの準備ができていれば（ロールアウト1回分のデータが溜まっていたら）, 1回学習を行う.
        if algo.is_update(steps):
            algo.update()

        # 一定のインターバルで評価する.
        if steps % EVAL_INTERVAL == 0:
            evaluate_policy(steps)
        if steps % EVAL_INTERVAL == 0:
            observer.render()

    # # モデルをsaveしたい
    # joblib.dump(agent.model['actor'], 'actor.pkl') # actorのモデルを'actor.pkl'に保存する
    # joblib.dump(agent.model['critic'], 'critic.pkl') # criticのモデルを'critic.pkl'に保存する
    # 学習曲線の描画
    import matplotlib.pyplot as plt # matplotlib.pyplotのインポート

    # plt.plot(rewards) # 報酬の折れ線グラフの描画
    # plt.title('Train Curve', fontsize=20) # タイトルを設定
    # plt.ylabel('Rewards', fontsize=20) # 縦軸のラベルを設定
    # plt.xlabel('Episode', fontsize=20) # 横軸のラベルを指定
    # plt.show() # グラフを表示
    moving_average = np.convolve(rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)),moving_average)
    plt.title('Actor-Critic: average rewards in %d episodes' % num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.show()

    env.close()




    # # actions = [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]
    # actions = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # for action in actions: # 定義した行動のリストを逐次的に入力していく
    #     new_state, reward, done, info = observer.step(action) # 行動を入力して進める
    #     print('\n行動:', action)
    #     print('報酬:', reward)
    #     print('状態:', new_state)
    #     print(observer.render()) # 状態の可視化
    #     if done: # 終了判定(done)がTrueとなった場合終了
    #         env.close()
    #         break
