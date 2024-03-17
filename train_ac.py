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



# actorとcriticのネットワーク（一部の重みを共有しています）
class ActorCriticNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=16):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size) # 状態を入力
        self.fc2a = nn.Linear(hidden_size, num_action)  # actor独自のlayer
        self.fc2c = nn.Linear(hidden_size, 1)  # critic独自のlayer

    def forward(self, x):
        h = F.elu(self.fc1(x))
        action_prob = F.softmax(self.fc2a(h), dim=-1)
        state_value = self.fc2c(h)
        # 行動選択確率, 状態価値
        return action_prob, state_value


class ActorCriticAgent:
    def __init__(self, num_state, num_action, gamma=0.99, lr=0.001):
        self.num_state = num_state
        self.gamma = gamma  # 割引率
        self.acnet = ActorCriticNetwork(num_state, num_action)
        self.optimizer = optim.Adam(self.acnet.parameters(), lr=lr)
        self.memory = []  # （報酬，行動選択確率，状態価値）のtupleをlistで保存

    # 方策を更新
    def update_policy(self):
        R = 0
        actor_loss = 0
        critic_loss = 0
        # エピソード内の各ステップの収益を後ろから計算（方策の良さの指標fをR-vとして, 方策勾配で目的関数を最大化していく）
        for r, prob, v in self.memory[::-1]:
            R = r + self.gamma*R
            advantage = R - v # 状態価値関数
            actor_loss -= torch.log(prob) * advantage #　方策勾配
            critic_loss += F.smooth_l1_loss(v, torch.tensor(R)) #　状態価値関数のF.smooth_l1_loss.
        actor_loss = actor_loss/len(self.memory)
        critic_loss = critic_loss/len(self.memory)
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

    # softmaxの出力が最も大きい行動を選択
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action_prob, _ = self.acnet(state_tensor.data)
        action = torch.argmax(action_prob.squeeze().data).item()
        print("action: ",action)
        return action

    # カテゴリカル分布からサンプリングして行動を選択
    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action_prob, state_value = self.acnet(state_tensor.data)
        print("action_prob: ",action_prob)
        action_prob, state_value = action_prob.squeeze(), state_value.squeeze()
        action = Categorical(action_prob).sample().item()
        print("action: ",action)
        return action, action_prob[action], state_value

    def add_memory(self, r, prob, v):
        self.memory.append((r, prob, v))

    def reset_memory(self):
        self.memory = []

class ACAgent():
    def __init__(self, actions): # 初期化メソッド
        self.actions = actions
        self.model = None
        self.initialized = False


    def policy(self, state,episode): # 状態を渡して行動を選択するメソッド
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+episode)
        if epsilon <= np.random.uniform(0, 1):
            estimated = self.estimate(state)
            action = np.argmax(estimated)  # 最大の報酬を返す行動を選択する
        else:
            action = np.random.choice(self.actions)  # ランダムに行動する
        return action
        # estimated = self.estimate(state)
        # prob_list = [np.exp(q)/np.exp(estimated).sum() for q in estimated]
        # prob_sum = sum(prob_list)
        # prob_list_normalized = [prob / prob_sum for prob in prob_list]
        # # print(sum(prob_list_normalized))
        # action = np.random.choice(self.actions, size = 1, p = prob_list_normalized)[0]
        # return action

    def initialize(self): # モデルを初期化するメソッド
        # モデルは中間層が1層で変数の数を32個
        actor = MLPRegressor(hidden_layer_sizes=(32,), max_iter=1) # Actorのモデル
        critic = MLPRegressor(hidden_layer_sizes=(32,), max_iter=1) # Criticのモデル
        self.model = {'actor':actor, 'critic':critic} # ActorとCriticを辞書型として持っておく
        self.initialized = False # 初期化フラグをTrueにしておく

    def load_models(self):
        models = {}
        models['actor'] = joblib.load('actor.pkl') # actorのモデルを読み込む
        models['critic'] = joblib.load('critic.pkl') # criticのモデルを読み込む
        self.model = models
        self.initialized = True # 初期化フラグをTrueにしておく

    # 重みの学習
    def update(self, replayBuffer, batch_size, gamma):
        self.initialized = True
        mini_batch = replayBuffer.get_batch()
        # 蓄積した経験において現在の状態と遷移先の状態の組を作る
        states = [] # 現在の状態
        new_states = [] # 遷移先の状態
        for state, action, reward, new_state,done in mini_batch:
            states.append(state)
            new_states.append(new_state)
        states = np.concatenate(states, axis=0) # (n, 11)のnumpy.arrayとした
        new_states = np.concatenate(new_states, axis=0) # (n, 11)のnumpy.arrayとした

        # criticの学習
        try: # partial_fitする前にpredictはできないため例外処理を実装する
            estimated_values = self.model['critic'].predict(new_states) # 現在の状態に対する新しい価値評価の見積もり(n,)
            for i, (state, action, reward, new_state,done)  in enumerate(mini_batch):
                value = reward
                if not done: # doneフラグがFalseの時(棒が倒れていない時)次の状態がある
                    value += gamma*estimated_values[i]
                estimated_values[i] = value
        except NotFittedError:
            estimated_values = np.random.random(size=len(states))
        self.model['critic'].partial_fit(states, estimated_values) # 新しい価値の見積もりに近い出力になるように学習

        # actorの学習
        try: # partial_fitする前にpredictはできないため例外処理を実装する
            estimated_action_values = self.model['actor'].predict(states) # 現在の状態に対する価値評価(n,len(self.actions))
        except NotFittedError:
            estimated_action_values = np.random.random(size=(len(states), len(self.actions)))
        for i, (state, action, reward, new_state,done) in enumerate(mini_batch): # とった行動に対して新しい価値評価の見積もりに変える
            estimated_action_values[i, action] = estimated_values[i]
        self.model['actor'].partial_fit(states, estimated_action_values) # 新しい価値の見積もりに近い出力になるように学習


    def estimate(self, state): # 状態を渡して各行動の価値評価を推定するメソッド
        if self.initialized:
            return self.model['actor'].predict(state)[0] # (1,len(self.actions))の形で返るので、(len(self.actions),)で出力する
        else:
            return np.random.random(size=len(self.actions)) # 初期化フラグがFalseの時はランダムな値を出力する

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
    # env = MyEnv()
    observer = Observer(env) # Observer作成
    # agent = ACAgent(actions = [0,1,2,3,4,5,6,7])
    actions = [0,1,2,3,4,5,6,7]
    observation_space_n = 11
    agent = ActorCriticAgent(observation_space_n,len(actions))

    # replayBuffer = ReplayBuffer(buffer_length,batch_size)
    max_steps = 10000 # エピソードの最大ステップ数
    state = observer.reset() # observerを初期化し、前処理済みの初期状態を返す

    # ---------------
    # agent.load_models()
    # agent.initialize() # モデルを初期化
    # ---------------

    for episode in range(num_episodes):
        state = observer.reset() # observerを初期化し、前処理済みの初期状態を返す
        done = False # エピソードの終了フラグ
        reward_per_episode = 0 # 1エピソード当たりの報酬の総和

        # while not done: # エピソードが終了しない間はずっと処理を行う
        for t in range(max_steps):
            print("episode: ",episode)
            action, prob, state_value = agent.get_action(state)  #  行動を選択
            # action = agent.policy(state,episode) # agentが戦略に従って行動を選択する
            new_state, reward, done, info = observer.step(action) # agentがとった行動に対してobserverが前処理済みの状態などを返す
            agent.add_memory(reward, prob, state_value)
            # replayBuffer.add(state, action,reward,new_state,done ) # 経験を蓄積
            reward_per_episode += reward # 獲得報酬を計算
            state = new_state
            if done:
                agent.update_policy()
                agent.reset_memory() # 方策が更新されているので
                break

            # if len(replayBuffer) >= batch_size: # 経験が蓄積しているとき学習を行う
            #     agent.update(replayBuffer,batch_size,gamma)

            if episode == 999:
                env.render() # 状態の可視化
            # state = new_state # 状態を更新

        rewards.append(reward_per_episode) # appendメソッドで獲得した報酬を格納
        # agent.update_policy()
        # agent.reset_memory() # 方策が更新されているので
        if episode % 50 == 0:
            print("Episode %d finished | Episode reward %f" % (episode, reward_per_episode))

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
