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

    num_episodes = 100
    gamma = 0.9
    buffer_length = 512
    batch_size = 128
    # buffer_length = 128
    # batch_size = 16

    env = MyEnv(render_mode="human")
    # env = MyEnv()
    observer = Observer(env) # Observer作成
    agent = ACAgent(actions = [0,1,2,3,4,5,6,7])
    replayBuffer = ReplayBuffer(buffer_length,batch_size)

    state = observer.reset() # observerを初期化し、前処理済みの初期状態を返す

    # ---------------
    agent.load_models()
    # agent.initialize() # モデルを初期化
    # ---------------

    for episode in range(num_episodes):
        state = observer.reset() # observerを初期化し、前処理済みの初期状態を返す
        done = False # エピソードの終了フラグ
        reward_per_episode = 0 # 1エピソード当たりの報酬の総和

        while not done: # エピソードが終了しない間はずっと処理を行う
            action = agent.policy(state,episode) # agentが戦略に従って行動を選択する
            new_state, reward, done, info = observer.step(action) # agentがとった行動に対してobserverが前処理済みの状態などを返す
            replayBuffer.add(state, action,reward,new_state,done ) # 経験を蓄積
            reward_per_episode += reward # 獲得報酬を計算

            if len(replayBuffer) >= batch_size: # 経験が蓄積しているとき学習を行う
                agent.update(replayBuffer,batch_size,gamma)

            print(env.render()) # 状態の可視化
            state = new_state # 状態を更新

        rewards.append(reward_per_episode) # appendメソッドで獲得した報酬を格納

    # モデルをsaveしたい
    joblib.dump(agent.model['actor'], 'actor.pkl') # actorのモデルを'actor.pkl'に保存する
    joblib.dump(agent.model['critic'], 'critic.pkl') # criticのモデルを'critic.pkl'に保存する
    # 学習曲線の描画
    import matplotlib.pyplot as plt # matplotlib.pyplotのインポート

    plt.plot(rewards) # 報酬の折れ線グラフの描画
    plt.title('Train Curve', fontsize=20) # タイトルを設定
    plt.ylabel('Rewards', fontsize=20) # 縦軸のラベルを設定
    plt.xlabel('Episode', fontsize=20) # 横軸のラベルを指定
    plt.show() # グラフを表示





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
