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

from Fighter import Fighter


import math
def dist(a, b):
    a1,a2 = a[0],a[1]
    b1,b2 = b[0],b[1]
    z_2 = (a2 - a1) ** 2 + (b2 - b1) ** 2
    z = math.sqrt(z_2)
    return z

import gymnasium as gym
import numpy as np
import pygame
from pygame import gfxdraw
import math
import random
from typing import Optional
# reset(self) ：環境を初期状態にして初期状態(state)の観測(observation)をreturnする
# step(self, action) ：行動を受け取り行動後の環境状態(state)の観測(observation)・即時報酬(reward)・エピソードの終了判定(done)・情報(info)をreturnする
# render(self, mode) ：modeで指定されたように描画もしは配列をreturnする
# close(self) ：環境を終了する際に必要なクリーンアップ処理を実施する
# seed(self, seed=None) ：シードを設定する

# 使用するデータ型
# Discrete：[0, n-1]で指定したn個の離散値空間を扱う整数型（int）
# 使い方はDiscrete(n)
# Box：[low, high]で指定した連続値空間を扱う浮動小数点型（float）
# 使い方はBox(low, high, shape, dtype)
# lowおよびhighはshapeで与えたサイズと同じndarrayになります。
# 次に、TupleとDictの使い方について示します。

# Tuple：DiscreteやBoxなどの型をタプルで合体
# 使い方の例はTuple((Discrete(2), Box(0, 1, (4, ))))
# Dict：DiscreteやBoxなどの型を辞書で合体
# 使い方の例はDict({'position':Discrete(2), 'velocity':Box(0, 1, (4, ))})




class MyEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }

    # def __init__(self,models,is_cpu,render_mode: Optional[str] = None):
    def __init__(self,render_mode: Optional[str] = None):
        # action_space ：エージェントが取りうる行動空間を定義
        # observation_space：エージェントが受け取りうる観測空間を定義
        # reward_range ：報酬の範囲[最小値と最大値]を定義

        self.screen = None
        # self.clock = None
        self.clock = pygame.time.Clock()
        self.window_x = 1000
        self.window_y = 700

        # self.models = models
        # self.is_cpu = is_cpu
        RIGIT_MAX = 80
        self.jump_speed = 30  # <= ジャンプの初速度
        self.gravity = 16
        self.move_speed = 10
        self.stage_pos = [0, 550]  # <= 表示するステージの位置
        self.size = [50,80]
        self.radius = 30

        self.player1 = None
        self.player2 = None

        self.player1_color = (200,100,0)
        self.player2_color = (100,180,250)

        # アクション数定義
        # 移動：「左」「右」「上」「移動なし」，攻撃：「する」「しない」
        ACTION_NUM=8
        self.action_space = gym.spaces.Discrete(ACTION_NUM)
        self.render_mode = render_mode

        # 状態の範囲を定義,inattackrangeが１のときはどちらもアタックできる距離にある
        # 水平距離，垂直距離，P1x,P1y,P2x,P2y,inatackrange,p1cooldown,p2cooldown,p1canjump,p2canjupm
        max_distancex = self.window_x - self.size[0]
        max_distancey = self.stage_pos[1] - self.size[1]
        max_x = self.window_x - self.size[0]
        max_y = self.stage_pos[1] - self.size[1]
        LOW = np.array([0,0,0,0,0,0,0,0,0])
        HIGH = np.array([max_distancex,max_distancey,max_x,max_y,max_x,max_y,1,RIGIT_MAX,RIGIT_MAX])
        self.observation_space = gym.spaces.Box(low=LOW, high=HIGH)
        # 即時報酬の値
        self.reward_range = (-50,50)
        self.reset()

    def reset(self):
        # 環境を初期状態にする関数
        # 初期状態をreturnする
        # リセットの際に、乱数seedのリセットはしてはいけないので注意してください。
        player1_pos = [600, 470]  # <= 操作キャラの位置
        direction1 =3 #キャラの方向，0=上,1=した,2=右.3=左

        player2_pos = [200, 470]
        direction2 =2 #キャラの方向，0=上,1=した,2=右.3=左

        self.player1 = Fighter(self.size, self.gravity, self.move_speed,self.jump_speed,player1_pos,direction1)
        self.player2 = Fighter(self.size, self.gravity, self.move_speed,self.jump_speed,player2_pos,direction2)

        #初期化
        observation=[player1_pos[0]-player2_pos[0],player1_pos[1]-player2_pos[1],player1_pos[0],player1_pos[1],player2_pos[0],player2_pos[1],0,0,0]
        return np.array(observation, dtype=np.float32), {}

    def step(self, action_index):
        # 行動を受け取り行動後の状態をreturnする
        # stepメソッドは、action_spaceで定義された型に沿った行動値を受け取り、環境を1ステップだけ進めます。
        # 進めた後の観測、報酬、終了判定、その他の情報を生成し、リターンします。
        # infoにはデバックに役立つ情報などを辞書型として格納することができます。
        # 唯一、自由に使える変数なので、存分にinfoを活用しましょう。

        # observation ：object型。observation_spaceで設定した通りのサイズ・型のデータを格納。
        # reward ：float型。reward_rangeで設定した範囲内の値を格納。
        # done ：bool型。エピソードの終了判定。
        # info ：dict型。デバッグに役立つ情報など自由に利用可能。

        done=False

        self.player1.controlfromAction(action_index)
        # self.player2.controlrandomNotAction()
        self.player2.controlrandom()
        # control_character_random(self.player2)

        self.player1.contact_judgment(self.player2)
        self.player2.contact_judgment(self.player1)

        self.player1.move()
        self.player2.move()

        self.player1.contact_judgment(self.player2)
        self.player2.contact_judgment(self.player1)

        self.player1.character_action(self.player2)
        self.player2.character_action(self.player1)

        reward = 0
        # 攻撃したかどうかで報酬変化
        if any(self.player2.hit_judg):
            reward += 5
        if any(self.player1.hit_judg):
            reward -= 5

        # 攻撃が不発なら報酬変化
        if self.player1.misfire:
            reward -= 2
        # if self.player2.misfire:
        #     reward -= 1

        # 死んだかどうかで報酬変化
        if self.player1.damage >= 390:
            reward = -30
            done = True
        if self.player2.damage >= 390:
            reward = 30
            done = True

        self.player1.hit_action()
        self.player2.hit_action()

        self.player1.contact_judgment(self.player2)
        self.player2.contact_judgment(self.player1)

        inattackrange = 0
        if self.player1.pos_x > self.player2.pos_x:
            circle_pos = (self.player1.pos_x, self.player1.pos_y + self.player1.height // 2)
            enemy_hit_pos = (self.player2.pos_x + self.player2.pos_x / 2, self.player2.pos_y + self.player2.height / 2)
        else:
            circle_pos = (self.player1.pos_x + self.player1.width, self.player1.pos_y + self.player1.height // 2)
            enemy_hit_pos = (self.player2.pos_x, self.player2.pos_y + self.player2.height / 2)
        if 0 <= dist(circle_pos,enemy_hit_pos) <= self.radius + self.player2.height / 2:
            inattackrange = 1

        # p1canjump = 1 if self.player1.canMoveRange[0] == 0 else 0
        # p2canjump = 1 if self.player2.canMoveRange[0] == 0 else 0
        # 水平距離，垂直距離，P1x,P1y,P2x,P2y,inattackrange,p1cooldown,p2cooldown,p1canjump,p2canjupm
        observation=[abs(self.player1.pos_x - self.player2.pos_x),
                     abs(self.player1.pos_y - self.player2.pos_y),
                     self.player1.pos_x,
                     self.player1.pos_y,
                     self.player2.pos_x,
                     self.player2.pos_y,
                     inattackrange,
                     self.player1.rigit_time,
                     self.player2.rigit_time]


        if done == False:
            reward -= 0.5
            # 敵との距離で報酬変化
            dist_x = abs(self.player1.pos_x - self.player2.pos_x)
            reward -= dist_x / 1000

        # 今回の例ではtruncatedは使用しない
        truncated = False
        # 今回の例ではinfoは使用しない
        info = {}
        print(reward)
        return np.array(observation, dtype=np.float32),reward,done,truncated,info

    def render(self):
        if self.render_mode is None:
            return
        if self.screen is None:
            #初期化
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.window_x, self.window_y)) # ウィンドウサイズの指定
                # self.font = pygame.font.Font(None, 55)
            else: # mode == "rgb_array"
                self.screen = pygame.Surface((self.window_x, self.window_y))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # modeとしてhuman, rgb_array, ansiが選択可能
        # humanなら描画し, rgb_arrayならそれをreturnし, ansiなら文字列をreturnする

        self.surf = pygame.Surface((self.window_x, self.window_y))
        self.surf.fill((250, 250, 250))
        # text = self.font.render('Score:'+str(self.point), True, (255,255,255))   # 描画する文字列の設定
        # ステージの描画
        pygame.draw.rect(self.surf , (0, 200, 100), (self.stage_pos[0], self.stage_pos[1], 1500, 50))


        # プレイヤー1と2の描画(figureとaction.lifeの描画)
        player1_rect = (self.player1.pos_x, self.player1.pos_y, self.size[0], self.size[1])
        gfxdraw.box(self.surf, player1_rect,self.player1_color)
        player2_rect = (self.player2.pos_x, self.player2.pos_y, self.size[0], self.size[1])
        gfxdraw.box(self.surf, player2_rect,self.player2_color)

        # 攻撃の描画
        if self.player1.circle_pos is not None:
            gfxdraw.filled_circle(self.surf,self.player1.circle_pos[0],self.player1.circle_pos[1],30,(0,0,250))
        if self.player2.circle_pos is not None:
            gfxdraw.filled_circle(self.surf,self.player2.circle_pos[0],self.player2.circle_pos[1],30,(0,0,250))

        # lifeゲージの描画
        gfxdraw.rectangle(self.surf, (570, 20, 400, 30), (120, 120, 120))
        if self.player1.damage < 390:
            gfxdraw.box(self.surf, (575, 25, 390 - self.player1.damage, 20),self.player1_color)
        gfxdraw.rectangle(self.surf, (30, 20, 400, 30), (120, 120, 120))
        if self.player2.damage < 390:
            gfxdraw.rectangle(self.surf, (35 + self.player2.damage, 25, 390 - self.player2.damage, 20),self.player2_color)

        self.surf = pygame.transform.flip(self.surf, False, False)
        self.screen.blit(self.surf, (0, 0))
        # self.screen.blit(text, [10, 10])# 文字列の表示位置
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()


        # if self.render_mode == "human":
        #     self.screen = pygame.display.set_mode((self.window_x, self.window_y)) # ウィンドウサイズの指定
        #     # pygame.time.wait(30)#更新時間間隔
        #     # pygame.display.set_caption("Pygame Test") # ウィンドウの上の方に出てくるアレの指定
        #     self.screen.fill((0,0,0,)) # 背景色の指定。RGBだと思う
        #     text = self.font.render('Score:'+str(self.point), True, (255,255,255))   # 描画する文字列の設定
        #     self.screen.blit(text, [10, 10])# 文字列の表示位置

        #     pygame.draw.rect(self.screen, (255,0,0), (self.rect_x,self.rect_y,self.rect_width,self.rect_height))#的の描画
        #     pygame.draw.circle(self.screen, (0,95,0), (self.ball_x,self.ball_y), 10, width=0)#ボールの描画
        #     # pygame.draw.aaline(self.screen, (255,0,255), (self.ball_x,self.ball_y), (self.ball_x_next,self.ball_y_next), 0)#バーの描画
        #     pygame.display.update() # 画面更新
        #     pygame.event.pump()
        #     self.clock.tick(self.metadata["render_fps"])
        #     pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        pygame.quit()
    def seed(self, seed=None):
        pass



if __name__ == '__main__':
    pass
