import gymnasium as gym
import numpy as np
import pygame
from pygame import gfxdraw
import math
import random
from typing import Optional
from game_parts import control, fighter, stage


class MyEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 20,
    }

    def __init__(self,render_mode: Optional[str] = None):
        # action_space ：エージェントが取りうる行動空間を定義
        # observation_space：エージェントが受け取りうる観測空間を定義
        # reward_range ：報酬の範囲[最小値と最大値]を定義

        self.screen = None
        self.clock = None
        self.window_x = 1000
        self.window_y = 700

        JUMP_SPEED = 30  # <= ジャンプの初速度
        GRAVITY = 16
        MOVE_SPEED = 10
        STAGE_POS = [0, 550]  # <= 表示するステージの位置

        # アクション数定義
        # 移動：「左」「右」「上」「移動なし」，攻撃：「する」「しない」
        ACTION_NUM=6
        self.action_space = gym.spaces.Discrete(ACTION_NUM)
        self.render_mode = render_mode

        # 状態の範囲を定義
        # 状態はボールの位置x,y、皿の位置x,射出方向
        LOW = np.array([0,0,0,-135.06847396007072])
        HIGH = np.array([self.window_x,self.window_y,self.window_x-self.rect_width,135.06847396007072])
        self.observation_space = gym.spaces.Box(low=LOW, high=HIGH)
        # 即時報酬の値
        self.reward_range = (0,1)
        self.reset()



pygame.init()
SURFACE = pygame.display.set_mode((1000, 700))
FPSCLOCK = pygame.time.Clock()
STAGE_POS = [0, 550]  # <= 表示するステージの位置


def main():
    player1_pos = [600, 470]  # <= 操作キャラの位置
    player1_size = [50, 80]  # <= 操作キャラの大きさ [横幅, 縦幅]
    move1 = [False, False, False, False]  # <= [上方向に移動、下方向に移動、右方向に移動、左方向に移動]
    jump_speed1 = 20  # <= ジャンプの初速度
    contact1 = [False, False, False, False]  # <= [キャラクターと地面との接触判定、頭上のオブジェクトとの接触判定、右側とオブジェクトとの接触判定、左側とオブジェクトの接触判定
    action1 = [False, False, False, False]  # <= [弱攻撃、必殺技、ガード、掴み]
    hit_judg1 = [False, False, False, False]  # <= [上方向にhit、下方向にhit、右方向にhit、左方向にhit]
    damage1 = 0  # <= 操作キャラが受けたトータルダメージ量
    rigit_time1 = 0  # <= 操作キャラの硬直時間
    blow_speed1 = 0  # <= 操作キャラが吹っ飛ばされたときの速度
    player2_pos = [200, 470]
    player2_size = [50, 80]
    move2 = [False, False, False, False]
    jump_speed2 = 20
    contact2 = [False, False, False, False]
    hit_judg2 = [False, False, False, False]
    damage2 = 0
    rigit_time2 = 0
    blow_speed2 = 0

    while True:
        # 背景を真っ白に設定する
        SURFACE.fill((250, 250, 250))

        # プレイヤー１のキャラクター操作
        move1, action1 = control.control_character(move1, action1, contact1)


        # 戦闘ステージを作成
        stage.base_stage(STAGE_POS)

        # プレイヤー１の操作キャラの表示、行動制御
        player_1 = fighter.Fighter(player1_pos, player1_size, 16, 10, contact1, rigit_time1)
        player_1.figure()
        player1_pos, jump_speed1, move1 = player_1.move(jump_speed1, move1)
        contact1 = player_1.contact_judgment(player2_pos, player2_size)
        player1_pos = player_1.position_corr(STAGE_POS)
        action1, rigit_time1, hit_judg2, damage2, rigit_time2, blow_speed2 = player_1.character_action(move1, action1, hit_judg2, player2_pos, player2_size, damage2)
        player_1.life(damage1, "right")

        # プレイヤー2のキャラ表示
        player_2 = fighter.Fighter(player2_pos, player2_size, 16, 10, contact2, rigit_time2)
        player_2.figure()
        player2_pos, jump_speed2, move2 = player_2.move(jump_speed2, move2)
        contact2 = player_2.contact_judgment(player1_pos, player1_size)
        player2_pos = player_2.position_corr(STAGE_POS)
        player2_pos, rigit_time2, hit_judg2 = player_2.hit_action(hit_judg2, rigit_time2, blow_speed2)
        # action2, rigit_time2, hit_judg1, damage1, rigit_time1, blow_speed1 = player_2.character_action(move2, action2, hit_judg1, player1_pos, player1_size, damage1)

        player_2.life(damage2, "left")

        # 画面を更新する
        pygame.display.update()

        # 画面の更新を30fps(1秒間に30枚画面が切り替わる)に設定する
        FPSCLOCK.tick(30)


if __name__ == '__main__':
    main()
