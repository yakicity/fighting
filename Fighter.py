
import pygame
import numpy as np
import math

def dist(a1, a2, b1, b2):
    z_2 = (a2 - a1) ** 2 + (b2 - b1) ** 2
    z = math.sqrt(z_2)

    return z
def calcMovingSize(can,want):
    if can <= want:
        return can
    else:
        return want


class Fighter:

    def __init__(self, size, gravity, move_speed,jump_speed,position,direction):
        """
        width,height : 操作キャラの大きさ
        gravity : 重力
        move_speed : 速度
        jump_speed : ジャンプの初速度
        pos_x,_y : 操作キャラの位置
        contact : bool[] : [キャラクターと地面との接触判定、頭上のオブジェクトとの接触判定、右側とオブジェクトとの接触判定、左側とオブジェクトの接触判定]
        canMoveRange : int[] : [下、上、右、左]方向に動ける範囲
        player_move : bool[]: [上方向に移動、下方向に移動、右方向に移動、左方向に移動]
        direction1 =0 #キャラの方向,0=上,1=した,2=右.3=左
        action : bool[] :[弱攻撃、必殺技、ガード、掴み]
        hit_judg : bool[]:ヒットした結果進む方向[上方向にhit、下方向にhit、右方向にhit、左方向にhit]，
        damage : キャラが受けたトータルダメージ量
        rigit_time: 操作キャラの硬直時間
        blow_speed :操作キャラが吹っ飛ばされたときの速度

        Returns
        -------
        変化後のキャラクターの位置情報、一定時間後のジャンプ速度、プレイヤー操作(上方向、下方向、右方向、左方向)を表す
        """

        self.max_jump_speed = 30
        self.max_pos_x = 1000 - size[0]
        self.max_pos_y = 550 - size[1]

        self.width = size[0]
        self.height = size[1]
        self.gravity = gravity
        self.move_speed = move_speed
        self.jump_speed = jump_speed
        self.pos_x = position[0]
        self.pos_y = position[1]
        # self.contact = [False, False, False, False]
        self.canMoveRange = [0, position[1], self.max_pos_x - position[0],position[0]]
        self.player_move = [False, False, False, False]
        self.direction = direction
        self.action = [False, False, False, False]
        self.hit_judg = [False, False, False, False]
        self.damage = 0
        self.rigit_time = 0
        self.blow_speed = 0
        self.actionrigit = 0

        self.damage_to_enemy = 39
        self.rigit_time_amount_to_enemy = 20
        self.rigit_time_amount = 80
        self.blow_speed_to_enemy = 10

        self.circle_pos = None
        # 攻撃が不発
        self.misfire = False

    def controlfromAction(self,actionindex):
        player_move_up = self.player_move[0]

        self.action = [False, False, False, False]
        self.player_move = [False, False, False, False]
        # 移動：「上」「左」「右」「移動なし」，攻撃：「する」「しない」
        # 上する，左する，右する，する，上しない，，，，，
        if actionindex == 0:
            self.action[0] = True
            self.player_move[0] = True if self.canMoveRange[0] == 0 else player_move_up
        elif actionindex == 1:
            self.action[0] = True
            self.player_move[3] = True
            self.direction = 3
        elif actionindex == 2:
            self.action[0] = True
            self.player_move[2] = True
            self.direction = 2
        elif actionindex == 3:
            self.action[0] = True
        elif actionindex == 4:
            self.player_move[0] = True if self.canMoveRange[0] == 0 else player_move_up
        elif actionindex == 5:
            self.player_move[3] = True
            self.direction = 3
        elif actionindex == 6:
            self.player_move[2] = True
            self.direction = 2

    def controlrandomNotAction(self):
        player_move_up = self.player_move[0]

        self.action = [False, False, False, False]
        self.player_move = [False, False, False, False]

        # 地面に接触してたらジャンプ
        action_type = np.random.choice([0,1], 1)[0]
        if action_type == 0:
            self.player_move[0] = True if self.canMoveRange[0] == 0 else player_move_up
        else:
            if self.canMoveRange[0] == 0:
                rand_move = np.random.choice([0,1,2,3], 1)[0]
            else:
                rand_move = np.random.choice([1,2,3], 1)[0]
            self.player_move[rand_move] = True
            if rand_move == 3:
                self.direction = 3
            elif rand_move == 2:
                self.direction = 2



    def controlrandom(self):
        player_move_up = self.player_move[0]

        self.action = [False, False, False, False]
        self.player_move = [False, False, False, False]

        # 地面に接触してたらジャンプ
        action_type = np.random.choice([0,1], 1)[0]
        if action_type == 0:
            self.player_move[0] = True if self.canMoveRange[0] == 0 else player_move_up
        else:
            if self.canMoveRange[0] == 0:
                rand_move = np.random.choice([0,1,2,3], 1)[0]
            else:
                rand_move = np.random.choice([1,2,3], 1)[0]
            self.player_move[rand_move] = True
            if rand_move == 3:
                self.direction = 3
            elif rand_move == 2:
                self.direction = 2

        rand_action = np.random.choice([0,1,2,3], 1)[0]
        self.action[rand_action] = True


    def move(self):
        """
        キャラクターの移動を制限する
        Parameters
        ----------
        jump_speed : (int) ジャンプして一定時間後の速度を表す
        player_move : プレイヤー操作(上方向、下方向、右方向、左方向)を表す

        Returns
        -------
        変化後のキャラクターの位置情報、一定時間後のジャンプ速度、プレイヤー操作(上方向、下方向、右方向、左方向)を表す

        """

        # ジャンプボタンを押した時キャラクターをジャンプさせる。ジャンプし終えた後は重力によって落下する
        if self.player_move[0]:
            movesize = calcMovingSize(self.canMoveRange[1],self.jump_speed)
            self.pos_y -= movesize
            self.player_move[0] = False if not self.jump_speed else self.player_move[0]
            self.jump_speed = self.max_jump_speed if not self.jump_speed else self.jump_speed - 2

        else:
            movesize = calcMovingSize(self.canMoveRange[0],self.gravity)
            self.pos_y += movesize

        # 操作キャラを右に動かす
        if self.player_move[2]:
            movesize = calcMovingSize(self.canMoveRange[2],self.move_speed)
            self.pos_x += movesize

        # 操作キャラを左に動かす
        if self.player_move[3]:
            movesize = calcMovingSize(self.canMoveRange[3],self.move_speed)
            self.pos_x -= movesize


    def contact_judgment(self, enemy):
        """
        操作キャラとステージ(または敵キャラ)をもとに，各方向にどれだけ動けるか
        canMoveRange : int[] : [下、上、右、左]方向に動ける範囲
        """
        enemy_pos_x = enemy.pos_x
        enemy_pos_y = enemy.pos_y
        enemy_width = enemy.width
        enemy_height = enemy.height

        # 上下
        # 敵と自分のxがかぶっていたら
        if self.pos_x - enemy_width < enemy_pos_x < self.pos_x + self.width:
            if enemy_pos_y > self.pos_y:
                self.canMoveRange[0] = enemy_pos_y - self.pos_y - self.height
                self.canMoveRange[1] = self.pos_y
            elif enemy_pos_y < self.pos_y:
                self.canMoveRange[1] = self.pos_y - enemy_pos_y - enemy_height
                self.canMoveRange[0] = self.max_pos_y - self.pos_y
        # かぶってなかったら
        else:
            self.canMoveRange[0] = self.max_pos_y - self.pos_y
            self.canMoveRange[1] = self.pos_y

        # 左右
        # 敵と自分のyがかぶっていたら
        if self.pos_y - enemy_height < enemy_pos_y < self.pos_y + self.height:
            if enemy_pos_x > self.pos_x:
                self.canMoveRange[2] = enemy_pos_x - self.pos_x - self.width
                self.canMoveRange[3] = self.pos_x
            elif enemy_pos_x < self.pos_x:
                self.canMoveRange[3] = self.pos_x - enemy_pos_x - enemy_width
                self.canMoveRange[2] = self.max_pos_x - self.pos_x
        # かぶってなかったら
        else:
            self.canMoveRange[2] = self.max_pos_x - self.pos_x
            self.canMoveRange[3] = self.pos_x

    def character_action(self, enemy):
        """
        敵プレイヤーを攻撃した際の判定
        Parameters
        ----------
        player_move : (list) 操作キャラの移動判定
        action : (list) 操作キャラの動作判定
        hit_judg : (list) 自分の攻撃と攻撃対象との当たり判定
        enemy_pos: (list) 攻撃対象キャラの位置
        enemy_size : (list) 攻撃対象キャラの幅と高さ
        enemy_damage : (int) 敵に与えるダメージ

        Returns
        -------

        """
        enemy_pos_x = enemy.pos_x
        enemy_pos_y = enemy.pos_y
        enemy_width = enemy.width
        enemy_height = enemy.height
        enemy_pos = (enemy_pos_x,enemy_pos_y)
        enemy_size = (enemy_width,enemy_height)
        enemy_damage = enemy.damage
        # enemy_rigit = blow_speed = 0
        # enemy.rigit_time = enemy_rigit
        # enemy.blow_speed = blow_speed

        self.circle_pos = None
        self.misfire = False


        if self.rigit_time != 0:
            self.rigit_time -= 1
            return
        # 弱攻撃したら
        if self.action[0]:
            self.rigit_time = self.rigit_time_amount
            enemy.hit_judg = [False, False, False, False]
            # 自分が左向きなら，相手を左向きにヒットさせる
            if self.direction == 3:
                # 自分の左側と相手の右側が当たる可能性あり
                self.circle_pos = (self.pos_x, self.pos_y + self.height // 2)
                enemy_hit_pos = (enemy_pos[0] + enemy_size[0] / 2, enemy_pos[1] + enemy_size[1] / 2)
            elif self.direction == 2:
                # 自分の右側と相手の左側が当たる可能性あり
                self.circle_pos = (self.pos_x + self.width, self.pos_y + self.height // 2)
                enemy_hit_pos = (enemy_pos[0], enemy_pos[1] + enemy_size[1] / 2)
            radius = 30

            distance = dist(self.circle_pos[0],enemy_hit_pos[0],self.circle_pos[1],enemy_hit_pos[1])
            if 0 <= distance <= radius + enemy_size[1] / 2:
                enemy.hit_judg[self.direction] = True
                enemy.damage += self.damage_to_enemy
                enemy.rigit_time = self.rigit_time_amount_to_enemy
                enemy.blow_speed = self.blow_speed_to_enemy
            else:
                self.misfire = True

        self.action[0] = False

    def hit_action(self):
        """
        相手の攻撃を受けた際に操作キャラの吹っ飛ぶ方向と距離を制御
        Parameters
        ----------
        hit_judg : (list) 上下左右のどちらの方向に飛ぶかの判定
        rigit_time : (int) 操作キャラの硬直時間
        blow_speed : (int) 攻撃が当たったときに吹っ飛ぶスピード

        Returns
        -------

        """
        if self.rigit_time == 0:
            self.hit_judg = [False, False, False, False]
        else:
            # 右に吹っ飛ぶ
            if self.hit_judg[2]:
                movesize = calcMovingSize(self.canMoveRange[2],self.blow_speed)
                self.pos_x += movesize
            # 左に吹っ飛ぶ
            elif self.hit_judg[3]:
                movesize = calcMovingSize(self.canMoveRange[3],self.blow_speed)
                self.pos_x -= movesize

            self.rigit_time -= 1




if __name__ == '__main__':
    pass