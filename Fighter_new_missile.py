
import pygame
import numpy as np
import math
from Missile import Missile

def dist(a1, a2, b1, b2):
    z_2 = (a2 - a1) ** 2 + (b2 - b1) ** 2
    z = math.sqrt(z_2)

    return z
def calcMovingSize(can,want):
    if can <= want:
        return can
    else:
        return want


class Fighter_new:

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
        action_all : bool[] : [上方向に移動、下方向に移動、右方向に移動、左方向に移動、弱攻撃]
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
        self.direction = direction
        self.action_all = [False, False, False, False,False,True,False]
        self.hit_judg = [False, False, False, False]
        self.damage = 0
        self.rigit_time = 0
        self.blow_speed = 0
        self.actionrigit = 0

        self.damage_to_enemy = 78
        self.rigit_time_amount_to_enemy = 100
        self.rigit_time_amount = 100
        self.blow_speed_to_enemy = 10
        self.radius = 30

        self.circle_pos = None
        # 攻撃が不発
        self.misfire = False
        self.action_ind = 5
        self.missile_num = 0
        self.missile = None
        self.missile_x = self.pos_x
        self.missile_y = self.pos_y
        self.next_missile_x = self.pos_x
        self.missile_hit = False

    def controlfromAction(self,actionindex):
        player_move_up = self.action_all[0]
        self.action_all = [False, False, False, False,False,False,False]
        # 上方向に移動、下方向に移動、右方向に移動、左方向に移動、弱攻撃、なし

        # 上キーが押されたら
        if actionindex == 0:
            self.action_all[0] = True if self.canMoveRange[0] == 0 else player_move_up
        else:
            self.action_all[actionindex] = True

        # 左右キー押されたらdirection変更
        if actionindex == 2:
            self.direction = 2
        elif actionindex == 3:
            self.direction = 3

    def controlrandomNotAction(self):
        player_move_up = self.action_all[0]
        self.action_all = [False, False, False, False,False,False,False]

        rand_move = np.random.choice([0,1,2,3], 1)[0]
        if rand_move == 0:
            self.action_all[0] = True if self.canMoveRange[0] == 0 else player_move_up
        else:
            self.action_all[rand_move] = True
            if rand_move == 3:
                self.direction = 3
            elif rand_move == 2:
                self.direction = 2


    def controlrandom(self):
        player_move_up = self.action_all[0]
        self.action_all = [False, False, False, False,False,False,False]
        rand_move = np.random.choice([0,1,2,3,4,5,6], 1)[0]
        if rand_move == 0:
            self.action_all[0] = True if self.canMoveRange[0] == 0 else player_move_up
        else:
            self.action_all[rand_move] = True
            if rand_move == 3:
                self.direction = 3
            elif rand_move == 2:
                self.direction = 2

    def controlstop(self):
        player_move_up = self.action_all[0]
        self.action_all = [False, False, False, False,False,False,False]

        rand_move = 5
        self.action_all[rand_move] = True

    def controlmissile(self):
        player_move_up = self.action_all[0]
        self.action_all = [False, False, False, False,False,False,False]
        rand_move = np.random.choice([0,2,3,6], 1)[0]
        if rand_move == 0:
            self.action_all[0] = True if self.canMoveRange[0] == 0 else player_move_up
        else:
            self.action_all[rand_move] = True
            if rand_move == 3:
                self.direction = 3
            elif rand_move == 2:
                self.direction = 2

    def move(self):
        """
        キャラクターの移動を制限する
        Parameters
        ----------
        jump_speed : (int) ジャンプして一定時間後の速度を表す
        action_all : プレイヤー操作(上方向、下方向、右方向、左方向、弱攻撃)を表す

        Returns
        -------
        変化後のキャラクターの位置情報、一定時間後のジャンプ速度、プレイヤー操作(上方向、下方向、右方向、左方向)を表す

        """

        # ジャンプボタンを押した時キャラクターをジャンプさせる。ジャンプし終えた後は重力によって落下する
        if self.action_all[0]:
            movesize = calcMovingSize(self.canMoveRange[1],self.jump_speed)
            self.pos_y -= movesize
            self.action_all[0] = False if not self.jump_speed else self.action_all[0]
            self.jump_speed = self.max_jump_speed if not self.jump_speed else self.jump_speed - 2
            if self.action_all[0] == False:
                self.action_all[5] = True

        else:
            movesize = calcMovingSize(self.canMoveRange[0],self.gravity)
            self.pos_y += movesize

        # 操作キャラを右に動かす
        if self.action_all[2]:
            movesize = calcMovingSize(self.canMoveRange[2],self.move_speed)
            self.pos_x += movesize

        # 操作キャラを左に動かす
        if self.action_all[3]:
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
        if self.action_all[4]:
            self.rigit_time = self.rigit_time_amount
            enemy.hit_judg = [False, False, False, False]
            # 自分が左向きなら，相手を左向きにヒットさせる
            if self.direction == 3 and (self.pos_x >= enemy.pos_x):
                # 自分の左側と相手の右側が当たる可能性あり
                self.circle_pos = (self.pos_x, self.pos_y + self.height // 2)
                # enemy_hit_pos = (enemy_pos[0] + enemy_size[0] / 2, enemy_pos[1] + enemy_size[1] / 2)
            elif self.direction == 2 and (self.pos_x <= enemy.pos_x):
                # 自分の右側と相手の左側が当たる可能性あり
                self.circle_pos = (self.pos_x + self.width, self.pos_y + self.height // 2)
                # enemy_hit_pos = (enemy_pos[0], enemy_pos[1] + enemy_size[1] / 2)

            # distance = dist(self.circle_pos[0],enemy_hit_pos[0],self.circle_pos[1],enemy_hit_pos[1])
            # if 0 <= distance <= radius + enemy_size[1] / 2:
            dist_x = abs(self.pos_x - enemy.pos_x) - self.width
            dist_y = abs(self.pos_y - enemy.pos_y)
            dist_xy = math.sqrt(dist_x ** 2 + dist_y ** 2)

            if 0 <= dist_xy <= self.radius:
                enemy.hit_judg[self.direction] = True
                enemy.damage += (self.damage_to_enemy / 2)
                enemy.rigit_time = self.rigit_time_amount_to_enemy
                enemy.blow_speed = self.blow_speed_to_enemy
            else:
                self.misfire = True

        elif self.action_all[6]:  # ミサイル攻撃
            self.rigit_time = self.rigit_time_amount / 2
            if self.missile_num == 0:
                self.missile_num = 1
                # 右向き
                if self.direction == 2:
                    missile = Missile(self.pos_x + self.width, self.pos_y + self.height // 2, 1,self.move_speed * 2)
                # 左向き
                elif self.direction == 3:
                    missile = Missile(self.pos_x, self.pos_y + self.height // 2, -1, self.move_speed * 2)
                self.missile = missile
                self.missile_x = missile.x
                self.missile_y = missile.y
                self.next_missile_x = missile.x + self.missile.speed
                if self.next_missile_x > 1000 or self.next_missile_x < 0:
                    self.next_missile_x = self.pos_x

        # self.action_all[4] = False

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

    def hit_missile(self,enemy):
        enemy.missile_hit = False

        if self.missile_num == 1:
            if self.missile.active == False:
                self.missile_num = 0
                self.missile_x = self.pos_x
                self.next_missile_x = self.pos_x
                self.missile_y = self.pos_y
            else:
                self.missile.move()
                self.missile_x = self.missile.x
                self.next_missile_x = self.missile.x + self.missile.speed
                if self.next_missile_x > 1000 or self.next_missile_x < 0:
                    self.next_missile_x = self.pos_x
                    self.next_missile_y = self.pos_y
                self.missile_y = self.missile.y
                if self.missile.check_collision(enemy.pos_x, enemy.pos_y, self.width, self.height):
                    enemy.missile_hit = True
                    enemy.damage += self.damage_to_enemy
        else:
            self.missile_x = self.pos_x
            self.next_missile_x = self.pos_x
            self.missile_y = self.pos_y



if __name__ == '__main__':
    pass