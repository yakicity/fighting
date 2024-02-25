
import pygame

import main
from game_parts.caluclation_parts import pythagoras_theorem as pyth
from game_parts.caluclation_parts import calcMovingSize as calcMovingSize


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

        self.damage_to_enemy = 13
        self.rigit_time_amount_to_enemy = 10
        self.rigit_time_amount = 15
        self.blow_speed_to_enemy = 10


    def figure(self):
        """
        キャラクターの大きさを決め、表示する

        """
        rect = (self.pos_x, self.pos_y, self.width, self.height)
        pygame.draw.rect(main.SURFACE, (255, 0, 0), rect)



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

    # def move(self):
    #     """
    #     キャラクターの移動を制限する
    #     Parameters
    #     ----------
    #     jump_speed : (int) ジャンプして一定時間後の速度を表す
    #     player_move : プレイヤー操作(上方向、下方向、右方向、左方向)を表す

    #     Returns
    #     -------
    #     変化後のキャラクターの位置情報、一定時間後のジャンプ速度、プレイヤー操作(上方向、下方向、右方向、左方向)を表す

    #     """

    #     # ジャンプボタンを押した時キャラクターをジャンプさせる。ジャンプし終えた後は重力によって落下する
    #     if self.player_move[0]:
    #         if self.jump_limit > 0:
    #             self.jump_limit -= 1
    #             self.pos_y -= self.jump_speed
    #             self.player_move[0] = False if not self.jump_speed else self.player_move[0]
    #             self.jump_speed = self.max_jump_speed if not self.jump_speed else self.jump_speed - 2

    #     elif not self.contact[0]:
    #         self.pos_y += self.gravity

    #     # 操作キャラを左右に動かす
    #     if self.player_move[2] and not self.contact[2]:
    #         self.pos_x += self.move_speed

    #     elif self.player_move[3] and not self.contact[3]:
    #         self.pos_x -= self.move_speed


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


    # def contact_judgment(self, enemy):
    #     """
    #     操作キャラとステージ(または敵キャラ)との接触判定、各接触方向に対して接触した場合のみCONTACTの各接触方向の要素にTrueを返す
    #     enemy_pos : 敵キャラの位置
    #     enemy_size : 敵キャラのサイズ

    #     Returns
    #     -------
    #     (list) 変更したcontactを返す

    #     """
    #     enemy_pos_x = enemy.pos_x
    #     enemy_pos_y = enemy.pos_y
    #     enemy_width = enemy.width
    #     enemy_height = enemy.height

    #     if self.pos_x == self.max_pos_x or (enemy_pos_x <= self.pos_x + self.width <= enemy_pos_x + enemy_width and self.pos_y == enemy_pos_y):
    #         self.contact[0] = True if self.max_pos_y <= self.pos_y + self.height else False
    #         self.contact[2] = True

    #     elif self.pos_x == 0 or (enemy_pos_x <= self.pos_x <= enemy_pos_x + enemy_width and self.pos_y == enemy_pos_y):
    #         self.contact[0] = True if self.max_pos_y <= self.pos_y + self.height else False
    #         self.contact[3] = True

    #     elif 0 < self.pos_x < self.max_pos_x:
    #         self.contact[0] = True if self.max_pos_y <= self.pos_y + self.height else False
    #         self.contact[2] = False
    #         self.contact[3] = False


    # def position_corr(self, stage_pos):
    #     """
    #     操作キャラとステージとの位置補正
    #     Parameters
    #     ----------
    #     stage_pos : 対象ステージの位置

    #     Returns
    #     -------
    #     補正後の操作キャラの位置(y軸方向)

    #     """

    #     if self.contact[0] and (self.pos_y + self.height - stage_pos[1]) != 0:
    #         self.pos_y = stage_pos[1] - self.height

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
        enemy_rigit = blow_speed = 0
        enemy.rigit_time = enemy_rigit
        enemy.blow_speed = blow_speed

        if self.rigit_time != 0:
            self.rigit_time -= 1
            return
        # 弱攻撃したら
        if self.action[0]:
            enemy.hit_judg = [False, False, False, False]
            # 自分が左向きなら，相手を左向きにヒットさせる
            if self.direction == 3:
                # 自分の左側と相手の右側が当たる可能性あり
                circle_pos = (self.pos_x, self.pos_y + self.height // 2)
                enemy_hit_pos = (enemy_pos[0] + enemy_size[0] / 2, enemy_pos[1] + enemy_size[1] / 2)
            elif self.direction == 2:
                # 自分の右側と相手の左側が当たる可能性あり
                circle_pos = (self.pos_x + self.width, self.pos_y + self.height // 2)
                enemy_hit_pos = (enemy_pos[0], enemy_pos[1] + enemy_size[1] / 2)
            radius = 30
            pygame.draw.circle(main.SURFACE, (0, 0, 250), circle_pos, radius)
            distance = pyth(circle_pos[0],enemy_hit_pos[0],circle_pos[1],enemy_hit_pos[1])
            if 0 <= distance <= radius + enemy_size[1] / 2:
                enemy.hit_judg[self.direction] = True
                enemy.damage += self.damage_to_enemy
                enemy.rigit_time = self.rigit_time_amount_to_enemy
                self.rigit_time = self.rigit_time_amount
                enemy.blow_speed = self.blow_speed_to_enemy

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



    # def hit_action(self):
    #     """
    #     相手の攻撃を受けた際に操作キャラの吹っ飛ぶ方向と距離を制御
    #     Parameters
    #     ----------
    #     hit_judg : (list) 上下左右のどちらの方向に飛ぶかの判定
    #     rigit_time : (int) 操作キャラの硬直時間
    #     blow_speed : (int) 攻撃が当たったときに吹っ飛ぶスピード

    #     Returns
    #     -------

    #     """
    #     if self.rigit_time == 0:
    #         self.hit_judg = [False, False, False, False]
    #     else:
    #         # 右に吹っ飛ぶ
    #         if self.hit_judg[2] is True and self.contact[2] is False:
    #             self.pos_x += self.blow_speed
    #         # 左に吹っ飛ぶ
    #         elif self.hit_judg[3] is True and self.contact[3] is False:
    #             self.pos_x -= self.blow_speed

    #         self.rigit_time -= 1

    def life(self, view):
        """
        キャラクターの体力ゲージ
        Parameters
        ----------
        damage : ダメージの蓄積量
        view : 体力ゲージの表示場所(left or right)

        """
        if view == "left":
            pygame.draw.rect(main.SURFACE, (120, 120, 120), (30, 20, 400, 30))
            if self.damage >= 390:
                return

            pygame.draw.rect(main.SURFACE, (250, 200, 0), (35 + self.damage, 25, 390 - self.damage, 20))

        elif view == "right":
            pygame.draw.rect(main.SURFACE, (120, 120, 120), (570, 20, 400, 30))
            if self.damage >= 390:
                return

            pygame.draw.rect(main.SURFACE, (250, 200, 0), (575, 25, 390 - self.damage, 20))
