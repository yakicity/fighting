import sys
import pygame
import numpy as np
from pygame.locals import QUIT, KEYDOWN, KEYUP, K_q, K_w, K_e, K_c, \
    K_u, K_i, K_o, K_p


def control_character_new(player):
    """
    対象キャラを操作する
    Parameters
    ----------
    player_move : (list) 対象キャラを移動させる
    action : (list) 攻撃、ガード、掴み
    contact : (list) 操作キャラとステージの接触判定
    [キャラクターと地面との接触判定、頭上のオブジェクトとの接触判定、右側とオブジェクトとの接触判定、左側とオブジェクトの接触判定]
    direction : 0=上,1=した,2=右.3=左

    Returns
    -------
    変更した移動、攻撃の判定を返す

    """
    action_all = player.action_all
    direction = player.direction
    put_action_flag = 0

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == KEYDOWN:
            if pygame.key.get_pressed()[K_w]:
                # 地面に接触してたらジャンプ
                action_all[0] = True if player.canMoveRange[0] == 0 else action_all[0]

            if pygame.key.get_pressed()[K_c]:
                action_all[1] = True

            if pygame.key.get_pressed()[K_q]:
                # 左の接触判定なかったら左に進む
                action_all[3] = True
                direction = 3

            if pygame.key.get_pressed()[K_e]:
                # 右の接触判定なかったら右に進む
                action_all[2] = True
                direction = 2


            if player.rigit_time == 0:
                if pygame.key.get_pressed()[K_u]:
                    action_all[4] = True
                    put_action_flag = 1
                    # print("aaaaaaaaaa")
                # else:
                #     action_all[4] = False
                #     print("bbbbbbbbbb")

                # if pygame.key.get_pressed()[K_i]:
                #     action_all[4] = True

                # if pygame.key.get_pressed()[K_o]:
                #     action_all[4] = True

                # if pygame.key.get_pressed()[K_p]:
                #     action_all[4] = True

        elif event.type == KEYUP:
            if event.key == K_c:
                action_all[1] = False

            if event.key == K_q:
                action_all[3] = False

            if event.key == K_e:
                action_all[2] = False

            if put_action_flag == 0:
                if event.key == K_u:
                    if player.rigit_time == 0:
                        action_all[4] = False
                        # print("ccccccccccc")
                    else:
                        action_all[4] = False
                        # print("ddddddddddd")

            # else:
            #     action_all[4] = False
            #     print("bbbbbbbbb")
            # if event.key == K_i:
            #     action_all[4] = False

            # if event.key == K_o:
            #     action_all[4] = False

            # if event.key == K_p:
            #     action_all[4] = False

    player.action_all = action_all
    # print(action_all)
    # player.action = [True,True,True,False]
    player.direction = direction




def control_character_random(player):
    """
    対象キャラを操作する
    Parameters
    ----------
    player_move : (list) 対象キャラを移動させる
    action : (list) 攻撃、ガード、掴み
    contact : (list) 操作キャラとステージの接触判定
    direction : 0=上,1=した,2=右.3=左

    Returns
    -------
    変更した移動、攻撃の判定を返す

    """
    direction = player.direction

    action_all = [False, False, False, False,False,False]
    # 地面に接触してたらジャンプ
    if player.canMoveRange[0] == 0:
        rand_move = np.random.choice([0,1,2,3,4,5], 1)[0]
    else:
        rand_move = np.random.choice([1,2,3,4,5], 1)[0]


    action_all[rand_move] = True
    if rand_move == 3:
        direction = 3
    elif rand_move == 2:
        direction = 2

    player.action_all = action_all
    # player.contact = contact
    player.direction = direction

    # return player_move, action,direction

# def control_character_leftatack(player):
#     """
#     対象キャラを操作する
#     Parameters
#     ----------
#     player_move : (list) 対象キャラを移動させる
#     action : (list) 攻撃、ガード、掴み
#     contact : (list) 操作キャラとステージの接触判定
#     direction : 0=上,1=した,2=右.3=左

#     Returns
#     -------
#     変更した移動、攻撃の判定を返す

#     """
#     player_move = player.player_move
#     action = player.action
#     direction = player.direction

#     player_move = [False, False, False, False]


#     player_move[2] = True
#     direction = 2

#     action = [False, False, False, False]
#     rand_action = np.random.choice([0,1,2,3], 1)[0]

#     action[0] = True

#     player.player_move = player_move
#     player.action = action
#     # player.contact = contact
#     player.direction = direction

if __name__ == '__main__':
    pass