import pygame

from game_parts import control, fighter, stage


pygame.init()
SURFACE = pygame.display.set_mode((1000, 700))
FPSCLOCK = pygame.time.Clock()
STAGE_POS = [0, 550]  # <= 表示するステージの位置

def main():


    JUMP_SPEED = 30  # <= ジャンプの初速度
    GRAVITY = 16
    MOVE_SPEED = 10

    player1_pos = [600, 470]  # <= 操作キャラの位置
    player1_size = [50, 80]  # <= 操作キャラの大きさ [横幅, 縦幅]
    direction1 =3 #キャラの方向，0=上,1=した,2=右.3=左

    player2_pos = [200, 470]
    player2_size = [50, 80]
    direction2 =2 #キャラの方向，0=上,1=した,2=右.3=左

    player_1 = fighter.Fighter(player1_size, GRAVITY, MOVE_SPEED,JUMP_SPEED,player1_pos,direction1)
    player_2 = fighter.Fighter(player2_size, GRAVITY, MOVE_SPEED,JUMP_SPEED,player2_pos,direction2)

    while True:
        # 背景を真っ白に設定する
        SURFACE.fill((250, 250, 250))
        # 戦闘ステージを作成
        stage.base_stage(STAGE_POS)
        # player_1.contact_judgment(player_2)
        # player_2.contact_judgment(player_1)
        # プレイヤー１のキャラクター操作
        control.control_character(player_1)
        # プレイヤー2のキャラクター操作
        control.control_character_random(player_2)
        # control.control_character_leftatack(player_2)
        player_1.contact_judgment(player_2)
        player_2.contact_judgment(player_1)

        player_1.figure()
        player_2.figure()

        player_1.move()
        player_2.move()

        player_1.contact_judgment(player_2)
        player_2.contact_judgment(player_1)

        # player_1.position_corr(STAGE_POS)
        # player_2.position_corr(STAGE_POS)

        player_1.character_action(player_2)
        player_2.character_action(player_1)

        player_1.hit_action()
        player_2.hit_action()

        player_1.contact_judgment(player_2)
        player_2.contact_judgment(player_1)

        player_1.life("right")
        player_2.life("left")



        # control.control_character_random(player_2)
        # player_2.figure()
        # player_2.move()
        # player_2.contact_judgment(player_1)
        # player_2.position_corr(STAGE_POS)
        # player_2.character_action(player_1)
        # player_2.hit_action()
        # player_2.life("left")


        # # プレイヤー１のキャラクター操作
        # control.control_character(player_1)
        # # プレイヤー１の操作キャラの表示、行動制御
        # # player_1.update(player1_pos, contact1, rigit_time1)
        # player_1.figure()
        # # player_1.contact_judgment(player_2)
        # player_1.move()
        # player_1.contact_judgment(player_2)
        # player_1.position_corr(STAGE_POS)
        # # action1, rigit_time1, hit_judg2, damage2, rigit_time2, blow_speed2 = player_1.character_action(move1, action1, hit_judg2, player2_pos, player2_size, damage2)
        # player_1.character_action(player_2)
        # player_1.contact_judgment(player_2)
        # player_1.hit_action()
        # player_1.life("right")

        # # プレイヤー2のキャラクター操作
        # control.control_character_random(player_2)
        # player_2.figure()
        # player_2.contact_judgment(player_1)
        # player_2.move()
        # player_2.position_corr(STAGE_POS)
        # player_2.character_action(player_1)
        # player_2.contact_judgment(player_1)
        # player_2.hit_action()
        # player_2.life("left")
        # # control.control_character_random(player_2)
        # # player_2.figure()
        # # player_2.move()
        # # player_2.contact_judgment(player_1)
        # # player_2.position_corr(STAGE_POS)
        # # player_2.character_action(player_1)
        # # player_2.hit_action()
        # # player_2.life("left")

        # 画面を更新する
        pygame.display.update()

        # 画面の更新を30fps(1秒間に30枚画面が切り替わる)に設定する
        FPSCLOCK.tick(30)


if __name__ == '__main__':
    main()



