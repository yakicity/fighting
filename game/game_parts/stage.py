import pygame

import main as main


def base_stage(position):
    """
    戦闘用ステージ
    Parameters
    ----------
    position : 対象キャラの位置

    """
    pygame.draw.rect(main.SURFACE, (0, 200, 100), (position[0], position[1], 1500, 50))
