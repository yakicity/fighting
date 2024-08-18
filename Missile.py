import pygame

# ミサイルオブジェクトを定義
class Missile:
    def __init__(self, x, y, direction,speed):
        self.x = x
        self.y = y
        self.speed = speed * direction  # ミサイルの速度と方向
        self.active = True

    def move(self):
        self.x += self.speed

    def draw(self, screen):
        if self.active:
            pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 5)

    def check_collision(self, player_x, player_y, player_width, player_height):
        if self.active:
            if player_x < self.x < player_x + player_width and player_y < self.y < player_y + player_height:
                self.active = False
                return True
            if self.x < 0 or self.x > 1000:
                self.active = False
        return False