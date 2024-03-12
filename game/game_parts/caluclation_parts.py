import math


def pythagoras_theorem(a1, a2, b1, b2):
    z_2 = (a2 - a1) ** 2 + (b2 - b1) ** 2
    z = math.sqrt(z_2)

    return z
def calcMovingSize(can,want):
    if can <= want:
        return can
    else:
        return want