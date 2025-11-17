import gymnasium as gym
import numpy as np
import pygame
import sys

# Touches
STEER_LEFT = pygame.K_q
STEER_RIGHT = pygame.K_d
ACCELERATE = pygame.K_z
BRAKE = pygame.K_s
QUIT = pygame.K_ESCAPE

# Tableau d'actions discrètes
ACTIONS = {
    0: np.array([0.0, 0.0, 0.0]),   # rien
    1: np.array([-1.0, 0.0, 0.0]),  # gauche
    2: np.array([1.0, 0.0, 0.0]),   # droite
    3: np.array([0.0, 1.0, 0.0]),   # accélérer
    4: np.array([0.0, 0.0, 1.0]),   # freiner
    5: np.array([-1.0, 1.0, 0.0]),  # gauche + accélérer
    6: np.array([1.0, 1.0, 0.0]),   # droite + accélérer
    7: np.array([-1.0, 0.0, 1.0]),  # gauche + frein
    8: np.array([1.0, 0.0, 1.0])    # droite + frein
}

def get_discrete_action(keys):
    """Retourne une action DISCRÈTE selon les touches appuyées"""

    left  = keys[STEER_LEFT]
    right = keys[STEER_RIGHT]
    gas   = keys[ACCELERATE]
    brake = keys[BRAKE]

    if left and gas:
        return 5
    if right and gas:
        return 6
    if left and brake:
        return 7
    if right and brake:
        return 8

    if left:
        return 1
    if right:
        return 2
    if gas:
        return 3
    if brake:
        return 4

    return 0  # rien


def main():
    pygame.init()
    pygame.display.set_caption("CarRacing Discrete Controls (ZQSD)")

    env = gym.make("CarRacing-v3", render_mode="human")
    obs, info = env.reset(seed=0)

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action_id = get_discrete_action(keys)
        action = ACTIONS[action_id]

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
