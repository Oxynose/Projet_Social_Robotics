import gymnasium as gym
import numpy as np
import pygame
import sys

def main():
    pygame.init()
    pygame.joystick.init()

    # vérifier qu'une manette est connectée
    if pygame.joystick.get_count() == 0:
        print("Aucune manette détectée.")
        sys.exit()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Manette détectée :", joystick.get_name())

    # environment
    env = gym.make("CarRacing-v3", render_mode="human")
    obs, info = env.reset(seed=0)

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ---- RÉCUPÉRATION DES AXES DE LA MANETTE ----
        # AXE 0 : stick gauche horizontal (steering)
        steer = joystick.get_axis(0)   # déjà entre -1 et +1

        # AXE 5 : gâchette droite (RT) – souvent de 0 à 1
        # AXE 4 : gâchette gauche (LT)
        # selon les manettes : parfois de -1 à 1 → donc normalisation
        rt = joystick.get_axis(5)
        lt = joystick.get_axis(4)

        # normaliser si les gâchettes renvoient [-1,1]
        # on les convertit en [0,1]
        gas = (rt + 1) / 2 if rt < 0 or rt > 0 else 0
        brake = (lt + 1) / 2 if lt < 0 or lt > 0 else 0

        # BOUTON START pour quitter (Xbox = 7)
        if joystick.get_button(7):
            running = False

        # ---- ACTION DU JEU ----
        action = np.array([steer, gas, brake], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
