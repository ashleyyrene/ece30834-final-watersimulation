import pygame
import numpy as np

from settings import WIDTH, HEIGHT, GRID_W, GRID_H
from water import (
    create_lily_pads,
    create_damping_map,
    add_ripple,
    update_waves,
    render_water
)

def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Interactive Water Simulation")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # create 2D height field grid
    current = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    previous = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    lily_pads = create_lily_pads()
    damping_map = create_damping_map()

    previous_mouse = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                add_ripple(current, previous, mouse_x, mouse_y, strength=20.0)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current[:, :] = 0
                    previous[:, :] = 0

        if pygame.mouse.get_pressed()[0]:
            mouse_x, mouse_y = pygame.mouse.get_pos()

            # control speed of ripples
            if previous_mouse is not None:
                dx = mouse_x - previous_mouse[0]
                dy = mouse_y - previous_mouse[1]
                speed = (dx ** 2 + dy ** 2) ** 0.5      # distance formula
                strength = min(25.0, max(8.0, speed * 0.8))     # ripple strength

                add_ripple(current, previous, mouse_x, mouse_y, strength * 2.5)

            previous_mouse = (mouse_x, mouse_y)
        else:
            previous_mouse = None

        current, previous = update_waves(
            current,
            previous,
            damping_map,
            lily_pads
        )

        render_water(screen, current, lily_pads, pygame.time.get_ticks() / 1000)

        fps_text = font.render(
            f"FPS: {int(clock.get_fps())} | Click/drag = ripples | Space = clear",
            True,
            (255, 255, 255)
        )

        screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
