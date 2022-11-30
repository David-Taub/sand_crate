import numpy as np
import pygame

from crate import Crate, R, TARGET_FRAME_RATE

SCREEN_X = 500
SCREEN_Y = 500


class GameGUI:
    screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y))
    pygame.display.set_caption("SandCrate")
    clock = pygame.time.Clock()

    def __init__(self, crate: Crate):
        self.crate = crate

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.crate.gen_particles()
                if event.key == pygame.K_RIGHT:
                    self.g = np.array([9.81, 0.0])
                if event.key == pygame.K_LEFT:
                    self.g = np.array([-9.81, 0.0])
                if event.key == pygame.K_q:
                    self.done = True
            if event.type == pygame.KEYUP:
                self.g = np.array([0.0, 9.81])
            if event.type == pygame.QUIT:
                self.done = True

    def display_particles(self):
        BLACK = (0, 0, 0)
        p_rad = int(SCREEN_X * R)
        self.screen.fill(BLACK)
        particles = self.crate.particles

        particles[:, 5] /= np.max(particles[:, 5])
        for i in range(particles.shape[0]):
            center = self.game_to_screen_coord(particles[i, 0], particles[i, 1])
            color = (255 - int(particles[i, 5] * 255), 255 - int(particles[i, 5] * 255), 255)
            pygame.draw.circle(self.screen, color, center, p_rad)
        pygame.display.update()

    @staticmethod
    def game_to_screen_coord(x: float, y: float):
        return int(x * (SCREEN_X - 1)), int(y * (SCREEN_Y - 1))

    def run_main_loop(self):
        self.done = False
        while not self.done:
            self.clock.tick(TARGET_FRAME_RATE)
            self.handle_input()
            self.crate.physics_tick()
            self.display_particles()
        pygame.quit()
