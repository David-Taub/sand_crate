import pygame
import time
import random
import numpy as np


#%%

def rand_vec():
    return np.random.rand(2)


class Particle(object):

    def __init__(self, pos=None, velocity=None):
        self.pos = rand_vec() if pos is None else pos
        self.velocity = (rand_vec() - 0.5)* 0.005 if velocity is None else velocity

    def update(self, dt, particles):
        self.velocity[1] += dt * g
        self.pos += self.velocity

        # stay in box
        self.pos = np.maximum(self.pos, [0, 0])
        self.pos = np.minimum(self.pos, [1, 1])

        if self.pos[1] == 1:
            # hit the floor
            self.velocity = [0, 0]


class Crate(object):
    g = 9.81
    time_pace = 0.00000001
    target_frame_rate = 60
    particle_count = 200

    SCREEN_X = 640
    SCREEN_Y = 480

    screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y))
    pygame.display.set_caption("SandCrate")
    clock = pygame.time.Clock()

    def __init__(self):
        self.gen_particles()

    def get_particle_location(self, particle):
        return int(particle.pos[0] * (self.SCREEN_X-1)), int(particle.pos[1] * (self.SCREEN_Y-1))

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.gen_particles()
                if event.key == pygame.K_RIGHT:
                    self.time_pace *= 10
                if event.key == pygame.K_LEFT:
                    self.time_pace /= 10
                if event.key == pygame.K_q:
                    self.done = True
            if event.type == pygame.QUIT:
                self.done = True

    def gen_particles(self):
        self.particles = [Particle() for i in range(self.particle_count)]

    def run_main_loop(self):
        self.done = False
        while not self.done:
            self.handle_input()
            dt = self.clock.tick(self.target_frame_rate) * self.time_pace
            buffer = np.zeros((self.SCREEN_X, self.SCREEN_Y))
            [particle.update(dt) for particle in self.particles]
            for particle in self.particles:
                buffer[self.get_particle_location(particle)] = 1
            buffer = 255 * buffer / buffer.max()
            surf = pygame.surfarray.make_surface(buffer)
            self.screen.blit(surf, (0, 0))
            pygame.display.update()

        pygame.quit()


crate = Crate()
crate.run_main_loop()
