
import pygame
import time
import random
import numpy as np
# import numpy.linalg


def rand_vec():
    return np.random.rand(2)


R = 0.03
DIAMETER = R * 2
PRESSURE_AMPLIFIER = 1
PARTICLE_MASS = 0.01
IGNORED_PRESSURE = 0.4
VISCOUSITY = 0.4
ELASTICITY = 0.5
TENSILE_ALPHA = 3
TENSILE_BETA = 0.01


class Particle(object):

    def __init__(self, pos=None, velocity=None):
        self.pos = rand_vec() if pos is None else pos
        self.velocity = (rand_vec() - 0.5) * 0.05 if velocity is None else velocity

    def calc_pressure(self, colliders):
        self.relative_vecs = [self.pos - collider.pos for collider in colliders]
        self.relative_vecs = [v if np.linalg.norm(v) > 0 else (np.random.rand(2) - 0.5) * 2 * DIAMETER for i, v in enumerate(self.relative_vecs)]
        self.colliders_overlap = [1 - (np.linalg.norm(v) / DIAMETER) for v in self.relative_vecs]
        self.total_overlap = sum(self.colliders_overlap)
        self.pressure = max(0, PRESSURE_AMPLIFIER * (self.total_overlap - IGNORED_PRESSURE))
        self.normalized_relative_vecs = [v / np.linalg.norm(v) for v in self.relative_vecs]
        self.tensile_b = sum([v * (1 - self.colliders_overlap[i]) * self.colliders_overlap[i] for i, v in enumerate(self.normalized_relative_vecs)])

    def apply_gravity(self, dt, g):
        self.velocity += g * PARTICLE_MASS * dt

    def apply_pressure(self, dt, colliders):
        for i, collider in enumerate(colliders):
            self.velocity += (dt * (self.pressure + collider.pressure) *
                              self.colliders_overlap[i] * self.normalized_relative_vecs[i])

    def apply_tensile(self, dt, colliders):
        for i, collider in enumerate(colliders):
            a = TENSILE_ALPHA * (self.total_overlap + collider.total_overlap - 2 * IGNORED_PRESSURE)
            b = TENSILE_BETA * (np.dot(collider.tensile_b - self.tensile_b, self.normalized_relative_vecs[i]))
            self.velocity += self.normalized_relative_vecs[i] * dt * (a + b)

    def apply_viscous(self, dt, colliders):
        for i, collider in enumerate(colliders):
            self.velocity += VISCOUSITY * dt * (collider.velocity - self.velocity)

    def update(self, dt, g, colliders):
        self.apply_gravity(dt, g)
        self.apply_pressure(dt, colliders)
        self.apply_tensile(dt, colliders)
        self.apply_viscous(dt, colliders)
        self.pos += self.velocity * dt
        # stay in box
        if self.pos[0] < 0:
            self.pos[0] = 0
            self.velocity[0] *= -ELASTICITY
        if self.pos[0] > 1:
            self.pos[0] = 1
            self.velocity[0] *= -ELASTICITY
        if self.pos[1] < 0:
            self.pos[1] = 0
            self.velocity[1] *= -ELASTICITY
        if self.pos[1] > 1:
            self.pos[1] = 1
            self.velocity[1] *= -ELASTICITY

        # if self.pos[1] == 1:
        #     # hit the floor
        #     self.velocity = [0, 0]

# TODO: finish, sort by floor y, then sort by floor x, then search for particles in neighbors cells
# see https://docs.google.com/presentation/d/1fEAb4-lSyqxlVGNPog3G1LZ7UgtvxfRAwR0dwd19G4g/edit#slide=id.p
# def detect_collisions_nlogn(particles):
#     ys = np.floor([particle.pos[1] / particle.effect_radius for particle in particles])
#     particles = [particles[i] for i in np.argsort(ys)]
#     ys = np.floor([particle.pos[1] / particle.effect_radius for particle in particles])
#     strip = []
#     cur_y_val = ys[0]
#     for i in range(len(ys)):
#         if ys[i] == cur_y_val:
#             strip.append(particles[i])
#         else:
#             ys = np.floor([particle.pos[1] / particle.effect_radius for particle in particles])
#             particles = [particles[i] for i in np.argsort(ys)]
#             ys = np.floor([particle.pos[1] / particle.effect_radius for particle in particles])
#             strip = []
#             cur_y_val = ys[i]
#     unique_vals = np.unique(ys)
#     for unique_val in unique_vals:
#         ys == unique_val
#     xs = np.floor([particle.pos[1] / particle.effect_radius for particle in particles])

# particles - list of particles
# returns a dict of indices to a list of their colliding particle indices
def detect_collisions(particles):
    colliding = {}
    n = len(particles)
    for i in range(n):
        colliding[i] = []
    for i in range(n):
        for j in range(i + 1, n):
            particle_i = particles[i]
            particle_j = particles[j]
            distance = np.linalg.norm(particle_i.pos - particle_j.pos)
            if distance < DIAMETER:
                colliding[i].append(j)
                colliding[j].append(i)
    return colliding




class Crate(object):
    g = np.array([0.0, 9.81])
    ticks_per_frame = 1
    target_frame_rate = 60
    particle_count = 100

    SCREEN_X = 500
    SCREEN_Y = 500

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
                    self.g[0] += 0.5
                    self.g[1] -= 0.5
                if event.key == pygame.K_LEFT:
                    self.g[0] -= 0.5
                    self.g[1] += 0.5
                if event.key == pygame.K_q:
                    self.done = True
            if event.type == pygame.QUIT:
                self.done = True

    def gen_particles(self):
        self.particles = [Particle() for i in range(self.particle_count)]

    def physics_tick(self):
        colliding = detect_collisions(self.particles)
        for i, particle in enumerate(self.particles):
            colliders = [self.particles[j] for j in colliding[i]]
            particle.calc_pressure(colliders)
        for i, particle in enumerate(self.particles):
            colliders = [self.particles[j] for j in colliding[i]]
            particle.update(0.04, self.g, colliders)

    def display_particles(self):
        buffer = np.zeros((self.SCREEN_X, self.SCREEN_Y))
        p_rad = int(self.SCREEN_X * R)
        for particle in self.particles:
            center = self.get_particle_location(particle)
            buffer[center[0] - p_rad: center[0] + p_rad, center[1] - p_rad: center[1] + p_rad] = 1
        buffer = 255 * buffer / buffer.max()
        surf = pygame.surfarray.make_surface(buffer)
        self.screen.blit(surf, (0, 0))
        pygame.display.update()

    def run_main_loop(self):
        self.done = False
        while not self.done:
            self.handle_input()
            self.clock.tick(self.target_frame_rate)
            for i in range(self.ticks_per_frame):
                self.physics_tick()
            self.display_particles()
        pygame.quit()



crate = Crate()
crate.run_main_loop()
particles = crate.particles