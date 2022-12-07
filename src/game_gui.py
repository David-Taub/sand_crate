from pathlib import Path
from typing import Optional

import numpy as np
import pygame
import zarr as zarr
from nptyping import NDArray
from tqdm import tqdm

from crate import Crate, PARTICLE_RADIUS
from typings import Particles

SCREEN_X = 1000
SCREEN_Y = 1000

BACKGROUND_COLOR = (0, 0, 0)


class GameGUI:
    def __init__(self, crate: Crate) -> None:
        self.crate = crate
        self.done = False

    def init_display(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("SandCrate")
        self.screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", SCREEN_X // 50)

    def run_live_simulation(self) -> None:
        self.init_display()
        while not self.done:
            # self.clock.tick(TARGET_FRAME_RATE)
            self.handle_input()
            self.crate.physics_tick()

            self.screen.fill(BACKGROUND_COLOR)
            self.display_segments(self.crate.segments)
            self.display_particles(self.crate.particles, self.crate.particles_pressure)
            self.display_debug(self.crate.debug_prints)
            pygame.display.update()
        pygame.quit()

    def record_simulation(self, num_of_ticks: int = 2000,
                          recording_output_file_path: Path = Path("recording.zarr")) -> None:
        particles_recording = np.zeros([num_of_ticks] + list(self.crate.particles.shape))
        for i in tqdm(range(num_of_ticks), desc="Simulating"):
            self.crate.physics_tick()
            particles_recording[i] = self.crate.particles
        zarr.save(str(recording_output_file_path), particles_recording)

    def show_recording(self, recording_file_path: Path = Path("recording.zarr")) -> None:
        particles_recording = zarr.load(str(recording_file_path))
        self.init_display()
        while not self.done:
            for particles in particles_recording:
                # self.clock.tick(TARGET_FRAME_RATE)
                self.screen.fill(BACKGROUND_COLOR)
                self.handle_input()
                self.display_particles(particles)
                self.display_debug(self.crate.debug_prints)
                pygame.display.update()
                if self.done:
                    break
        pygame.quit()

    def handle_input(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.crate.gravity = np.array([9.81, 0.0])
                if event.key == pygame.K_LEFT:
                    self.crate.gravity = np.array([-9.81, 0.0])
                if event.key == pygame.K_q:
                    self.done = True
            if event.type == pygame.KEYUP:
                self.crate.gravity = np.array([0.0, 9.81])

    def display_segments(self, segments) -> None:
        for segment in segments:
            print(segment)

    def display_particles(self, particles: Particles, particles_color: Optional[NDArray] = None) -> None:
        particle_radius = int(SCREEN_X * PARTICLE_RADIUS)
        for i in range(particles.shape[0]):
            particle_center = self.crate_to_screen_coord(particles[i, 0], particles[i, 1])
            if particles_color is not None:
                particle_color = (
                    255 - int(particles_color[i] * 255), 255 - int(particles_color[i] * 255),
                    255)
            else:
                particle_color = (100, 100, 255)
            particle_color = np.clip(particle_color, 0, 255)
            pygame.draw.circle(self.screen, particle_color, particle_center, particle_radius)

    @staticmethod
    def crate_to_screen_coord(x: float, y: float) -> tuple[int, int]:
        return int(x * (SCREEN_X - 1)), int(y * (SCREEN_Y - 1))

    def display_debug(self, text: str):
        for line, line_text in enumerate(text.split("\n")):
            text_surface = self.font.render(line_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (0, line * self.font.get_linesize()))
