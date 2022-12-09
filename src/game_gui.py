from pathlib import Path
from typing import Optional

import numpy as np
import pygame
import zarr as zarr
from nptyping import NDArray
from tqdm import tqdm

from crate import Crate
from typings import Particles

SCREEN_X = 1000
SCREEN_Y = 1000
TEXT_MARGIN = 6
BACKGROUND_COLOR = (0, 0, 0)
LINE_COLOR = (255, 255, 255)


class GameGUI:
    def __init__(self, crate: Crate) -> None:
        self.crate = crate
        self.done = False
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.Font] = None
        self.current_physical_field_index: int = 0

    def init_display(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("SandCrate")
        self.screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y))
        self.font = pygame.font.SysFont("monospace", SCREEN_X // 60)

    def run_live_simulation(self) -> None:
        self.init_display()
        while not self.done:
            self.handle_input()
            self.crate.physics_tick()

            self.screen.fill(BACKGROUND_COLOR)
            self.display_particles(self.crate.particles, self.crate.particle_radius, self.crate.particles_pressure)
            self.display_segments(self.crate.segments)
            self.display_debug(self.crate.debug_prints)
            pygame.display.update()
        pygame.quit()

    def record_simulation(self, num_of_ticks: int, recording_output_dir_path: str) -> None:
        particles_recording = np.zeros([num_of_ticks, self.crate.max_particles, 2])
        segments_recording = np.zeros([num_of_ticks] + list(self.crate.segments.shape))
        for i in tqdm(range(num_of_ticks), desc="Simulating"):
            self.crate.physics_tick()
            particles_recording[i, : self.crate.particle_count] = self.crate.particles
            segments_recording[i] = self.crate.segments
        zarr.save(str(Path(recording_output_dir_path) / "particles"), particles_recording)
        zarr.save(str(Path(recording_output_dir_path) / "segments"), segments_recording)

    def show_recording(self, recording_dir_path: str) -> None:
        particles_recording = zarr.load(str(Path(recording_dir_path) / "particles"))
        segments_recording = zarr.load(str(Path(recording_dir_path) / "segments"))
        self.init_display()
        while not self.done:
            for particles, segments in zip(particles_recording, segments_recording):
                self.screen.fill(BACKGROUND_COLOR)
                self.handle_input()
                self.display_particles(particles, self.crate.particle_radius)
                self.display_segments(segments)
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
                if event.key == pygame.K_w:
                    self.current_physical_field_index -= 1
                if event.key == pygame.K_s:
                    self.current_physical_field_index += 1
                if event.key == pygame.K_a:
                    self.edit_physics(increase=False)
                if event.key == pygame.K_d:
                    self.edit_physics(increase=True)
            if event.type == pygame.KEYUP:
                self.crate.gravity = np.array([0.0, 9.81])

    def display_segments(self, segments) -> None:
        for segment in segments:
            pygame.draw.line(
                self.screen,
                LINE_COLOR,
                self.crate_to_screen_coord(*segment[0]),
                self.crate_to_screen_coord(*segment[1]),
                width=2,
            )

    def display_particles(
            self, particles: Particles, particle_radius: float, particles_color: Optional[NDArray] = None
    ) -> None:
        particle_radius = int(SCREEN_X * particle_radius)
        for i in range(particles.shape[0]):
            particle_center = self.crate_to_screen_coord(particles[i, 0], particles[i, 1])
            if particles_color is not None:
                particle_color = (255 - int(particles_color[i] * 255), 255 - int(particles_color[i] * 255), 255)
            else:
                particle_color = (100, 100, 255)
            particle_color = np.clip(particle_color, 0, 255)
            pygame.draw.circle(self.screen, particle_color, particle_center, particle_radius)

    @staticmethod
    def crate_to_screen_coord(x: float, y: float) -> tuple[int, int]:
        return int(x * (SCREEN_X - 1)), int(y * (SCREEN_Y - 1))

    def display_debug(self, text: str) -> None:
        for line, line_text in enumerate(text.split("\n")):
            text_surface = self.font.render(line_text, True, (255, 255, 255))

            self.screen.blit(text_surface, (TEXT_MARGIN, TEXT_MARGIN + line * self.font.get_linesize()))

    def edit_physics(self, increase: bool, change_factor: float = 0.1) -> None:
        physical_fields = [
            "particle_radius",
            "dt",
            "wall_collision_decay",
            "spring_overlap_balance",
            "spring_amplifier",
            "pressure_amplifier",
            "ignored_pressure",
            "collider_noise_level",
            "viscosity",
            "max_particles",
        ]
        physical_field = physical_fields[self.current_physical_field_index % len(physical_fields)]
        current_value = getattr(self.crate, physical_field)
        change_rate = 1 + change_factor if increase else 1 - change_factor
        setattr(self.crate, physical_field, current_value * change_rate)
