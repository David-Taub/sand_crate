import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
import zarr as zarr
from nptyping import NDArray

from crate.crate import Crate
from crate.load_config import CONFIG_FILE_PATH, load_config
from crate.utils.types import Particles, Segments
from src.crate.utils.pygame_utils import draw_arrow

SCREEN_X = 1000
SCREEN_Y = 1000
TEXT_MARGIN = 6
BACKGROUND_COLOR = pygame.Color(0, 0, 0)
RIGID_BODY_COLOR = pygame.Color(255, 255, 255)
FORCES_COLOR = pygame.Color(0, 255, 0)
SEGMENT_INDEX_COLOR = pygame.Color(0, 255, 0)
PARTICLE_INDEX_COLOR = pygame.Color(255, 0, 0)
PLAYBACK_PARTICLE_COLOR = pygame.Color(100, 100, 255)
DEBUG_TEXT_COLOR = pygame.Color(255, 255, 255)


class Playback:
    def __init__(self, crate: Crate) -> None:
        self.crate = crate
        self.done = False
        self.pause = False
        self.step_one = False
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.Font] = None
        self.current_physical_field_index: int = 0

    def init_display(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("SandCrate")
        self.screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y))
        self.font = pygame.font.SysFont("monospace", SCREEN_X // 60)

    def show_particle_forces(self, particle_index: int) -> None:
        particle_position = self.crate.particles[particle_index]
        particle_velocity = self.crate.particle_velocities[particle_index]
        draw_arrow(
            self.screen,
            color=FORCES_COLOR,
            start=self.crate_to_screen_coord(*particle_position),
            end=self.crate_to_screen_coord(*(particle_position + particle_velocity))
        )

    def run_live_simulation(self, num_of_ticks: int, recording_output_dir_path: Path, save_recording: bool) -> None:
        self.init_display()
        if save_recording:
            particles_recording = np.zeros([num_of_ticks, self.crate.max_particles, 2])
            segments_recording = np.zeros([num_of_ticks] + list(self.crate.segments.shape))
        recording_frame_index = 0
        while not self.done:
            self.handle_play_control()
            self.handle_input()

            self.crate.physics_tick()
            if recording_frame_index < num_of_ticks and save_recording:
                particles_recording[recording_frame_index, : self.crate.particle_count] = self.crate.particles
                segments_recording[recording_frame_index] = self.crate.segments
            self.screen.fill(BACKGROUND_COLOR)
            self.display_particles(self.crate.particles, self.crate.particle_radius, self.crate.particles_pressure,
                                   show_indices=True)
            self.display_segments(self.crate.segments)
            self.display_debug_text(self.crate.debug_prints)
            pygame.display.update()
        if save_recording:
            self.save_recording(particles_recording, segments_recording, recording_output_dir_path)
            shutil.copy(CONFIG_FILE_PATH, recording_output_dir_path)
        pygame.quit()

    def handle_play_control(self) -> None:
        while self.pause and not self.done:
            self.handle_input()
            time.sleep(0.01)
            if self.step_one:
                self.step_one = False
                return

    def save_recording(self, particles_recording: NDArray, segments_recording: NDArray,
                       recording_output_dir_path: Path) -> None:
        zarr.save(str(Path(recording_output_dir_path) / "particles"), particles_recording)
        zarr.save(str(Path(recording_output_dir_path) / "segments"), segments_recording)
        shutil.copy(CONFIG_FILE_PATH, recording_output_dir_path)

    def show_recording(self, recording_dir_path: Path) -> None:
        particles_recording = zarr.load(str(recording_dir_path / "particles"))
        segments_recording = zarr.load(str(recording_dir_path / "segments"))
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
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                if event.key == pygame.K_n:
                    self.step_one = True
            if event.type == pygame.KEYUP:
                self.crate.gravity = np.array([0.0, 9.81])

    def reset(self):
        self.crate = Crate(load_config().world_config)

    def display_segments(self, segments: Segments, show_indices: bool = False) -> None:
        for i, segment in enumerate(segments):
            pygame.draw.line(
                self.screen,
                RIGID_BODY_COLOR,
                self.crate_to_screen_coord(*segment[0]),
                self.crate_to_screen_coord(*segment[1]),
                width=2,
            )
            if show_indices:
                self.screen.blit(self.font.render(str(i), True, SEGMENT_INDEX_COLOR),
                                 self.crate_to_screen_coord(*segment[0]))

    def display_particles(
            self, particles: Particles, particle_radius: float, particles_color: Optional[NDArray] = None,
            show_indices: bool = False
    ) -> None:
        particle_radius = int(SCREEN_X * particle_radius)
        for i in range(particles.shape[0]):
            particle_center = self.crate_to_screen_coord(particles[i, 0], particles[i, 1])
            if particles_color is not None:
                particle_color = (255 - int(particles_color[i] * 255), 255 - int(particles_color[i] * 255), 255)
            else:
                particle_color = PLAYBACK_PARTICLE_COLOR
            particle_color = np.clip(particle_color, 0, 255)
            pygame.draw.circle(self.screen, particle_color, particle_center, particle_radius)
            if show_indices:
                self.screen.blit(self.font.render(str(i), True, PARTICLE_INDEX_COLOR),
                                 (particle_center[0] - 5, particle_center[1] - 8))

    @staticmethod
    def crate_to_screen_coord(x: float, y: float) -> pygame.Vector2:
        return pygame.Vector2(int(x * (SCREEN_X - 1)), int(y * (SCREEN_Y - 1)))

    def display_debug_text(self, text: str) -> None:
        for line, line_text in enumerate(text.split("\n")):
            text_surface = self.font.render(line_text, True, DEBUG_TEXT_COLOR)

            self.screen.blit(text_surface, (TEXT_MARGIN, TEXT_MARGIN + line * self.font.get_linesize()))

    def edit_physics(self, increase: bool, change_factor: float = 0.1) -> None:
        coefficients = self.crate.editable_coefficients()
        coefficient = coefficients[self.current_physical_field_index % len(coefficients)]
        current_value = getattr(self.crate, coefficient)
        change_rate = 1 + change_factor if increase else 1 - change_factor
        setattr(self.crate, coefficient, current_value * change_rate)
