import shutil
import time
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
import zarr as zarr
from nptyping import NDArray
from tqdm.rich import tqdm

from crate.crate import Crate
from crate.load_config import load_config
from crate.utils.pygame_utils import draw_arrow
from crate.utils.types import Particles, Segments

SCROLL_ZOOM_FACTOR = 0.2
TEXT_MARGIN = 6
BACKGROUND_COLOR = pygame.Color(0, 0, 0)
RIGID_BODY_COLOR = pygame.Color(255, 255, 255)
DEBUG_ARROWS_COLOR = pygame.Color(0, 255, 0)
SEGMENT_INDEX_COLOR = pygame.Color(0, 255, 0)
PARTICLE_INDEX_COLOR = pygame.Color(255, 0, 0)
PLAYBACK_PARTICLE_COLOR = pygame.Color(100, 100, 255)
DEBUG_TEXT_COLOR = pygame.Color(255, 255, 255)


class Playback:
    def __init__(self, config_file_path: Path, recording_dir_path: Optional[Path] = None) -> None:
        self.config_file_path = config_file_path
        self.config = load_config(config_file_path=config_file_path)
        if recording_dir_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_dir_path = self.config.playback_config.recording_output_dir_path / f"{timestamp}"
        else:
            self.recording_dir_path = recording_dir_path
        self.crate = Crate(self.config.world_config)
        self.done = False
        self.pause = False
        self.step_one = False
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.Font] = None
        self.current_physical_field_index: int = 0
        self.zoom_center = self.screen_original_center.copy()
        self.zoom_factor = 1.0

    def run_live_simulation(self) -> None:
        self.init_display()
        num_of_ticks = self.config.playback_config.ticks_to_record
        if self.config.playback_config.save_recording:
            particles_recording = np.zeros([num_of_ticks, self.crate.max_particles, 2])
            segments_recording = np.zeros([num_of_ticks] + list(self.crate.segments.shape))
        for frame_index in tqdm(range(num_of_ticks)):
            self.handle_play_control()
            self.handle_input()
            self.crate.physics_tick()
            if self.config.playback_config.save_recording:
                particles_recording[frame_index, : self.crate.particle_count] = self.crate.particles
                segments_recording[frame_index] = self.crate.segments
            self.draw_scene()
            if self.done:
                break

        if self.config.playback_config.save_recording:
            self.save_recording(particles_recording, segments_recording, self.recording_dir_path, self.config_file_path)
            self.play_recording()
        pygame.quit()

    def init_display(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("SandCrate")
        self.screen = pygame.display.set_mode(
            (self.config.playback_config.screen_x, self.config.playback_config.screen_y))
        self.font = pygame.font.SysFont("monospace", self.config.playback_config.screen_x // 60)

    def draw_scene(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_particles(self.crate.particles, self.crate.particle_radius, self.crate.particles_pressure,
                            show_indices=False)
        self.draw_segments(self.crate.segments)
        self.draw_debug_arrows()
        self.draw_debug_text(self.crate.debug_prints)
        pygame.display.update()

    def handle_play_control(self) -> None:
        while self.pause and not self.done:
            self.handle_input()
            time.sleep(0.01)
            if self.step_one:
                self.step_one = False
                return

    def draw_debug_arrows(self):
        for start, direction in self.crate.debug_arrows:
            if np.isnan(start).any() or np.isnan(direction).any():
                continue
            direction = direction / np.power(np.linalg.norm(direction) + 0.001, 0.3)
            draw_arrow(
                self.screen,
                color=DEBUG_ARROWS_COLOR,
                start=self.crate_to_screen_coord(*start),
                end=self.crate_to_screen_coord(*(start + direction)),
                head_width=4,
                head_height=2
            )

    @staticmethod
    def save_recording(particles_recording: NDArray, segments_recording: NDArray,
                       recording_output_dir_path: Path, config_file_path: Path) -> None:
        zarr.save(str(Path(recording_output_dir_path) / "particles"), particles_recording)
        zarr.save(str(Path(recording_output_dir_path) / "segments"), segments_recording)
        shutil.copy(config_file_path, recording_output_dir_path)

    def play_recording(self) -> None:
        particles_recording = zarr.load(str(self.recording_dir_path / "particles"))
        segments_recording = zarr.load(str(self.recording_dir_path / "segments"))
        self.init_display()
        while not self.done:
            for particles, segments in zip(particles_recording, segments_recording):
                self.screen.fill(BACKGROUND_COLOR)
                self.handle_input()
                self.draw_particles(particles, self.crate.particle_radius)
                self.draw_segments(segments)
                pygame.display.update()
                if self.done:
                    break
        pygame.quit()

    def handle_input(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEWHEEL:
                self.scale_zoom(event.y)
                self.draw_scene()
            if event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:
                    self.translate(pygame.Vector2(*event.rel))
                    self.draw_scene()
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
                    self.zoom_center = self.screen_original_center
                    self.zoom_factor = 1.0
                if event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                if event.key == pygame.K_n:
                    self.step_one = True
            if event.type == pygame.KEYUP:
                self.crate.gravity = np.array([0.0, 9.81])

    def reset(self):
        self.crate = Crate(self.config.world_config)

    def draw_segments(self, segments: Segments, show_indices: bool = False) -> None:
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

    def draw_particles(
            self, particles: Particles, particle_radius: float, particles_color: Optional[NDArray] = None,
            show_indices: bool = False
    ) -> None:
        particle_radius = int(self.config.playback_config.screen_x * particle_radius) * self.zoom_factor
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

    def crate_to_screen_coord(self, x: float, y: float) -> pygame.Vector2:
        screen_coordination = pygame.Vector2(int(x * (self.config.playback_config.screen_x - 1)),
                                             int(y * (self.config.playback_config.screen_y - 1)))
        screen_coordination = ((screen_coordination - self.zoom_center) * self.zoom_factor) + \
                              self.screen_original_center
        return screen_coordination

    def draw_debug_text(self, text: str) -> None:
        for line, line_text in enumerate(text.split("\n")):
            text_surface = self.font.render(line_text, True, DEBUG_TEXT_COLOR)

            self.screen.blit(text_surface, (TEXT_MARGIN, TEXT_MARGIN + line * self.font.get_linesize()))

    def edit_physics(self, increase: bool, change_factor: float = 0.1) -> None:
        coefficients = self.crate.editable_coefficients()
        coefficient = coefficients[self.current_physical_field_index % len(coefficients)]
        current_value = getattr(self.crate, coefficient)
        change_rate = 1 + change_factor if increase else 1 - change_factor
        setattr(self.crate, coefficient, current_value * change_rate)

    def translate(self, relative_motion: pygame.Vector2) -> None:
        self.zoom_center -= relative_motion / self.zoom_factor

    def scale_zoom(self, scale_direction: int) -> None:
        """
        makes the point under the mouse stay at the same place after zoom
        """
        mouse_pos = pygame.Vector2(*pygame.mouse.get_pos())
        new_zoom_factor = self.zoom_factor + self.zoom_factor * scale_direction * SCROLL_ZOOM_FACTOR
        zooms_ratio = new_zoom_factor / self.zoom_factor
        zoom_center_target = (1 - 1 / zooms_ratio) * mouse_pos + (1 / zooms_ratio) * self.screen_original_center

        self.zoom_factor = new_zoom_factor
        self.zoom_center = (self.zoom_center + (zoom_center_target - self.screen_original_center) / self.zoom_factor)

    @cached_property
    def screen_original_center(self) -> pygame.Vector2:
        return pygame.Vector2(self.config.playback_config.screen_x, self.config.playback_config.screen_y) / 2
