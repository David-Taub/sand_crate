from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class WorldConfig:
    rigid_bodies: list
    particle_sources: list
    coefficients: dict


@dataclass
class PlaybackConfig:
    save_recording: bool
    ticks_to_record: bool
    recording_output_dir_path: Path
    screen_x: int
    screen_y: int


@dataclass
class Config:
    world_config: WorldConfig
    playback_config: PlaybackConfig


def load_config(config_file_path: Path) -> Config:
    with open(config_file_path, "r") as f:
        raw_config = yaml.safe_load(f)
    raw_world_config = raw_config["world"]
    world_config = WorldConfig(
        rigid_bodies=raw_world_config.get("rigid_bodies", []),
        particle_sources=raw_world_config.get("particle_sources"),
        coefficients=raw_world_config.get("coefficients"),
    )
    playback_config = raw_config["playback"]
    playback_config = PlaybackConfig(
        save_recording=playback_config["save_recording"],
        ticks_to_record=playback_config["ticks_to_record"],
        recording_output_dir_path=Path(playback_config["recording_output_dir_path"]),
        screen_x=playback_config["screen_x"],
        screen_y=playback_config["screen_y"]
    )
    return Config(world_config=world_config, playback_config=playback_config)


