from itertools import product
from pathlib import Path
from typing import Optional

import fire

from playback import Playback
from src.crate.load_config import load_config, Config

options = {
    "pressure_amplifier": [20, 40],
    "ignored_pressure": [0.3, 0.1],
    "viscosity": [4, 8],
    "surface_smoothing": [40, 100],
    "target_pressure": [-5, -2, 2],
}


def main(config_file_path: Path, play_recording: Optional[Path] = None):
    config = load_config(config_file_path=config_file_path)
    for config_variant in config_options(options, config):
        playback = Playback(config=config_variant, recording_dir_path=play_recording)
        playback.run_live_simulation()


def config_options(options: dict, config: Config):
    vals = []
    for k, vs in options.items():
        key_vals = []
        for v in vs:
            key_vals.append((k, v))
        vals.append(key_vals)
    for kvs in product(*vals):
        for k, v in kvs:
            config.world_config.coefficients[k] = v
        yield config


if __name__ == "__main__":
    fire.Fire(main)
