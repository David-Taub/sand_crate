from pathlib import Path
from typing import Optional

import fire

from playback import Playback


def main(config_file_path: Path, play_recording: Optional[Path] = None):
    playback = Playback(config_file_path=config_file_path, recording_dir_path=play_recording)
    if play_recording is not None:
        playback.play_recording()
    else:
        playback.run_live_simulation()


if __name__ == "__main__":
    fire.Fire(main)
