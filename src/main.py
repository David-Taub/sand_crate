from datetime import datetime

from crate.crate import Crate
from crate.load_config import load_config
from playback import Playback


def main():
    config = load_config()
    crate = Crate(config.world_config)
    gui = Playback(crate)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_dir = config.playback_config.recording_output_dir_path / f"{timestamp}"
    gui.run_live_simulation(config.playback_config.ticks_to_record, recording_dir,
                            config.playback_config.save_recording)
    gui.show_recording(recording_dir)


if __name__ == "__main__":
    main()
