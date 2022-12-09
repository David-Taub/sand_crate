from crate import Crate
from game_gui import GameGUI
from load_world import load_config


def main():
    config = load_config()
    crate = Crate(config.world_config)
    gui = GameGUI(crate)
    if config.playback_config.live_simulation:
        gui.run_live_simulation()
    else:
        gui.record_simulation(config.playback_config.ticks_to_record, config.playback_config.recording_output_dir_path)
        gui.show_recording(config.playback_config.recording_output_dir_path)


if __name__ == "__main__":
    main()
