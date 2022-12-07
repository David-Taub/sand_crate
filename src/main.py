from crate import Crate
from game_gui import GameGUI


def main():
    crate = Crate(300)
    # GameGUI(crate).run_live_simulation()
    game = GameGUI(crate)
    game.record_simulation(300)
    game.show_recording()


if __name__ == '__main__':
    main()
