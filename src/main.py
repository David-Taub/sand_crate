from crate import Crate
from game_gui import GameGUI


def main():
    crate = Crate()
    GameGUI(crate).run_main_loop()


if __name__ == '__main__':
    main()
