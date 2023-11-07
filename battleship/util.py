import numpy as np

BOARD_SYMBOL_MAPPING = {"H": -1, "W": 0, "B": 1, "R": 2, "P": 3}


def convert_board_to_numeric(board: np.ndarray):
    for c, v in BOARD_SYMBOL_MAPPING.items():
        board = np.char.replace(board, c, str(v))
    return board.astype(int)


def convert_board_to_char(board: np.ndarray):
    board = board.astype(str)
    for c, v in BOARD_SYMBOL_MAPPING.items():
        board = np.char.replace(board, str(v), c)
    return board
