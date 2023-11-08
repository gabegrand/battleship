"""Wrapper class for working with battleship boards."""
import numpy as np

BOARD_SYMBOL_MAPPING = {"H": -1, "W": 0, "B": 1, "R": 2, "P": 3}


class Board(object):
    def __init__(self, board: np.ndarray):
        assert board.dtype == np.dtype(int)
        self.board = board

    def __repr__(self):
        return self.board.__repr__()

    @staticmethod
    def from_string_array(board: np.ndarray):
        """Instantiate a Board object from a string array."""
        return Board(Board.convert_to_numeric(board))

    def to_string_array(self):
        """Convert a Board object to a string array."""
        return Board.convert_to_string(self.board.copy())

    @staticmethod
    def convert_to_numeric(board: np.ndarray):
        """Convert a string array to a numeric array."""
        for c, v in BOARD_SYMBOL_MAPPING.items():
            board = np.char.replace(board, c, str(v))
        return board.astype(int)

    @staticmethod
    def convert_to_string(board: np.ndarray):
        """Convert a numeric array to a string array."""
        board = board.astype(str)
        for c, v in BOARD_SYMBOL_MAPPING.items():
            board = np.char.replace(board, str(v), c)
        return board
