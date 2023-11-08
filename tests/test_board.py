import numpy as np

from battleship.board import Board

BOARD_STRING = np.array(
    [
        ["H", "W", "H"],
        ["R", "B", "P"],
        ["H", "W", "H"],
    ]
)

BOARD_NUMERIC = np.array(
    [
        [-1, 0, -1],
        [2, 1, 3],
        [-1, 0, -1],
    ]
)


def test_convert_to_numeric():
    assert np.array_equal(Board.convert_to_numeric(BOARD_STRING), BOARD_NUMERIC)


def test_convert_to_string():
    assert np.array_equal(Board.convert_to_string(BOARD_NUMERIC), BOARD_STRING)
