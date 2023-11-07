import numpy as np

from battleship.util import convert_board_to_char
from battleship.util import convert_board_to_numeric

BOARD_CHAR = np.array(
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


def test_convert_board_to_numeric():
    assert np.array_equal(convert_board_to_numeric(BOARD_CHAR), BOARD_NUMERIC)


def test_convert_board_to_char():
    assert np.array_equal(convert_board_to_char(BOARD_NUMERIC), BOARD_CHAR)
