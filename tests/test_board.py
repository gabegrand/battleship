import os
from tempfile import NamedTemporaryFile

import numpy as np

from battleship.board import Board

BOARD_SYMBOLIC = np.array(
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
    assert np.array_equal(Board.convert_to_numeric(BOARD_SYMBOLIC), BOARD_NUMERIC)


def test_convert_to_symbolic():
    assert np.array_equal(Board.convert_to_symbolic(BOARD_NUMERIC), BOARD_SYMBOLIC)


def test_from_text_file():
    board = Board.from_text_file(
        os.path.join(os.path.dirname(__file__), "data", "test_board.txt")
    )
    assert np.array_equal(board.board, BOARD_NUMERIC)


def test_to_text_file():
    with NamedTemporaryFile(mode="w", delete=True) as tmpfile:
        board = Board(BOARD_NUMERIC)
        board.to_text_file(tmpfile.name)
        assert np.array_equal(Board.from_text_file(tmpfile.name).board, BOARD_NUMERIC)
