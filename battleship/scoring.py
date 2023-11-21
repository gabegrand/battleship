import os

from eig import compute_eig_fast
from eig.battleship.program import ProgramSyntaxError

from battleship.board import Board


def compute_score(board: Board, program: str):
    assert isinstance(board, Board)

    try:
        score = compute_eig_fast(
            program,
            board.board,
            grid_size=6,
            ship_labels=[1, 2, 3],
            ship_sizes=[2, 3, 4],
            orientations=["V", "H"],
        )
    except ProgramSyntaxError:
        score = 0
    except RuntimeError:
        score = 0

    return score
