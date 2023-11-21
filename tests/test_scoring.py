from battleship.board import Board
from battleship.scoring import compute_score


def test_compute_score():
    board = Board.from_trial_id(1)
    assert compute_score(board, "foo") == 0
    assert compute_score(board, "(size Red)") > 0
