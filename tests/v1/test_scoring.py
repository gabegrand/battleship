from battleship.board import Board
from battleship.scoring import compute_score


def test_compute_score():
    board = Board.from_trial_id(1)
    assert compute_score(program="foo", board=board) == 0
    assert compute_score(program="(size Red)", board=board) > 0
