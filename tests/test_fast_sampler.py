import numpy as np
import pytest

from battleship.agents import CodeQuestion
from battleship.agents import Question
from battleship.board import Board
from battleship.board import BOARD_SYMBOL_MAPPING
from battleship.fast_sampler import FastSampler


def _make_code_question_from_fn(fn):
    # Build a minimal CodeQuestion, then inject the predicate that returns True/False
    q = CodeQuestion(
        question=Question(text=""),
        fn_text="""
def answer(true_board, partial_board):
    return True
""",
        translation_prompt="",
        completion={},
    )
    # Attach predicate for the test runtime (must return a boolean)
    q.fn = lambda true_board, partial_board: bool(fn(true_board, partial_board))
    return q


def test_prior_is_center_heavier_than_edges():
    # Empty board prior
    empty = Board(np.full((8, 8), BOARD_SYMBOL_MAPPING["H"]))
    sampler = FastSampler(empty, seed=123)
    posterior = sampler.compute_posterior(n_samples=1000, normalize=True)

    # Check central mass > edge mass (coarse, approximate)
    center_inds = [(3, 3), (3, 4), (4, 3), (4, 4)]
    edge_inds = [(0, 0), (0, 7), (7, 0), (7, 7)]

    center_mass = sum(posterior[i, j] for (i, j) in center_inds)
    edge_mass = sum(posterior[i, j] for (i, j) in edge_inds)

    assert (
        center_mass > edge_mass * 1.5
    )  # central tiles should have substantially higher probability


def _three_red_partial_board():
    b = Board(np.full((8, 8), BOARD_SYMBOL_MAPPING["H"]))
    b.board[3, 0] = BOARD_SYMBOL_MAPPING["R"]
    b.board[3, 1] = BOARD_SYMBOL_MAPPING["R"]
    b.board[3, 2] = BOARD_SYMBOL_MAPPING["R"]
    return b


def _red_length_equals(length):
    def pred(true_board, partial_board):
        red_id = BOARD_SYMBOL_MAPPING["R"]
        return np.sum(true_board == red_id) == length

    return pred


def _red_length_gt(threshold):
    def pred(true_board, partial_board):
        red_id = BOARD_SYMBOL_MAPPING["R"]
        return np.sum(true_board == red_id) > threshold

    return pred


def test_partial_red_no_tracker_allows_len4_or_5():
    partial = _three_red_partial_board()
    sampler = FastSampler(partial, seed=123)
    weighted = sampler.get_weighted_samples(n_samples=200)
    boards = [b for (b, _) in weighted]
    assert len(boards) > 0

    red_id = BOARD_SYMBOL_MAPPING["R"]
    red_lengths = {int(np.sum(b.board == red_id)) for b in boards}
    # Without tracker, implementation allows lengths 3, 4, or 5 (given 3 visible contiguous tiles)
    assert red_lengths.issubset({3, 4, 5})
    assert red_lengths & {3, 4, 5}


def test_partial_red_tracker_sunk_len3_enforced():
    partial = _three_red_partial_board()
    # Provide remaining ships' available lengths so all ships can be placed
    tracker = [(3, "R"), (2, None), (3, None), (4, None)]
    sampler = FastSampler(partial, seed=123, ship_tracker=tracker)

    # All samples should have red length exactly 3
    cq = _make_code_question_from_fn(_red_length_equals(3))
    weighted = sampler.get_weighted_samples(n_samples=200, constraints=[(cq, True)])
    assert len(weighted) > 0
    for board, _ in weighted:
        assert np.sum(board.board == BOARD_SYMBOL_MAPPING["R"]) == 3


def test_partial_red_tracker_unsunk_requires_len_gt3():
    partial = _three_red_partial_board()
    # Provide lengths for all ships (four entries). Ensure >3 available for red.
    tracker = [(4, None), (5, None), (2, None), (2, None)]
    sampler = FastSampler(partial, seed=123, ship_tracker=tracker)

    cq = _make_code_question_from_fn(_red_length_gt(3))
    weighted = sampler.get_weighted_samples(n_samples=200, constraints=[(cq, True)])
    assert len(weighted) > 0
    for board, _ in weighted:
        assert np.sum(board.board == BOARD_SYMBOL_MAPPING["R"]) > 3


def test_partial_red_tracker_inconsistent_sunk_raises():
    partial = _three_red_partial_board()
    # Claim sunk length 2 but 3 tiles visible
    tracker = [(2, "R")]
    with pytest.raises(ValueError):
        FastSampler(partial, seed=123, ship_tracker=tracker)


def test_partial_red_tracker_all_lengths_2_raises():
    partial = _three_red_partial_board()
    tracker = [(2, None), (2, None), (2, None), (2, None)]
    with pytest.raises(ValueError):
        FastSampler(partial, seed=123, ship_tracker=tracker)
