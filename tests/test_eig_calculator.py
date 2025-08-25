import numpy as np

from battleship.agents import CodeQuestion
from battleship.agents import EIGCalculator
from battleship.agents import Question
from battleship.board import Board


def _code_from_body(body: str) -> CodeQuestion:
    return CodeQuestion(
        question=Question(text=""),
        fn_text=f"""
def answer(true_board, partial_board):
{body}
""",
        translation_prompt="",
        completion={},
    )


def test_eig_random_is_one():
    board_empty = Board(np.full((8, 8), -1))
    calculator = EIGCalculator(seed=42, samples=2000, epsilon=0.0)

    code_question_random = _code_from_body(
        """
    import numpy as np
    return bool(np.random.randint(0, 2))
    """
    )
    eig_random = calculator(code_question_random, board_empty)
    assert np.isclose(eig_random, 1.0, atol=0.01)


def test_eig_true_is_zero():
    board_empty = Board(np.full((8, 8), -1))
    calculator = EIGCalculator(seed=42, samples=2000, epsilon=0.0)

    code_question_true = _code_from_body(
        """
    return True
    """
    )
    eig_true = calculator(code_question_true, board_empty)
    assert eig_true == 0.0


def test_eig_none_is_nan():
    board_empty = Board(np.full((8, 8), -1))
    calculator = EIGCalculator(seed=42, samples=2000, epsilon=0.0)

    code_question_none = _code_from_body(
        """
    return None
    """
    )
    eig_none = calculator(code_question_none, board_empty)
    assert np.isnan(eig_none)
