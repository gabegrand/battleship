from abc import ABC
from abc import abstractmethod
from typing import Tuple
from typing import TYPE_CHECKING

from battleship.game import Decision

# Forward declarations to avoid circular imports
if TYPE_CHECKING:
    from battleship.agents import ActionData, Question, Answer
    from battleship.board import Board


class BaseStrategy(ABC):
    def __init__(self):
        pass


class DecisionStrategy(BaseStrategy):
    @abstractmethod
    def __call__(
        self, state, history, questions_remaining, moves_remaining, ship_tracker
    ) -> Tuple[Decision, "ActionData"]:
        pass


class MoveStrategy(BaseStrategy):
    @abstractmethod
    def __call__(
        self,
        state,
        history,
        ship_tracker,
        questions_remaining,
        moves_remaining,
        constraints,
    ) -> Tuple[Tuple[int, int], "ActionData"]:
        pass


class QuestionStrategy(BaseStrategy):
    @abstractmethod
    def __call__(
        self, state, history, ship_tracker, questions_remaining, moves_remaining
    ) -> Tuple["Question", "ActionData"]:
        pass


class AnswerStrategy(BaseStrategy):
    @abstractmethod
    def __call__(
        self, question, board: "Board", history=None
    ) -> Tuple["Answer", "ActionData"]:
        pass
