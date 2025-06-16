import json
import os
import time
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np

from battleship.agents import get_openai_client
from battleship.game import Decision

# Forward declarations to avoid circular imports
if TYPE_CHECKING:
    from battleship.agents import ActionData, Question, Answer


class BaseStrategy(ABC):
    def __init__(self, index_counter=None):
        self.index_counter = index_counter


class DecisionStrategy(BaseStrategy):
    @abstractmethod
    def make_decision(
        self, state, history, questions_remaining, moves_remaining, sunk
    ) -> Tuple[Decision, "ActionData"]:
        pass


class MoveStrategy(BaseStrategy):
    @abstractmethod
    def make_move(
        self, state, history, sunk, questions_remaining, moves_remaining, constraints
    ) -> Tuple[Tuple[int, int], "ActionData"]:
        pass


class QuestionStrategy(BaseStrategy):
    @abstractmethod
    def ask_question(
        self, state, history, sunk, questions_remaining, moves_remaining
    ) -> Tuple["Question", "ActionData"]:
        pass


class AnswerStrategy(BaseStrategy):
    @abstractmethod
    def answer_question(
        self, question, occ_tiles, history=None
    ) -> Tuple["Answer", "ActionData"]:
        pass
