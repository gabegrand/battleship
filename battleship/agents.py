from typing import Dict
from typing import List
from typing import Tuple

from battleship.board import Board


class Agent:
    def __init__(self):
        pass

    def decision(
        self,
        state: Board,
        history: List[Dict],
        questions_remaining: int,
        moves_remaining: int,
    ):
        raise NotImplementedError


class CaptainAgent(Agent):
    def question(self, state: Board, history: List[Dict]):
        raise NotImplementedError

    def move(self, state: Board, history: List[Dict]):
        raise NotImplementedError


class SpotterAgent(Agent):
    def answer(
        self,
        state: Board,
        history: List[Dict],
        question: Tuple[int, int],
    ):
        raise NotImplementedError
