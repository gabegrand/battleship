from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from battleship.board import Board
from battleship.game import Decision


class Agent:
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)


class Captain(Agent):
    def decision(
        self,
        state: Board,
        history: List[Dict],
        questions_remaining: int,
        moves_remaining: int,
    ):
        raise NotImplementedError

    def question(self, state: Board, history: List[Dict]):
        raise NotImplementedError

    def move(self, state: Board, history: List[Dict]):
        raise NotImplementedError


class Spotter(Agent):
    def answer(
        self,
        state: Board,
        history: List[Dict],
        question: Tuple[int, int],
    ):
        raise NotImplementedError


class RandomCaptain(Captain):
    def decision(self, *args, **kwargs):
        return Decision.MOVE

    def move(self, state: Board, history: List[Dict]) -> Tuple[int, int]:
        hidden_tiles = np.argwhere(state.board == Board.hidden)
        if len(hidden_tiles) == 0:
            raise ValueError("No hidden tiles left.")
        coords = self.rng.choice(hidden_tiles)
        return tuple(coords)
