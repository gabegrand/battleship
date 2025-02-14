"""Core game logic for Battleship."""
import logging
from copy import deepcopy
from enum import StrEnum
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type

import numpy as np
from IPython.display import display

from battleship.board import Board
from battleship.board import BOARD_SYMBOL_MAPPING


logger = logging.getLogger(__name__)


class Decision(StrEnum):
    QUESTION = "question"
    MOVE = "move"


class BattleshipGame:
    def __init__(
        self,
        board_target: Board,
        captain: Type["CaptainAgent"],
        spotter: Type["SpotterAgent"],
        board_start=None,
        max_questions: int = 15,
        max_moves: int = 40,
    ):
        self.captain = captain
        self.spotter = spotter

        if board_start is None:
            self.start = Board.hidden_board(board_target.size)
        else:
            self.start = board_start

        self.state = deepcopy(self.start)
        self.target = board_target

        self.max_questions = max_questions
        self.max_moves = max_moves

        self.stage_count = 0
        self.question_count = 0
        self.move_count = 0

        self.history = []

    def __len__(self):
        return self.stage_count

    def __repr__(self):
        return f"BattleshipGame(stage={self.stage_count}, hits={self.hits}, misses={self.misses}, questions={self.question_count}/{self.max_questions}, moves={self.move_count}/{self.max_moves})"

    def __str__(self):
        return self.__repr__()

    def _ipython_display_(self):
        display(self.state)

    @property
    def hits(self):
        return np.sum(self.state.board > Board.water) - np.sum(
            self.start.board > Board.water
        )

    @property
    def misses(self):
        return np.sum(self.state.board == Board.water) - np.sum(
            self.start.board == Board.water
        )

    def play(self):
        while not self.is_done():
            self.next_stage()

    def next_stage(self):
        decision = self.captain.decision(
            state=self.state,
            questions_remaining=self.max_questions - self.question_count,
            moves_remaining=self.max_moves - self.move_count,
        )

        if decision == Decision.QUESTION:
            q = self.captain.question(self.state)
            a = self.spotter.answer(q)
            self.question_count += 1
            self.history.append(
                {
                    "stage": self.stage_count,
                    "decision": Decision.QUESTION,
                    "question": q,
                    "answer": a,
                }
            )
        elif decision == Decision.MOVE:
            coords = self.captain.move(self.state, self.history)
            self.update_state(coords)
            self.move_count += 1
            self.history.append(
                {"stage": self.stage_count, "decision": Decision.MOVE, "coords": coords}
            )
        else:
            raise ValueError(f"Invalid decision: {decision}")

        self.stage_count += 1

    def update_state(self, coords: Tuple[int, int]):
        if self.state.board[coords].item() != Board.hidden:
            raise ValueError(f"Invalid move: tile already revealed: {coords}")

        self.state.board[coords] = self.target.board[coords]

    def is_done(self):
        if self.is_won():
            return True
        elif self.move_count >= self.max_moves:
            return True
        else:
            return False

    def is_won(self):
        return self.state == self.target
