"""Core game logic for Battleship."""
import json
import logging
import os
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
from battleship.board import SYMBOL_MEANING_MAPPING


logger = logging.getLogger(__name__)


class Decision(StrEnum):
    QUESTION = "question"
    MOVE = "move"


class BattleshipGame:
    def __init__(
        self,
        board_target: Board,
        captain: Type["Captain"],
        spotter: Type["Spotter"],
        board_start=None,
        max_questions: int = 15,
        max_moves: int = 40,
        save_dir: str = None,
    ):
        self.captain = captain
        self.spotter = spotter

        if board_start is None:
            self.start = Board.hidden_board(board_target.size)
        else:
            self.start = deepcopy(board_start)

        self.state = deepcopy(self.start)
        self.target = deepcopy(board_target)

        self.max_questions = max_questions
        self.max_moves = max_moves

        self.stage_count = 0
        self.question_count = 0
        self.move_count = 0

        self.history = []

        # Setup save directory for game history if provided
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            self.save_path = os.path.join(self.save_dir, "game.json")
        else:
            self.save_path = None

    def __len__(self):
        return self.stage_count

    def __repr__(self):
        return (
            f"BattleshipGame(\n"
            f"  stage={self.stage_count},\n"
            f"  hits={self.hits},\n"
            f"  misses={self.misses},\n"
            f"  questions={self.question_count}/{self.max_questions},\n"
            f"  moves={self.move_count}/{self.max_moves},\n"
            f"  is_done={self.is_done()},\n"
            f"  is_won={self.is_won()}\n"
            f")"
        )

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

    def __iter__(self):
        def generator():
            while not self.is_done():
                yield self.next_stage()

        return generator()

    def play(self):
        while not self.is_done():
            self.next_stage()

    def next_stage(self):
        # TODO: Move string formatting logic into the Agent classes
        sunk_string = ", ".join(
            [
                f"{ship}: {'sunk' if status else 'not sunk'}"
                for ship, status in self.target.ship_tracker(self.state).items()
            ]
        )

        decision = self.captain.decision(
            state=self.state,
            history=self.history,
            questions_remaining=self.max_questions - self.question_count,
            moves_remaining=self.max_moves - self.move_count,
            sunk=sunk_string,
        )

        if decision == Decision.QUESTION:
            q = self.captain.question(self.state, self.history, sunk_string)
            a = self.spotter.answer(
                question=q, occ_tiles=self.state.board, history=self.history
            )
            if a.code_question is not None:
                self.captain.sampling_constraints.append(a.code_question)

            self.question_count += 1
            self.history.append(
                {
                    "stage": self.stage_count,
                    "decision": Decision.QUESTION,
                    "question": q,
                    "answer": a,
                    "state": self.state.board.tolist(),
                }
            )
        elif decision == Decision.MOVE:
            coords = self.captain.move(
                self.state,
                self.history,
                sunk_string,
                self.captain.sampling_constraints,
            )

            if coords is not None:
                self.update_state(coords)

            self.move_count += 1
            self.history.append(
                {
                    "stage": self.stage_count,
                    "decision": Decision.MOVE,
                    "coords": tuple(int(x) for x in coords),
                    "state": self.state.board.tolist(),
                }
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
        return np.all(
            self.state.board[self.target.board > Board.water]
            == self.target.board[self.target.board > Board.water]
        )

    def score(self):
        return self.target.score(self.state)

    def save(self):
        if self.save_path is None:
            return

        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=4)
