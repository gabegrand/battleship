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
from battleship.board import SYMBOL_MEANING_MAPPING


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
            self.start = deepcopy(board_start)

        self.state = deepcopy(self.start)
        self.target = deepcopy(board_target)

        self.max_questions = max_questions
        self.max_moves = max_moves

        self.stage_count = 0
        self.question_count = 0
        self.move_count = 0

        self.history = []

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

    def sunk_ships(self):
        """
        Return a string describing the sinking status of each ship.

        Example: "Green ship sunk, Red ship sunk, Purple ship not yet sunk, Orange ship not yet sunk"
        """
        status_parts = []

        # Create reverse mapping from ship number to name
        reverse_mapping = {}
        for symbol, number in BOARD_SYMBOL_MAPPING.items():
            if number > 0:  # Skip hidden and water
                reverse_mapping[number] = SYMBOL_MEANING_MAPPING[symbol]

        # Skip water (0) and hidden (-1) tiles
        for ship_type in range(1, int(np.max(self.target.board)) + 1):
            if ship_type in reverse_mapping:
                ship_name = reverse_mapping[ship_type]

                # Count tiles of this ship type in target and state
                target_count = np.sum(self.target.board == ship_type)
                state_count = np.sum(self.state.board == ship_type)

                # Determine if ship is sunk
                if target_count > 0:
                    if target_count == state_count:
                        status_parts.append(f"{ship_name} sunk")
                    else:
                        status_parts.append(f"{ship_name} not yet sunk")

        return ", ".join(status_parts)

    def next_stage(self):
        decision = self.captain.decision(
            state=self.state,
            history=self.history,
            questions_remaining=self.max_questions - self.question_count,
            moves_remaining=self.max_moves - self.move_count,
            sunk=self.sunk_ships(),
        )

        if decision == Decision.QUESTION:
            q = self.captain.question(self.state, self.history, self.sunk_ships())
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
            coords = self.captain.move(self.state, self.history, self.sunk_ships())
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
        return np.all(
            self.state.board[self.target.board > Board.water]
            == self.target.board[self.target.board > Board.water]
        )
