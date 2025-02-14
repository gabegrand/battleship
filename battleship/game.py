"""Core game logic for Battleship."""
from enum import StrEnum
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type

from battleship.board import Board
from battleship.board import BOARD_SYMBOL_MAPPING


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

        self.state = self.start
        self.target = board_target

        self.max_questions = max_questions
        self.max_moves = max_moves

        self.stage_count = 0
        self.question_count = 0
        self.move_count = 0

        self.history = []

    def simulate(self):
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
            coords = self.captain.move(self.state)
            self.update_state(coords)
            self.move_count += 1
            self.history.append(
                {"stage": self.stage_count, "decision": Decision.MOVE, "coords": coords}
            )
        else:
            raise ValueError(f"Invalid decision: {decision}")

        self.stage_count += 1

    def update_state(self, coords: Tuple[int, int]):
        if self.state.board[coords] != Board.hidden:
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


class Decision(StrEnum):
    QUESTION = "question"
    MOVE = "move"
