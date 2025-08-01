"""Fast conditional sampler for generating random Battleship boards.

Author: Gabe Grand (grandg@mit.edu)
"""
import logging
from collections import defaultdict
from enum import StrEnum
from typing import List

import numpy as np

from battleship.board import Board
from battleship.board import BOARD_SYMBOL_MAPPING


logger = logging.getLogger(__name__)


class Orientation(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class Span:
    def __init__(self, id: int, topleft: tuple, bottomright: tuple):
        self.id = id
        self.topleft = topleft
        self.bottomright = bottomright

        # Ensure that topleft and bottomright are valid coordinates
        assert 0 <= topleft[0] <= bottomright[0]
        assert 0 <= topleft[1] <= bottomright[1]

        # Ensure that orientation is valid
        assert self.orientation in [Orientation.HORIZONTAL, Orientation.VERTICAL]

    def __repr__(self):
        return f"Span({self.id}, {self.topleft}, {self.bottomright})"

    def __eq__(self, other):
        return self.topleft == other.topleft and self.bottomright == other.bottomright

    def __hash__(self):
        return hash((self.topleft, self.bottomright))

    @property
    def orientation(self):
        if self.topleft[0] == self.bottomright[0]:
            return Orientation.HORIZONTAL
        elif self.topleft[1] == self.bottomright[1]:
            return Orientation.VERTICAL
        else:
            raise ValueError(f"Invalid span orientation for {self}")

    @property
    def length(self):
        if self.orientation == Orientation.VERTICAL:
            return self.bottomright[0] - self.topleft[0] + 1
        else:
            return self.bottomright[1] - self.topleft[1] + 1

    @property
    def tiles(self):
        for i in range(self.topleft[0], self.bottomright[0] + 1):
            for j in range(self.topleft[1], self.bottomright[1] + 1):
                yield (i, j)


class FastSampler:
    """
    A fast sampler for placing ships on a (partially complete) board.

    Args:
        board (Board): The board on which the ships will be placed.
        ship_lengths (List[int]): The lengths of the ships to be placed; e.g., [2, 3, 4, 5].
        ship_labels (List[str]): The labels of the ships to be placed; e.g., ["R", "G", "P", "0"].
        seed (int, optional): The seed for the random number generator. Defaults to 0.
    Attributes:
        rng (numpy.random.Generator): The random number generator. Setting the state of this object can be used to reproduce the same board.
    Methods:
        populate_board(): Randomly places all ships on the board, ensuring that ships are placed in unoccupied spans.
    """

    def __init__(
        self,
        board: Board,
        ship_lengths: List[int],
        ship_labels: List[str],
        seed: int = 0,
    ):
        self.board = board
        self.ship_lengths = ship_lengths
        self.ship_labels = ship_labels

        assert len(ship_lengths) == len(ship_labels)
        assert len(set(ship_labels)) == len(ship_labels)

        self.rng = np.random.default_rng(seed)

        self._init_spans()
        self._compute_available_spans()

    def _init_spans(self):
        """Computes a data structure that stores all possible spans for ships with the specified lengths."""

        # Stores all unique spans
        self._spans = set()

        # Maps span_id to Span object
        self._spans_by_id = {}

        # Maps tile to span_id of all spans that contain that tile
        self._spans_by_tile = defaultdict(set)

        for length in self.ship_lengths:
            # Add vertical spans
            for i in range(self.board.size - length + 1):
                for j in range(self.board.size):
                    topleft = (i, j)
                    bottomright = (i + length - 1, j)
                    span = Span(len(self._spans), topleft, bottomright)
                    self._add_span(span)

            # Add horizontal spans
            for i in range(self.board.size):
                for j in range(self.board.size - length + 1):
                    topleft = (i, j)
                    bottomright = (i, j + length - 1)
                    span = Span(len(self._spans), topleft, bottomright)
                    self._add_span(span)

    def _add_span(self, span):
        self._spans.add(span)
        self._spans_by_id[span.id] = span

        for tile in span.tiles:
            self._spans_by_tile[tile].add(span.id)

    def _compute_available_spans(self):
        board = self.board.board.copy()

        # Initialize the set of spans available across the whole board
        self.available_span_ids_global = set(self._spans_by_id.keys())

        # Discard all spans that contain a water tile (0)
        for tile in zip(*np.where(board == BOARD_SYMBOL_MAPPING["W"])):
            for span_id in self._spans_by_tile[tile]:
                self.available_span_ids_global.discard(span_id)

        # Independently compute the set of available spans for each ship
        self.available_span_ids_by_ship_id = {}
        for ship_label in self.ship_labels:
            ship_id = BOARD_SYMBOL_MAPPING[ship_label]

            # Start with all spans available
            available_span_ids = self.available_span_ids_global.copy()

            # If the ship is already (partially) placed on the board, discard all spans that do not contain the ship
            if ship_id in board:
                # Find all spans that contain all of the ship's visible tiles
                ship_tiles = list(zip(*np.where(board == ship_id)))
                ship_span_ids = set.intersection(
                    *[self._spans_by_tile[tile] for tile in ship_tiles]
                )

                # Restrict the set of possible spans for the ship
                available_span_ids.intersection_update(ship_span_ids)

            # If there is nowhere to place the ship, raise an error
            if len(available_span_ids) == 0:
                raise ValueError(
                    f"Ship `{ship_label}` has no possible placement locations on the given board."
                )

            self.available_span_ids_by_ship_id[ship_id] = available_span_ids

    def populate_board(self):
        """Randomly places all ships on the board, ensuring that ships are placed in unoccupied spans."""

        # Initialize the new board
        new_board = self.board.board.copy()

        # Copy the set of available spans - this will be modified as ships are placed
        available_span_ids_local = self.available_span_ids_global.copy()

        # Place each ship on the board in order from most-to-least constrained
        for ship_label in sorted(
            self.ship_labels,
            key=lambda ship_label: len(
                self.available_span_ids_by_ship_id[BOARD_SYMBOL_MAPPING[ship_label]]
            ),
        ):
            ship_id = BOARD_SYMBOL_MAPPING[ship_label]

            available_span_ids = self.available_span_ids_by_ship_id[
                ship_id
            ].intersection(available_span_ids_local)

            # If there is nowhere to place the ship, return None
            if len(available_span_ids) == 0:
                return None

            # Randomly select a span to place the ship
            span_id = self.rng.choice(list(available_span_ids))
            span = self._spans_by_id[span_id]

            for tile in span.tiles:
                # Add the ship to the board
                new_board[tile] = ship_id

                # Discard all spans that contain this tile
                for span_id in self._spans_by_tile[tile]:
                    available_span_ids_local.discard(span_id)

        # Set all remaining hidden tiles to water
        new_board[new_board == BOARD_SYMBOL_MAPPING["H"]] = BOARD_SYMBOL_MAPPING["W"]

        return Board(new_board)

    def compute_posterior(self, n_samples: int, normalize: bool = True):
        """Computes an approximate posterior distribution over ship locations."""
        # Initialize the count of each board
        board_counts = np.zeros((self.board.size, self.board.size), dtype=int)

        for _ in range(n_samples):
            new_board = self.populate_board()
            if new_board is not None:
                board_counts += (new_board.board > 0).astype(int)

        if normalize:
            return board_counts / board_counts.sum()
        else:
            return board_counts

    def constrained_posterior(
        self,
        true_board: Board,
        n_samples: int = 10000,
        min_samples: int = 10,
        constraints: list = [],
        normalize: bool = True,
    ):
        """Computes an approximate posterior distribution over ship locations with constraints."""
        EPSILON = 0.1

        board_counts = np.zeros((self.board.size, self.board.size), dtype=float)
        total_sampled = 0
        total_consistent = 0
        active_constraints = constraints.copy()
        true_answers = [
            code_question(true_board.to_numpy(), self.board.board)
            for code_question in active_constraints
        ]

        candidate_boards = []
        for _ in range(n_samples):
            new_board = self.populate_board()
            if new_board is not None:
                candidate_boards.append(new_board)

        # Check constraints and update counts
        for new_board in candidate_boards:
            board_probability = 1

            for code_question, true_answer in zip(active_constraints, true_answers):
                # Skip if the true answer or its value is None
                if true_answer is None or true_answer.value is None:
                    continue

                # Evaluate the new answer and skip if it is None or its value is different from the true answer
                new_answer = code_question(new_board.to_numpy(), self.board.board)
                if new_answer is None or new_answer.value != true_answer.value:
                    board_probability *= EPSILON
                else:
                    board_probability *= (1 - EPSILON)

            board_counts += board_probability * (new_board.board > 0).astype(float)
            total_sampled += 1

            if total_sampled >= n_samples:
                break

        logging.warning(f"FastSampler.constrained_posterior(): {total_consistent}/{min_samples} samples collected")

        if normalize:
            return board_counts / board_counts.sum()

        logging.debug(
            f"FastSampler.constrained_posterior(): Successfully sampled {total_sampled}/{n_samples} samples (minimum {min_samples})"
        )
        return board_counts

    def heatmap(self, n_samples: int, **fig_kwargs):
        """Computes a heatmap of the approximate posterior distribution over ship locations."""
        posterior = self.compute_posterior(n_samples)
        return Board._to_figure(posterior, mode="heatmap", **fig_kwargs)
