"""Fast conditional sampler for generating random Battleship boards.

Author: Gabe Grand (grandg@mit.edu)
"""
import logging
from collections import defaultdict
from enum import StrEnum
from typing import List
from typing import Optional
from typing import Tuple

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
        ship_tracker (List[Tuple[int, Optional[str]]]): The tracker of the ships to be placed; e.g., [(4, None), (3, "R"), (2, "G"), (3, None)]. If no tracker is provided, all ships are assumed to be unsunk and all lengths are available.
        ship_lengths (List[int]): The possible lengths of the ships to be placed; e.g., [2, 3, 4, 5]. Note that there *can* be multiple ships of the same length.
        ship_labels (List[str]): The labels of the ships to be placed; e.g., ["R", "G", "P", "O"]. Note that each label is unique; there *cannot* be multiple ships with the same label.
        seed (int, optional): The seed for the random number generator. Defaults to 0.
    Attributes:
        rng (numpy.random.Generator): The random number generator. Setting the state of this object can be used to reproduce the same board.
    Methods:
        populate_board(): Randomly places all ships on the board, ensuring that ships are placed in unoccupied spans.
    """

    def __init__(
        self,
        board: Board,
        ship_tracker: List[Tuple[int, Optional[str]]] = None,
        ship_lengths: List[int] = Board.SHIP_LENGTHS,
        ship_labels: List[str] = Board.SHIP_LABELS,
        seed: int = 0,
    ):
        self.board = board

        self.ship_lengths = ship_lengths
        self.ship_labels = ship_labels

        if ship_tracker is None:
            ship_tracker = []

        self.ship_tracker = ship_tracker
        self.sunk_ships = {
            label: length for length, label in ship_tracker if label is not None
        }
        self.available_ship_lengths_unsunk = [
            length for length, label in ship_tracker if label is None
        ]

        assert len(ship_lengths) == len(ship_labels)
        assert len(set(ship_labels)) == len(ship_labels)
        assert set(self.available_ship_lengths_unsunk).issubset(set(ship_lengths))
        assert set(self.sunk_ships.values()).issubset(set(ship_lengths))
        assert set(self.sunk_ships.keys()).issubset(set(ship_labels))

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

            # If the ship is sunk, there is only one possible span
            if ship_label in self.sunk_ships:
                length = self.sunk_ships[ship_label]
                # Find the span that contains the ship
                ship_tiles = list(zip(*np.where(board == ship_id)))
                if len(ship_tiles) != length:
                    raise ValueError(
                        f"Ship `{ship_label}` is marked sunk in tracker, but has {len(ship_tiles)} tiles on the board; expected {length}."
                    )

                # Spans that contain all tiles of this ship
                candidate_span_ids = set.intersection(
                    *[self._spans_by_tile[tile] for tile in ship_tiles]
                )

                # Keep only spans of the correct length and currently available
                candidate_span_ids = {
                    s_id
                    for s_id in candidate_span_ids.intersection(available_span_ids)
                    if self._spans_by_id[s_id].length == length
                }

                if len(candidate_span_ids) != 1:
                    raise ValueError(
                        f"Ship `{ship_label}` has {len(candidate_span_ids)} possible spans; expected exactly 1."
                    )

                # Restrict available spans to this unique span
                available_span_ids = candidate_span_ids

            # The ship is not sunk
            else:
                # If the ship is already partially placed on the board, discard all spans that do not contain the ship
                if ship_id in board:
                    # Find all spans that contain all of the ship's visible tiles
                    ship_tiles = list(zip(*np.where(board == ship_id)))
                    ship_span_ids = set.intersection(
                        *[self._spans_by_tile[tile] for tile in ship_tiles]
                    )

                    # Restrict available spans to the spans that contain the ship's visible tiles
                    available_span_ids.intersection_update(ship_span_ids)

                    if len(available_span_ids) == 0:
                        raise ValueError(
                            f"Ship `{ship_label}` has no possible placement locations on the given board: It is partially placed, but there are no  spans that contain all of its visible tiles."
                        )

                # Restrict the set of possible spans to the lengths that are still available in the ship tracker
                if self.ship_tracker:
                    available_span_ids = {
                        s_id
                        for s_id in available_span_ids
                        if self._spans_by_id[s_id].length
                        in self.available_ship_lengths_unsunk
                    }

                    if len(available_span_ids) == 0:
                        raise ValueError(
                            f"Ship `{ship_label}` has no possible placement locations on the given board: There are no spans consistent with the available ship lengths specified in the ship tracker."
                        )

            # If there is nowhere to place the ship, raise an error
            if len(available_span_ids) == 0:
                raise ValueError(
                    f"Ship `{ship_label}` has no possible placement locations on the given board: There are no available spans, but the reason is unknown. This suggests there is a bug in the ship tracker."
                )

            self.available_span_ids_by_ship_id[ship_id] = available_span_ids

    def populate_board(self):
        """Randomly places all ships on the board, ensuring that ships are placed in unoccupied spans."""

        # Initialize the new board
        new_board = self.board.board.copy()

        # Copy the set of available spans - this will be modified as ships are placed
        remaining_span_ids = self.available_span_ids_global.copy()

        # Copy the set of all available ship lengths (including sunk ships) - this will be modified as ships are placed
        remaining_ship_lengths = [length for length, _ in self.ship_tracker]

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
            ].intersection(remaining_span_ids)

            logger.debug(
                f"Placing ship `{ship_label}`:\n- {remaining_ship_lengths} remaining ship lengths\n- {len(self.available_span_ids_by_ship_id[BOARD_SYMBOL_MAPPING[ship_label]])} globally-available spans\n- {len(remaining_span_ids)} remaining spans"
            )

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
                    remaining_span_ids.discard(span_id)

            # Remove the length of the ship from the remaining ship lengths
            if self.ship_tracker:
                remaining_ship_lengths.remove(span.length)

                # Discard all spans that are not consistent with the remaining ship lengths
                remaining_span_ids = {
                    s_id
                    for s_id in remaining_span_ids
                    if self._spans_by_id[s_id].length in remaining_ship_lengths
                }

        # Set all remaining hidden tiles to water
        new_board[new_board == BOARD_SYMBOL_MAPPING["H"]] = BOARD_SYMBOL_MAPPING["W"]

        return Board(new_board)

    def get_weighted_samples(
        self,
        n_samples: int,
        constraints: List[Tuple["CodeQuestion", bool]] = [],
        epsilon: float = 0.1,
    ):
        """
        Generate weighted board samples based on constraints.

        Args:
            n_samples: Number of samples to generate
            constraints: List of (CodeQuestion, bool) tuples representing Q/A pairs
            epsilon: Weight for constraint violations

        Returns:
            List of (Board, weight) tuples
        """
        # Generate candidate boards
        candidate_boards = []
        for _ in range(n_samples):
            new_board = self.populate_board()
            if new_board is not None:
                candidate_boards.append(new_board)

        if len(candidate_boards) == 0:
            logger.warning(
                "FastSampler.get_weighted_samples(): Unable to generate any candidate boards - returning empty list"
            )
            return []

        # Calculate weights for each board based on constraint satisfaction
        # Handle None answers by applying the average multiplier for that constraint
        weights = [1.0 for _ in candidate_boards]

        for code_question, expected_answer in constraints:
            per_board_multiplier = []
            has_any_answer = False

            # First pass: compute multipliers where answers exist
            for board in candidate_boards:
                constraint_answer = code_question(board.to_numpy(), self.board.board)

                if constraint_answer is None or constraint_answer.value is None:
                    per_board_multiplier.append(None)
                    continue

                has_any_answer = True
                if constraint_answer.value == expected_answer:
                    per_board_multiplier.append(1 - epsilon)
                else:
                    per_board_multiplier.append(epsilon)

            # Determine default multiplier for boards with None
            if has_any_answer:
                observed = [m for m in per_board_multiplier if m is not None]
                default_multiplier = sum(observed) / len(observed)
            else:
                # If no board yielded an answer for this constraint, it provides no information
                default_multiplier = 1.0

            # Second pass: apply multipliers (use average for None)
            for i, m in enumerate(per_board_multiplier):
                weights[i] *= m if m is not None else default_multiplier

        weighted_boards = list(zip(candidate_boards, weights))

        # Normalize weights so they sum to 1 (if possible)
        total_weight = sum(weight for _, weight in weighted_boards)
        if total_weight > 0:
            return [(board, weight / total_weight) for board, weight in weighted_boards]
        else:
            logger.warning(
                "FastSampler.get_weighted_samples(): All weights are zero - returning uniform weights"
            )
            # Fallback: if all weights are zero but we have candidates, use uniform distribution
            # This can occur when epsilon is 0 and all of the candidate boards are invalid
            uniform_weight = 1.0 / len(candidate_boards)
            return [(board, uniform_weight) for board in candidate_boards]

    def compute_posterior(
        self,
        n_samples: int,
        normalize: bool = True,
        constraints: list = [],
        epsilon: float = 0.1,
        min_samples: int = 10,
    ):
        """Computes an approximate posterior distribution over ship locations.

        Args:
            n_samples: Number of samples to generate
            normalize: Normalizes each tile to [0, 1]. If False, returns unnormalized counts.
            constraints: List of (CodeQuestion, bool) tuples representing Q/A pairs (optional)
            epsilon: Weight for constraint violations (used only with constraints)
            min_samples: Minimum samples for logging (used only with constraints)
        """
        weighted_boards = self.get_weighted_samples(
            n_samples=n_samples, constraints=constraints, epsilon=epsilon
        )

        # Accumulate weighted board counts
        board_counts = np.zeros((self.board.size, self.board.size), dtype=float)
        total_sampled = len(weighted_boards)

        for board, weight in weighted_boards:
            board_counts += weight * (board.board > 0).astype(float)

        # Independently normalize each tile to [0, 1]
        if normalize:
            weights = np.array([weight for _, weight in weighted_boards])
            board_counts /= weights.sum()

        if total_sampled < min_samples:
            logger.warning(
                f"FastSampler.compute_posterior(): {total_sampled}/{min_samples} samples collected"
            )
        else:
            logger.debug(
                f"FastSampler.compute_posterior(): Successfully sampled {total_sampled}/{n_samples} samples (minimum {min_samples})"
            )

        return board_counts

    def heatmap(
        self,
        **kwargs,
    ):
        posterior = self.compute_posterior(normalize=False, **kwargs)
        return Board._to_figure(posterior, mode="heatmap")
