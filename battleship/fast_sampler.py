from collections import defaultdict
from enum import StrEnum
from typing import List

import numpy as np

from battleship.board import Board


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
        if self.orientation == Orientation.HORIZONTAL:
            return self.bottomright[0] - self.topleft[0] + 1
        else:
            return self.bottomright[1] - self.topleft[1] + 1

    @property
    def tiles(self):
        for i in range(self.topleft[0], self.bottomright[0] + 1):
            for j in range(self.topleft[1], self.bottomright[1] + 1):
                yield (i, j)


class FastSampler:
    def __init__(self, board: Board, ship_lengths: List[int], seed: int = 0):
        self.board = board
        self.ship_lengths = ship_lengths

        self.rng = np.random.default_rng(seed)

        self._compute_spans()

    def _compute_spans(self):
        """Computes a data structure that stores all possible spans for ships with the specified lengths."""

        # Stores all unique spans
        self.spans = set()

        # Maps span_id to Span object
        self.spans_by_id = {}

        # Maps ship_length to Span objects
        self.spans_by_length = defaultdict(set)

        # Maps tile to span_id of all spans that contain that tile
        self.spans_by_tile = defaultdict(set)

        for length in self.ship_lengths:
            # Add vertical spans
            for i in range(self.board.size - length + 1):
                for j in range(self.board.size):
                    topleft = (i, j)
                    bottomright = (i + length - 1, j)
                    span = Span(len(self.spans), topleft, bottomright)
                    self.spans.add(span)
                    self.spans_by_id[span.id] = span
                    self.spans_by_length[length].add(span.id)

                    for tile in span.tiles:
                        self.spans_by_tile[tile].add(span.id)

            # Add horizontal spans
            for i in range(self.board.size):
                for j in range(self.board.size - length + 1):
                    topleft = (i, j)
                    bottomright = (i, j + length - 1)
                    span = Span(len(self.spans), topleft, bottomright)
                    self.spans.add(span)
                    self.spans_by_id[span.id] = span
                    self.spans_by_length[length].add(span.id)

                    for tile in span.tiles:
                        self.spans_by_tile[tile].add(span.id)

    def complete_board(self):
        """Randomly places all ships on the board, ensuring that ships are placed in unoccupied spans."""
        new_board = self.board.board.copy()

        available_span_ids_global = set(self.spans_by_id.keys())

        # Discard all spans that contain a water tile (0)
        for tile in zip(*np.where(new_board == 0)):
            for span_id in self.spans_by_tile[tile]:
                available_span_ids_global.discard(span_id)

        for ship_id, ship_length in enumerate(self.ship_lengths):
            # ship_id is 1-indexed
            ship_id += 1

            print(ship_id)

            # Start with all spans matching the ship's length
            available_span_ids = available_span_ids_global.intersection(
                self.spans_by_length[ship_length]
            )

            # If the ship is already (partially) placed on the board, discard all spans that do not contain the ship
            if ship_id in new_board:
                # Find all spans that contain all of the ship's visible tiles
                ship_tiles = list(zip(*np.where(new_board == ship_id)))
                ship_span_ids = set.intersection(
                    *[self.spans_by_tile[tile] for tile in ship_tiles]
                )

                # Restrict the set of possible spans for the ship
                available_span_ids.intersection_update(ship_span_ids)

            # If there is nowhere to place the ship, return None
            if len(available_span_ids) == 0:
                return None

            # Randomly select a span to place the ship
            span_id = self.rng.choice(list(available_span_ids))
            span = self.spans_by_id[span_id]

            for tile in span.tiles:
                # Add the ship to the board
                new_board[tile] = ship_id

                # Discard all spans that contain this tile
                for span_id in self.spans_by_tile[tile]:
                    available_span_ids_global.discard(span_id)

        return Board(new_board)
