"""Wrapper class for working with battleship boards."""
import base64
import io
import os
from enum import StrEnum
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

BOARD_SYMBOL_MAPPING = {"H": -1, "W": 0, "R": 1, "G": 2, "P": 3, "O": 4}
BOARD_COLOR_MAPPING = {
    -1: "#eaeae4",
    0: "#9b9c97",
    1: "#ac2028",
    2: "#04af70",
    3: "#6d467b",
    4: "#ffa500",
}
SYMBOL_MEANING_MAPPING = {
    "H": "hidden",
    "W": "water",
    "G": "green ship",
    "R": "red ship",
    "P": "purple ship",
    "O": "orange ship",
}
TRIAL_IDS = list(range(1, 19))


def tile_to_coords(tile: str):
    """Convert a tile string to coordinates."""
    row = ord(tile[0]) - ord("A")
    col = int(tile[1:]) - 1
    return row, col


def coords_to_tile(tile: tuple):
    """Convert a tile string to coordinates."""
    row = chr(tile[0] + ord("A"))
    col = str(tile[1] + 1)
    return row + col


class BoardFormat(StrEnum):
    GRID = "grid"
    TEXTUAL = "textual"
    VISUAL = "visual"


class Board(object):
    SHIP_LENGTHS = range(2, 6)
    SHIP_LABELS = [
        symbol for symbol, value in BOARD_SYMBOL_MAPPING.items() if value > 0
    ]

    hidden = BOARD_SYMBOL_MAPPING["H"]
    water = BOARD_SYMBOL_MAPPING["W"]
    green = BOARD_SYMBOL_MAPPING["G"]
    red = BOARD_SYMBOL_MAPPING["R"]
    purple = BOARD_SYMBOL_MAPPING["P"]
    orange = BOARD_SYMBOL_MAPPING["O"]

    def __init__(self, board: np.ndarray):
        assert board.dtype == np.dtype(int)
        assert board.shape[0] == board.shape[1]

        self._board = board

    @property
    def board(self):
        return self._board

    @property
    def size(self):
        return self._board.shape[0]

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def __repr__(self):
        return str(self)

    def __str__(self):
        output = "  " + " ".join([str(i + 1) for i in range(self.size)]) + "\n"
        for i, row in enumerate(self.to_symbolic_array()):
            output += chr(ord("A") + i) + " "
            output += " ".join([str(cell) for cell in row])
            output += "\n"
        return output

    def __deepcopy__(self, memo):
        return Board(self._board.copy())

    def _ipython_display_(self):
        display(self.to_figure())

    @staticmethod
    def symbol_to_int(symbol: str):
        return BOARD_SYMBOL_MAPPING[symbol]

    @staticmethod
    def int_to_symbol(value: int):
        return {v: k for k, v in BOARD_SYMBOL_MAPPING.items()}[value]

    @staticmethod
    def hidden_board(size: int):
        return Board(np.full((size, size), BOARD_SYMBOL_MAPPING["H"]))

    @staticmethod
    def from_symbolic_array(board: np.ndarray):
        """Instantiate a Board object from a string array."""
        return Board(Board.convert_to_numeric(board))

    def to_symbolic_array(self):
        """Convert a Board object to a string array."""
        return Board.convert_to_symbolic(self._board.copy())

    def to_serialized(self):
        """Convert a Board object into a JSON serializable string."""
        return str(self.to_symbolic_array().tolist())

    def to_numpy(self):
        """Convert a Board object into a numpy array."""
        return self._board.copy()

    @staticmethod
    def from_serialized(serialized_board):
        """Converts a JSON serializable string back into a Board object"""
        s = eval(serialized_board)
        symb = np.array(s)
        return Board.from_symbolic_array(symb)

    @staticmethod
    def from_occ_tiles(occ_tiles: str):
        A = np.array(eval(occ_tiles))
        return Board(A)

    def to_textual_description(self, include_hidden: bool = False):
        """Convert a Board object into its serialized representation"""
        repr = []
        for i, row in enumerate(self.to_symbolic_array()):
            for j in range(len(row)):
                letter = chr(ord("A") + j)
                value = row[j]
                if include_hidden or value != "H":
                    repr.append(
                        f"{i+1}-{letter} is a {SYMBOL_MEANING_MAPPING[value]} tile."
                    )
        repr = "\n".join(repr)
        return repr

    def to_format(self, fmt: BoardFormat):
        """Convert a Board object to a specified format."""
        if fmt == BoardFormat.GRID:
            return str(self)
        elif fmt == BoardFormat.TEXTUAL:
            return self.to_textual_description()
        elif fmt == BoardFormat.VISUAL:
            return self.to_base64()
        else:
            raise ValueError(f"Unknown board format: {fmt}")

    @staticmethod
    def from_text_file(path: str):
        board = pd.read_csv(path, header=None).values.astype(str)
        return Board.from_symbolic_array(board)

    def to_text_file(self, path: str):
        pd.DataFrame(self.to_symbolic_array()).to_csv(path, header=False, index=False)

    @staticmethod
    def from_trial_id(trial_id: int, experiment: str = "collaborative"):
        board_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "experiments",
            experiment,
            "contexts",
            f"board_{trial_id}.txt",
        )
        return Board.from_text_file(board_path)

    @staticmethod
    def convert_to_numeric(board: np.ndarray):
        """Convert a symbolic array to a numeric array."""
        for c, v in BOARD_SYMBOL_MAPPING.items():
            board = np.char.replace(board, c, str(v))
        return board.astype(int)

    @staticmethod
    def convert_to_symbolic(board: np.ndarray):
        """Convert a numeric array to a symbolic array."""
        board = board.astype(str)
        for c, v in BOARD_SYMBOL_MAPPING.items():
            board = np.char.replace(board, str(v), c)
        return board

    def score(self, partial_board: "Board"):
        """Computes hits, misses, precision, recall, and F1 score."""
        if not isinstance(partial_board, Board):
            raise ValueError("partial_board must be a Board object")

        assert self.size == partial_board.size

        # Compute hits and misses
        partial_board_ship_tiles = partial_board.board > Board.water
        # Everywhere that is a ship in the partial board must be the same ship in the full board
        assert np.array_equal(
            partial_board.board[partial_board_ship_tiles],
            self.board[partial_board_ship_tiles],
        ), "Ship identities do not match"

        partial_board_water_tiles = partial_board.board == Board.water
        # Everywhere that is water in the partial board must be water in the full board
        assert np.array_equal(
            partial_board.board[partial_board_water_tiles],
            self.board[partial_board_water_tiles],
        ), "Water identities do not match"

        partial_board_hidden_tiles = partial_board.board == Board.hidden
        total_ship_tiles = np.sum(self.board > Board.water)

        hits = np.sum(partial_board_ship_tiles)
        misses = np.sum(partial_board_water_tiles)
        hidden = np.sum(partial_board_hidden_tiles)

        # Compute precision and recall
        precision = hits / (hits + misses) if (hits + misses) > 0 else 0
        recall = hits / total_ship_tiles if total_ship_tiles > 0 else 0

        # Compute F1 score
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "hits": hits,
            "misses": misses,
            "hidden": hidden,
            "total_ship_tiles": total_ship_tiles,
            "is_won": hits == total_ship_tiles,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def ship_tracker(self, partial_board: "Board"):
        """
        Return a string describing the sinking status of each ship.

        Example: "Green ship sunk, Red ship sunk, Purple ship not yet sunk, Orange ship not yet sunk"
        """
        tracker = {}

        # Create reverse mapping from ship number to name
        reverse_mapping = {}
        for symbol, number in BOARD_SYMBOL_MAPPING.items():
            if number > 0:  # Skip hidden and water
                reverse_mapping[number] = SYMBOL_MEANING_MAPPING[symbol]

        # Skip water (0) and hidden (-1) tiles
        for ship_type in range(1, int(np.max(self.board)) + 1):
            if ship_type in reverse_mapping:
                ship_name = reverse_mapping[ship_type]

                # Count tiles of this ship type in target and state
                target_count = np.sum(self.board == ship_type)
                state_count = np.sum(partial_board.board == ship_type)

                # Determine if ship is sunk
                if target_count > 0:
                    if target_count == state_count:
                        tracker[ship_name] = True
                    else:
                        tracker[ship_name] = False

        return tracker

    def to_figure(self, inches: int = 6, dpi: int = 128):
        return Board._to_figure(board_array=self.board, inches=inches, dpi=dpi)

    @staticmethod
    def _to_figure(
        board_array: np.ndarray, inches: int = 6, dpi: int = 128, mode: str = "default"
    ):
        """Convert a Board object to a matplotlib figure."""

        if mode == "default":
            cmap, norm = matplotlib.colors.from_levels_and_colors(
                [-1, 0, 1, 2, 3, 4, 5], list(BOARD_COLOR_MAPPING.values())
            )
        elif mode == "heatmap":
            cmap = matplotlib.cm.get_cmap("viridis")
            norm = None
        elif mode == "heatmap_fixed_scale":
            cmap = matplotlib.cm.get_cmap("viridis")
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
        ax.matshow(board_array, cmap=cmap, norm=norm)

        # Add gridlines
        ax.set_xticks(np.arange(-0.5, len(board_array), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(board_array), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

        # Add labels
        ax.set_xticks(np.arange(0, len(board_array), 1))
        ax.set_yticks(np.arange(0, len(board_array), 1))
        ax.set_xticklabels(
            np.arange(1, len(board_array) + 1, 1),
            fontsize=24,
            fontweight="bold",
            color="#9b9c97",
        )
        ax.set_yticklabels(
            [chr(ord("A") + i) for i in np.arange(0, len(board_array), 1)],
            fontsize=24,
            fontweight="bold",
            color="#9b9c97",
        )

        # Hide ticks
        ax.tick_params(axis="both", which="both", length=0)

        # Set border to white
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        plt.close(fig)
        return fig

    def to_base64(self):
        """Convert a Board object to a PNG image encoded in UTF-8. Uses BytesIO to avoid unnecessary I/O on disk."""
        bytes = io.BytesIO()
        self.to_figure().savefig(bytes, format="png", bbox_inches="tight")
        bytes.seek(0)
        return base64.b64encode(bytes.read()).decode("utf-8")
