"""Wrapper class for working with battleship boards."""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

BOARD_SYMBOL_MAPPING = {"H": -1, "W": 0, "B": 1, "R": 2, "P": 3}


class Board(object):
    def __init__(self, board: np.ndarray):
        assert board.dtype == np.dtype(int)
        assert board.shape[0] == board.shape[1]

        self.board = board
        self.size = board.shape[0]

    def __repr__(self):
        return str(self)

    def __str__(self):
        output = "  " + " ".join([str(x) for x in range(1, self.size + 1)]) + "\n"
        for i, row in enumerate(self.to_symbolic_array()):
            output += chr(ord("A") + i) + " "
            output += " ".join([str(cell) for cell in row])
            output += "\n"
        return output

    @staticmethod
    def from_symbolic_array(board: np.ndarray):
        """Instantiate a Board object from a string array."""
        return Board(Board.convert_to_numeric(board))

    def to_symbolic_array(self):
        """Convert a Board object to a string array."""
        return Board.convert_to_symbolic(self.board.copy())

    @staticmethod
    def from_text_file(path: str):
        board = pd.read_csv(path, header=None).values.astype(str)
        return Board.from_symbolic_array(board)

    def to_text_file(self, path: str):
        pd.DataFrame(self.to_symbolic_array()).to_csv(path, header=False, index=False)

    def to_figure(self):
        """Convert a Board object to a matplotlib figure."""
        cmap = ListedColormap(["#eaeae4", "#9b9c97", "#2d7bac", "#ac2028", "#6d467b"])
        fig, ax = plt.subplots()
        ax.matshow(self.board, cmap=cmap)

        # Add gridlines
        ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

        # Add labels
        ax.set_xticks(np.arange(0, self.size, 1))
        ax.set_yticks(np.arange(0, self.size, 1))
        ax.set_xticklabels(
            np.arange(1, self.size + 1, 1),
            fontsize=24,
            fontweight="bold",
            color="#9b9c97",
        )
        ax.set_yticklabels(
            [chr(ord("A") + i) for i in np.arange(0, self.size, 1)],
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
