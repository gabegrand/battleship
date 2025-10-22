"""Wrapper class for working with battleship boards."""
import base64
import io
import os
from enum import StrEnum
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle

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

    def __init__(self, board: np.ndarray, transparent: bool = None):
        assert board.dtype == np.dtype(int)
        assert board.shape[0] == board.shape[1]

        self._board = board
        self.transparent = transparent

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
        display(self.to_figure(transparent=self.transparent or False))

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
        if isinstance(occ_tiles, str):
            return Board(np.array(eval(occ_tiles)))
        elif isinstance(occ_tiles, (list, np.ndarray)):
            return Board(np.array(occ_tiles))
        elif isinstance(occ_tiles, Board):
            return occ_tiles
        else:
            raise ValueError(
                f"occ_tiles must be a string, list, numpy array, or Board object. Got {type(occ_tiles)}"
            )

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
        if isinstance(trial_id, int):
            trial_id = f"B{trial_id:02d}"
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

    def ship_tracker(self, partial_board: "Board") -> List[Tuple[int, Optional[str]]]:
        """
        Return an array describing the sinking status of each ship.

        Example: [(4, None), (3, "R"), (2, "G")] implies there is a ship of length 4 that has not been sunk, a ship of length 3 that is "R" and has been sunk, and a ship of length 2 that is "G" and has been sunk.
        """
        tracker = []

        # Create reverse mapping from ship number to name
        reverse_mapping = {}
        for symbol, number in BOARD_SYMBOL_MAPPING.items():
            if number > 0:  # Skip hidden and water
                reverse_mapping[number] = symbol

        # Skip water (0) and hidden (-1) tiles
        for ship_type in range(1, int(np.max(self.board)) + 1):
            if ship_type in reverse_mapping:
                ship_symbol = reverse_mapping[ship_type]

                # Count tiles of this ship type in target and state
                target_count = np.sum(self.board == ship_type)
                state_count = np.sum(partial_board.board == ship_type)

                # Determine if ship is sunk
                if target_count > 0:
                    if target_count == state_count:
                        tracker.append((target_count, ship_symbol))
                    else:
                        tracker.append((target_count, None))

        return tracker

    def to_figure(self, inches: int = 6, dpi: int = 128, transparent: bool = False):
        return Board._to_figure(
            board_array=self.board,
            inches=inches,
            dpi=dpi,
            transparent=transparent or self.transparent,
        )

    def spotter_view(
        self,
        partial_board: "Board",
        inches: int = 6,
        dpi: int = 128,
        transparent: bool = False,
        hatch: str = "///",
        hatch_color: str = "#ffffff",
        overlay_color: Optional[str] = None,
        tile_alpha: float = 0.5,
    ):
        """Render the spotter view.

            This renders the true board (``self``) while overlaying a diagonal
            cross-hatching pattern on tiles that are hidden to the Captain in
            ``partial_board``. Tiles that are visible to the Captain are rendered
            normally.

            Parameters
            - partial_board: Board - the Captain's current (partial) view. Must be
              the same size as ``self``.
            - inches, dpi, transparent: forwarded to the underlying figure
              creation similar to ``to_figure``.
        - hatch: matplotlib hatch pattern used for hidden tiles.
        - hatch_color: color used for the hatch lines.
        - overlay_color: fill color for hidden tiles (defaults to hatch_color).
        - tile_alpha: transparency (0-1) applied to the hidden tile fill only.

            Returns
            - matplotlib.figure.Figure: the rendered figure.
        """
        if not isinstance(partial_board, Board):
            raise ValueError("partial_board must be a Board object")
        if partial_board.size != self.size:
            raise ValueError("partial_board must be the same size as the board")

        # First, render the true board using the existing helper to avoid code duplication.
        fig = Board._to_figure(
            board_array=self.board,
            inches=inches,
            dpi=dpi,
            transparent=transparent or self.transparent,
        )

        # Overlay hatch on cells hidden in the Captain's view.
        ax = fig.axes[0] if fig.axes else fig.gca()
        hidden_mask = partial_board.board == Board.hidden

        n = self.size
        # Choose overlay fill color (with alpha applied to face only)
        if overlay_color is None:
            overlay_color = hatch_color
        face_rgba = matplotlib.colors.to_rgba(overlay_color, alpha=tile_alpha)
        for i in range(n):
            for j in range(n):
                if hidden_mask[i, j]:
                    # Place a 1x1 rectangle centered on the (i,j) cell in data coords.
                    rect = Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=face_rgba,
                        edgecolor=hatch_color,
                        hatch=hatch,
                        linewidth=0.0,
                        zorder=3,
                    )
                    ax.add_patch(rect)

        return fig

    def ship_tracker_figure(
        self,
        partial_board: "Board",
        inches: float = 2.5,
        dpi: int = 128,
        transparent: bool = False,
        legend: bool = True,
    ):
        """Render a compact ship tracker panel as a matplotlib Figure.

        - Shows a title "Ship Tracker"
        - Optional legend of the ship colors in a static order (R, G, P, O)
        - Renders one horizontal row of unit squares per ship; gray until sunk,
          then filled with the ship's color.

        Returns a matplotlib.figure.Figure.
        """
        if not isinstance(partial_board, Board):
            raise ValueError("partial_board must be a Board object")
        if partial_board.size != self.size:
            raise ValueError("partial_board must be the same size as the board")

        tracker = self.ship_tracker(partial_board)  # [(length, symbol_or_None)]
        if not tracker:
            # Fallback: infer by color ids on board
            lengths = [int(np.sum(self.board == k)) for k in range(1, 5)]
            tracker = [(l, None) for l in lengths if l > 0]

        # Sort rows by ship length descending for a consistent look
        tracker = sorted(tracker, key=lambda x: x[0], reverse=True)
        num_rows = len(tracker)
        max_len = max(l for l, _ in tracker) if tracker else 5

        # Layout in axis units
        pad = 0.5
        legend_h = 1.5 if legend else 1.0
        row_h = 1.2
        # Precompute legend width so the last color box doesn't get clipped
        legend_label_offset = 3.2  # space to the right of the text label before boxes
        box_size = 0.5
        box_gap = 0.2
        boxes_total_width = 4 * box_size + 3 * box_gap
        legend_required_width = pad + legend_label_offset + boxes_total_width + pad
        width_units = max(max_len + 2 * pad, legend_required_width)
        height_units = legend_h + num_rows * row_h + pad

        # Keep squares square by adjusting the figure aspect
        aspect = height_units / width_units
        fig_w = inches
        fig_h = inches * aspect
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        if transparent:
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")

        # Axis setup
        ax.set_xlim(0, width_units)
        ax.set_ylim(height_units, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        # Title (should appear above the ship colors)
        title_y = legend_h * 0.25
        ax.text(
            pad,
            title_y,
            "Ship Tracker",
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="center",
            color="#1f2937",
        )

        # Legend of ship colors in a static order: R, G, P, O
        if legend:
            # Place legend below the title
            legend_label_y = legend_h * 0.65
            ax.text(
                pad,
                legend_label_y,
                "Ship Colors:",
                fontsize=9,
                ha="left",
                va="center",
                color="#6b7280",
            )
            color_order = ["R", "G", "P", "O"]
            x = pad + legend_label_offset  # some space after the label
            for sym in color_order:
                col = BOARD_COLOR_MAPPING[BOARD_SYMBOL_MAPPING[sym]]
                ax.add_patch(
                    Rectangle(
                        (x, legend_label_y - box_size / 2),
                        box_size,
                        box_size,
                        facecolor=col,
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
                x += box_size + box_gap

        # Rows: one per ship
        start_y = legend_h + pad
        water_gray = BOARD_COLOR_MAPPING[0]
        for r, (length, sunk_symbol) in enumerate(tracker):
            y = start_y + r * row_h
            x0 = pad
            fill_color = (
                water_gray
                if sunk_symbol is None
                else BOARD_COLOR_MAPPING[BOARD_SYMBOL_MAPPING[sunk_symbol]]
            )
            for k in range(length):
                ax.add_patch(
                    Rectangle(
                        (x0 + k, y),
                        1.0,
                        1.0,
                        facecolor=fill_color,
                        edgecolor="white",
                        linewidth=1.0,
                    )
                )

        plt.close(fig)
        return fig

    def combined_view(
        self,
        partial_board: "Board",
        inches: float = 14.0,
        dpi: int = 128,
        transparent: bool = False,
        width_ratios=(1.0, 1.6, 1.6),
        show_titles: bool = True,
        title_fontsize: int = 18,
        moves_remaining: Optional[int] = None,
        questions_remaining: Optional[int] = None,
    ):
        """Create a composite figure with (left-to-right):
        1) Ship Tracker, 2) Captain View (partial board), 3) Spotter View.

        Returns a matplotlib.figure.Figure that can be saved to PDF.
        """
        if not isinstance(partial_board, Board):
            raise ValueError("partial_board must be a Board object")
        if partial_board.size != self.size:
            raise ValueError("partial_board must be the same size as the board")

        def _fig_to_rgb_array(fig):
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)

        # Render the panels as standalone figures to avoid code duplication
        st_fig = self.ship_tracker_figure(
            partial_board, inches=3.0, dpi=dpi, transparent=transparent
        )
        cap_fig = Board._to_figure(
            board_array=partial_board.board,
            inches=4.0,
            dpi=dpi,
            transparent=transparent,
        )
        spot_fig = self.spotter_view(
            partial_board, inches=4.0, dpi=dpi, transparent=transparent
        )

        # Optional counters panel
        ct_fig = None
        if moves_remaining is not None or questions_remaining is not None:
            ct_fig = self.counters_figure(
                moves_remaining=moves_remaining,
                questions_remaining=questions_remaining,
                inches=3.0,
                dpi=dpi,
                transparent=transparent,
            )

        st_img = _fig_to_rgb_array(st_fig)
        cap_img = _fig_to_rgb_array(cap_fig)
        spot_img = _fig_to_rgb_array(spot_fig)
        ct_img = _fig_to_rgb_array(ct_fig) if ct_fig is not None else None

        # Create the composite figure
        fig = plt.figure(
            figsize=(inches, inches * 0.38), dpi=dpi, constrained_layout=True
        )
        if transparent:
            fig.patch.set_alpha(0.0)
        gs = fig.add_gridspec(1, 3, width_ratios=width_ratios)
        # Left column can be a subgrid when counters are present
        if ct_img is not None:
            left_sub = gs[0, 0].subgridspec(2, 1, height_ratios=(1.2, 2.2))
            ax1_top = fig.add_subplot(left_sub[0, 0])
            ax1_bot = fig.add_subplot(left_sub[1, 0])
        else:
            ax1_bot = fig.add_subplot(gs[0, 0])
            ax1_top = None
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # Place left counters (if any)
        if ax1_top is not None and ct_img is not None:
            ax1_top.imshow(ct_img)
            ax1_top.axis("off")
            ax1_top.set_aspect("equal")

        for ax, img in ((ax1_bot, st_img), (ax2, cap_img), (ax3, spot_img)):
            ax.imshow(img)
            ax.axis("off")
            ax.set_aspect("equal")

        if show_titles:
            title_style = dict(
                fontsize=title_fontsize, fontweight="bold", color="#1f2937"
            )
            ax2.set_title("Captain", **title_style)
            ax3.set_title("Spotter", **title_style)

        # Clean up child figs
        plt.close(st_fig)
        plt.close(cap_fig)
        plt.close(spot_fig)
        if ct_fig is not None:
            plt.close(ct_fig)

        return fig

    def counters_figure(
        self,
        moves_remaining: Optional[int] = None,
        questions_remaining: Optional[int] = None,
        inches: float = 3.0,
        dpi: int = 128,
        transparent: bool = False,
    ):
        """Render a small counters panel with Moves Left and Questions Left.

        Returns a matplotlib.figure.Figure.
        """
        fig_h_scale = 0.75
        fig, ax = plt.subplots(figsize=(inches, inches * fig_h_scale), dpi=dpi)
        if transparent:
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")

        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(10, 0)

        # Draw one rounded "pill" box
        def pill(x, y, w, h, label, value):
            bg = "#f3f4f6"  # light gray
            txt = "#1f2937"  # slate-800
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.2,rounding_size=0.3",
                    linewidth=0.8,
                    edgecolor="white",
                    facecolor=bg,
                )
            )
            ax.text(
                x + 0.4,
                y + h * 0.5,
                label,
                fontsize=14,
                color="#6b7280",
                va="center",
                ha="left",
            )
            ax.text(
                x + w - 0.4,
                y + h * 0.5,
                "—" if value is None else str(value),
                fontsize=14,
                fontweight="bold",
                color=txt,
                va="center",
                ha="right",
            )

        # Vertical layout (avoid right-edge clipping)
        margin_x = 0.7
        margin_y = 0.8
        box_w = 10 - 2 * margin_x
        box_h = 2.0
        vgap = 0.9
        y_top = margin_y
        y_bottom = y_top + box_h + vgap
        pill(margin_x, y_top, box_w, box_h, "Moves Left", moves_remaining)
        pill(margin_x, y_bottom, box_w, box_h, "Questions Left", questions_remaining)

        plt.close(fig)
        return fig

    @staticmethod
    def _to_figure(
        board_array: np.ndarray,
        inches: int = 6,
        dpi: int = 128,
        mode: str = "default",
        transparent: bool = False,
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
        # If transparent, make the figure and axes backgrounds transparent so that
        # saved images (or in-notebook renders) have no opaque background.
        if transparent:
            # Make figure patch fully transparent
            fig.patch.set_alpha(0.0)
            # Make axes background transparent
            ax.set_facecolor("none")
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
