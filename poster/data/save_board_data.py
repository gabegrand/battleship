"""Save board data (true board, partial board, and samples) for the poster diagram."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

import json
import numpy as np
from battleship.board import Board
from battleship.fast_sampler import FastSampler

# --- True board B11 ---
true_board = Board.from_text_file(
    os.path.join(REPO_ROOT, "experiments", "collaborative", "contexts", "board_B11.txt")
)
true_board.to_text_file(f"{DATA_DIR}/true_board_B11.txt")

# --- Partial board (Captain's view from spotter image) ---
partial = np.full((8, 8), Board.hidden, dtype=int)

# Orange ship tiles (all revealed/sunk)
partial[0, 4] = 4
partial[1, 4] = 4
partial[2, 4] = 4
partial[3, 4] = 4

# Purple ship tiles (all revealed/sunk)
partial[1, 5] = 3
partial[2, 5] = 3

# Water tiles (misses visible in spotter view image)
partial[0, 2] = 0  # A3
partial[0, 5] = 0  # A6
partial[2, 1] = 0  # C2
partial[3, 3] = 0  # D4
partial[3, 5] = 0  # D6
partial[4, 4] = 0  # E5
partial[5, 5] = 0  # F6
partial[7, 3] = 0  # H4

partial_board = Board(partial)
partial_board.to_text_file(f"{DATA_DIR}/partial_board.txt")

# --- Samples ---
# Ship tracker: Purple (len 2) and Orange (len 4) are sunk
ship_tracker = [
    (2, "P"),  # Purple, length 2, SUNK
    (4, "O"),  # Orange, length 4, SUNK
    (4, None),  # Unsunk ship, length 4
    (5, None),  # Unsunk ship, length 5
]

# Selected seeds from sampling run
selected = [
    ("s1", 67, True),  # seed=67, answer=Yes
    ("s2", 6, False),  # seed=6, answer=No
    ("s3", 87, True),  # seed=87, answer=Yes
]

metadata = {
    "source_board": "B11",
    "question": "Is there another ship near yellow?",
    "sunk_ships": {"P": 2, "O": 4},
    "remaining_ship_lengths": [4, 5],
    "samples": [],
}

for name, seed, answer in selected:
    sampler = FastSampler(
        board=partial_board,
        ship_tracker=ship_tracker,
        ship_lengths=[2, 4, 4, 5],
        ship_labels=["R", "G", "P", "O"],
        seed=seed,
    )
    sample = sampler.populate_board()
    sample.to_text_file(f"{DATA_DIR}/sample_{name}.txt")

    # Record ship info
    r_len = int(np.sum(sample.board == 1))
    g_len = int(np.sum(sample.board == 2))

    metadata["samples"].append(
        {
            "name": name,
            "seed": seed,
            "answer": answer,
            "red_length": r_len,
            "green_length": g_len,
        }
    )
    print(
        f"  {name}: seed={seed}, answer={'Y' if answer else 'N'}, R={r_len}, G={g_len}"
    )

# Write metadata
with open(f"{DATA_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nAll data written to {DATA_DIR}/")
