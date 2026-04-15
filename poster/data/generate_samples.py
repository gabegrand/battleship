"""Generate 3 sample boards for the poster diagram.

Requirements:
- All samples consistent with the visible state of board B11's spotter view
- Purple (3) and Orange (4) are sunk/revealed
- Two remaining ships: Red and Green, lengths 4 and 5 (either assignment)
- 2 samples where "Is there another ship near yellow?" → Yes
- 1 sample where "Is there another ship near yellow?" → No
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "poster", "figures")
sys.path.insert(0, REPO_ROOT)

import numpy as np
from battleship.board import Board
from battleship.fast_sampler import FastSampler

# --- Step 1: Load the true board B11 ---
true_board = Board.from_text_file(
    os.path.join(REPO_ROOT, "experiments", "collaborative", "contexts", "board_B11.txt")
)
print("True board B11:")
print(true_board.board)
print()

# Ship positions in B11:
# Orange (4): (0,4), (1,4), (2,4), (3,4) — length 4, vertical
# Purple (3): (1,5), (2,5) — length 2, vertical
# Red (1):    (4,0), (4,1), (4,2), (4,3) — length 4, horizontal
# Green (2):  (6,1), (6,2), (6,3), (6,4), (6,5) — length 5, horizontal

# --- Step 2: Construct partial board from the spotter view image ---
# H=-1 (hidden), W=0 (water/miss), 3=purple, 4=orange
partial = np.full((8, 8), Board.hidden, dtype=int)

# Orange ship tiles (all revealed/sunk)
partial[0, 4] = 4
partial[1, 4] = 4
partial[2, 4] = 4
partial[3, 4] = 4

# Purple ship tiles (all revealed/sunk)
partial[1, 5] = 3
partial[2, 5] = 3

# Water tiles (misses) — visible in the spotter view image
# Row A: A3, A6
partial[0, 2] = 0
partial[0, 5] = 0
# Row C: C2
partial[2, 1] = 0
# Row D: D4, D6
partial[3, 3] = 0
partial[3, 5] = 0
# Row E: E5
partial[4, 4] = 0
# Row F: F6
partial[5, 5] = 0
# Row H: H4
partial[7, 3] = 0

partial_board = Board(partial)
print("Partial board (Captain's view):")
print(partial_board.board)
print()

# --- Step 3: Define ship tracker ---
# Purple (length 2) and Orange (length 4) are sunk
# Red and Green are unsunk, with lengths 4 and 5
ship_tracker = [
    (2, "P"),  # Purple, length 2, SUNK
    (4, "O"),  # Orange, length 4, SUNK
    (4, None),  # Unsunk ship, length 4
    (5, None),  # Unsunk ship, length 5
]


# --- Step 4: Define the "near yellow" check ---
def is_near_yellow(board_array, partial_array):
    """Check: 'Is there another ship near yellow?'
    Returns True if any non-sunk ship has a tile adjacent to the orange ship."""
    found = set(partial_array[partial_array > 0].flatten())  # sunk ship IDs
    yellow_tiles = np.argwhere(board_array == 4)  # orange/yellow ship

    for x, y in yellow_tiles:
        # Check all 4 neighbors (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                ship = board_array[nx, ny]
                if ship > 0 and ship not in found:
                    return True
    return False


# --- Step 5: Sample boards and categorize ---
yes_samples = []
no_samples = []

for seed in range(10000):
    sampler = FastSampler(
        board=partial_board,
        ship_tracker=ship_tracker,
        ship_lengths=[
            2,
            4,
            4,
            5,
        ],  # Match the actual game: P=2, O=4, plus remaining 4 and 5
        ship_labels=["R", "G", "P", "O"],
        seed=seed,
    )
    sample = sampler.populate_board()
    if sample is None:
        continue

    result = is_near_yellow(sample.board, partial_board.board)
    if result and len(yes_samples) < 10:
        yes_samples.append((seed, sample))
    elif not result and len(no_samples) < 10:
        no_samples.append((seed, sample))

    if len(yes_samples) >= 10 and len(no_samples) >= 10:
        break

print(f"Found {len(yes_samples)} YES samples, {len(no_samples)} NO samples")
print()

# --- Step 6: Print candidates ---
for label, samples in [("YES", yes_samples), ("NO", no_samples)]:
    print(f"=== {label} samples ===")
    for seed, sample in samples[:5]:
        print(f"Seed {seed}:")
        print(sample.board)
        # Check ship lengths
        for ship_id, ship_name in [(1, "Red"), (2, "Green")]:
            count = np.sum(sample.board == ship_id)
            print(f"  {ship_name}: length {count}")
        print()

# --- Step 7: Select final 3 and render ---
# Pick 2 yes + 1 no with diverse ship placements
selected = []
if len(yes_samples) >= 2:
    selected.append(("s1_yes", yes_samples[0]))
    selected.append(("s3_yes", yes_samples[1]))
if len(no_samples) >= 1:
    selected.append(("s2_no", no_samples[0]))

for name, (seed, sample) in selected:
    print(f"Rendering {name} (seed={seed})...")
    fig = sample.to_figure(inches=3, dpi=300)
    outpath = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.05)
    print(f"  Saved to {outpath}")

print("\nDone!")
