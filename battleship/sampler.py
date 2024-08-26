from board import Board
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
import math
import random


class SeenShip(object):
    def __init__(self, board: Board, label) -> None:
        board = board.to_symbolic_array()
        self.label = label
        self.locations = np.where(board == label)
        self.length_seen = len(self.locations[0])
        location_tuples = [
            (self.locations[0][i], self.locations[1][i])
            for i in range(self.length_seen)
        ]

        self.location_tuples = location_tuples
        if self.length_seen > 1:
            if all([i == self.locations[0][0] for i in self.locations[0]]):
                self.orientation = ["H"]
                self.extremities = (
                    min(location_tuples, key=lambda item: item[1]),
                    max(location_tuples, key=lambda item: item[1]),
                )
                self.length = self.extremities[1][1] - self.extremities[0][1] + 1
            elif all([i == self.locations[1][0] for i in self.locations[1]]):
                self.orientation = ["V"]
                self.extremities = (
                    min(location_tuples, key=lambda item: item[0]),
                    max(location_tuples, key=lambda item: item[0]),
                )
                self.length = self.extremities[1][0] - self.extremities[0][0] + 1
            else:
                raise ValueError(f"{self.label} ship oriented incorrectly")
        elif self.length_seen == 1:
            self.orientation = ["H", "V"]
            self.extremities = (location_tuples[0], location_tuples[0])
            self.length = 1
        else:
            self.orientation = None
            self.extremities = (None, None)
            self.length = 0


def occlusion_fixing(starting_board: Board, ship_labels):
    symbolic_board = starting_board.to_symbolic_array()
    for ship in ship_labels:
        ship_part = SeenShip(starting_board, ship)
        ship_length = ship_part.length
        ship_extremities = ship_part.extremities
        if ship_length > 1:
            ship_tiles = []
            for i in range(ship_length):
                if ship_part.orientation[0] == "H":
                    ship_tiles.append(
                        (ship_extremities[0][0], ship_extremities[0][1] + i)
                    )
                if ship_part.orientation[0] == "V":
                    ship_tiles.append(
                        (ship_extremities[0][0] + i, ship_extremities[0][1])
                    )
            for tile in ship_tiles:
                if symbolic_board[tile[0]][tile[1]] in ["H", "W", ship]:
                    symbolic_board[tile[0]][tile[1]] = ship
                else:
                    return None
    return Board.from_symbolic_array(symbolic_board)


def grow_ship(partial_board: Board, ship_picked, length_picked, max_length_dict):
    partial_board = partial_board.to_symbolic_array()
    # print(f"[GS] attempting to grow {ship_picked.label}, orientation {ship_picked.orientation}")
    if ship_picked.length == 1:
        allowable_orientations = [
            orientation
            for orientation in ship_picked.orientation
            if max_length_dict[ship_picked.label][orientation] >= length_picked
        ]
        if len(allowable_orientations) == 1:
            ship_picked.orientation = allowable_orientations[0]
        else:
            ship_picked.orientation = random.choice(allowable_orientations)
        # print(f"[GROW SHIP] gave ship orientation {ship_picked.orientation}")
    else:
        ship_picked.orientation = ship_picked.orientation[0]

    # print(f"ship orientation H {ship_picked.orientation == 'H'}")
    if ship_picked.orientation == "H":
        # print(f"[GROW SHIP] ship with orientation H being grown to {length_picked}")
        row = partial_board[ship_picked.extremities[0][0], :]
        while (
            np.count_nonzero(row == ship_picked.label) < length_picked
        ):  # keep first steady, change second
            ship_picked = SeenShip(
                Board.from_symbolic_array(partial_board), ship_picked.label
            )
            valid_expansions = []
            left_extremity, right_extremity = (
                ship_picked.extremities[0],
                ship_picked.extremities[1],
            )
            if (
                left_extremity[1] - 1 >= 0
                and partial_board[left_extremity[0]][left_extremity[1] - 1] == "H"
            ):
                valid_expansions.append((left_extremity[0], left_extremity[1] - 1))
            if (
                right_extremity[1] + 1 < len(row)
                and partial_board[right_extremity[0]][right_extremity[1] + 1] == "H"
            ):
                valid_expansions.append((right_extremity[0], right_extremity[1] + 1))

            if len(valid_expansions) != 0:
                extension = random.choice(valid_expansions)
            else:
                return None
            # print(f"[GROW SHIP] assigning horizontal ship {ship_picked.label} extension {extension}")
            partial_board[extension[0]][extension[1]] = ship_picked.label
    # print(f"ship orientation V {ship_picked.orientation == 'V'}")
    if ship_picked.orientation == "V":
        # print(f"[GROW SHIP] ship with orientation V being grown to {length_picked}")
        column = partial_board[:, ship_picked.extremities[0][1]]
        while (
            np.count_nonzero(column == ship_picked.label) < length_picked
        ):  # keep second steady, change first
            ship_picked = SeenShip(
                Board.from_symbolic_array(partial_board), ship_picked.label
            )
            valid_expansions = []
            up_extremity, down_extremity = (
                ship_picked.extremities[0],
                ship_picked.extremities[1],
            )
            if (
                down_extremity[0] + 1 < len(column)
                and partial_board[down_extremity[0] + 1][down_extremity[1]] == "H"
            ):
                valid_expansions.append((down_extremity[0] + 1, down_extremity[1]))
            if (
                up_extremity[0] - 1 >= 0
                and partial_board[up_extremity[0] - 1][up_extremity[1]] == "H"
            ):
                valid_expansions.append((up_extremity[0] - 1, up_extremity[1]))
            # print(f"extremities down and up: {down_extremity, up_extremity}")
            # print(f"valid extensions {valid_expansions}")
            if len(valid_expansions) != 0:
                extension = random.choice(valid_expansions)
            else:
                return None
            # print(f"[GROW SHIP] assigning vertical ship {ship_picked.label} extension {extension}")
            partial_board[extension[0]][extension[1]] = ship_picked.label
            # print(f"[GS] New board: {partial_board}")
    return partial_board


def seed_board(board: Board, missing_ships):
    symbolic_board = board.to_symbolic_array()
    hidden_tiles = np.where(np.array(symbolic_board) == "H")
    location_tuples = [
        (hidden_tiles[0][i], hidden_tiles[1][i]) for i in range(len(hidden_tiles[0]))
    ]

    seeded_board = deepcopy(symbolic_board)
    seeds = random.sample(location_tuples, len(missing_ships))
    for index, seed in enumerate(seeds):
        seeded_board[seed[0]][seed[1]] = missing_ships[index]

    returnable_board = Board.from_symbolic_array(seeded_board)

    return returnable_board


def sample_board(partial_board: Board, ship_labels, ship_lengths):
    partial_board = occlusion_fixing(partial_board, ship_labels)
    if partial_board is None:
        return None

    temp_ships = [SeenShip(partial_board, ship) for ship in ship_labels]
    partial_board = seed_board(
        partial_board, [ship.label for ship in temp_ships if ship.length == 0]
    )
    ships = [SeenShip(partial_board, ship) for ship in ship_labels]

    partial_board = partial_board.to_symbolic_array()

    assigned_length_dict = {ship.label: None for ship in ships}

    while any(
        [assigned_length == None for assigned_length in assigned_length_dict.values()]
    ):

        if partial_board is None:
            return None

        unassigned_lengths = [
            length
            for length in ship_lengths
            if length not in assigned_length_dict.values()
        ]
        current_length_dict = {ship.label: ship.length for ship in ships}
        max_length_dict = {
            ship.label: {
                "V": np.count_nonzero(
                    partial_board[:, ship.extremities[0][1]] == ship.label
                )
                + np.count_nonzero(partial_board[:, ship.extremities[0][1]] == "H"),
                "H": np.count_nonzero(
                    partial_board[ship.extremities[0][0], :] == ship.label
                )
                + np.count_nonzero(partial_board[ship.extremities[0][0], :] == "H"),
            }
            for ship in ships
        }
        possible_length_dict = {
            ship.label: [
                length
                for length in unassigned_lengths
                if length >= current_length_dict[ship.label]
                and length
                <= max(
                    [max_length_dict[ship.label][orient] for orient in ship.orientation]
                )
            ]
            for ship in ships
        }
        forced_unassigned_ships = {
            ship_label: ship_lengths[0]
            for ship_label, ship_lengths in possible_length_dict.items()
            if len(ship_lengths) == 1 and assigned_length_dict[ship_label] == None
        }
        if len(forced_unassigned_ships.keys()) != 0:
            forced_ship = random.choice(list(forced_unassigned_ships.items()))
            assigned_length_dict[forced_ship[0]] = forced_ship[1]
            # print(f"[SAMPLE BOARD] forced assignment of {forced_ship[0]}", assigned_length_dict)

            ship = SeenShip(Board.from_symbolic_array(partial_board), forced_ship[0])

            if ship.length != forced_ship[1]:
                partial_board = grow_ship(
                    partial_board, ship, forced_ship[1], max_length_dict
                )
            else:
                pass
                # print(f"[SB] ship already of desired length, moving on")
            continue
        else:
            unforced_unassigned_ships = [
                ship.label for ship in ships if assigned_length_dict[ship.label] == None
            ]
            try:
                ship_label_picked = random.choice(unforced_unassigned_ships)
                length_picked = random.choice(possible_length_dict[ship_label_picked])
                assigned_length_dict[ship_label_picked] = length_picked
            except:
                return None

            ship_picked = SeenShip(
                Board.from_symbolic_array(partial_board), ship_label_picked
            )
            # print(f"[SAMPLE BOARD] unforced picked ship {ship_picked.label} of length {ship_picked.length}")

            if ship_picked.length != length_picked:
                partial_board = grow_ship(
                    partial_board, ship_picked, length_picked, max_length_dict
                )
            else:
                pass
            #    print(f"[SB] ship already of desired length, moving on")
            # print(assigned_length_dict)
            continue

    if partial_board is None:
        return None
    else:
        partial_board[partial_board == "H"] = "W"

        return Board.from_symbolic_array(partial_board)


def to_heatmap(
    board_list, ship_labels, inches: int = 6, dpi: int = 128, returnData=False
):
    board_list = [board.to_symbolic_array() for board in board_list]
    for board_index, board in enumerate(board_list):
        for row_index, row in enumerate(board):
            for item_index, item in enumerate(row):
                if item in ship_labels:
                    board_list[board_index][row_index][item_index] = 1
                else:
                    board_list[board_index][row_index][item_index] = 0

    heatmap = np.sum([board.astype(int) for board in board_list], axis=0)

    if returnData:
        return heatmap

    length = len(board_list[0])

    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    ax.matshow(heatmap, cmap="viridis")

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, length, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, length, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

    # Add labels
    ax.set_xticks(np.arange(0, length, 1))
    ax.set_yticks(np.arange(0, length, 1))
    ax.set_xticklabels(
        [chr(ord("A") + i) for i in np.arange(0, length, 1)],
        fontsize=24,
        fontweight="bold",
        color="#9b9c97",
    )
    ax.set_yticklabels(
        np.arange(1, length + 1, 1),
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
