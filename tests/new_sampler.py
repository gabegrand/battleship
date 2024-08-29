import random
from collections import Counter
from copy import deepcopy

import numpy as np

from board import Board
from sampler import *


def alt_sampler(partial_board, ship_labels, ship_lengths):
    """Samples boards as follows:
    1) calculates all gaps for all possible lengths of the ships both horizontally and vertically
    2) picks a random ship
    3) at random, picks a gap that is valid for the ship to be in
       (i.e. not of a length that's already assigned to another ship)
    4) goes back to step 1), and ends when every ship has a length assigned to it
    """
    partial_board = occlusion_fixing(partial_board, ship_labels)
    if partial_board is None:
        return None
    ships = [SeenShip(partial_board, label) for label in ship_labels]
    partial_board = partial_board.to_symbolic_array()

    ship_lengths = sorted(ship_lengths)

    current_length_dict = {ship.label: ship.length for ship in ships}
    assigned_length_dict = {ship: None for ship in ships}

    while any(
        [i is None for i in assigned_length_dict.values()]
    ):  # while there exists a ship that has no length to it
        possible_placements = {}
        for label in ["H"] + ship_labels:  # for every ship, as well as the hidden tiles
            board_rows = partial_board
            board_columns = partial_board.transpose()
            length_gaps = {}
            for length in ship_lengths:  # for all possible lengths L:
                gaps = {}

                if (
                    label == "H"
                ):  # finds all the spaces of length L where there is no ship at all -- we can place new ships here
                    gaps["horizontal"] = {
                        row_idx: [
                            idx
                            for idx in range(len(row) - length + 1)
                            if all([i == "H" for i in row[idx : idx + (length)]])
                        ]
                        for row_idx, row in enumerate(board_rows)
                    }
                    gaps["vertical"] = {
                        col_idx: [
                            idx
                            for idx in range(len(column) - length + 1)
                            if all([i == "H" for i in column[idx : idx + (length)]])
                        ]
                        for col_idx, column in enumerate(board_columns)
                    }
                else:  # finds all the spaces of length L where a gap is formed out of just the specific ship and hidden tiles -- we can grow existing ships here
                    gaps["horizontal"] = {
                        row_idx: [
                            idx
                            for idx in range(len(row) - length + 1)
                            if Counter(row[idx : idx + (length)])[label]
                            >= current_length_dict[label]
                            and all(
                                [i in ["H", label] for i in row[idx : idx + (length)]]
                            )
                        ]
                        for row_idx, row in enumerate(board_rows)
                    }
                    gaps["vertical"] = {
                        col_idx: [
                            idx
                            for idx in range(len(column) - length + 1)
                            if Counter(column[idx : idx + (length)])[label]
                            >= current_length_dict[label]
                            and all(
                                [
                                    i in ["H", label]
                                    for i in column[idx : idx + (length)]
                                ]
                            )
                        ]
                        for col_idx, column in enumerate(board_columns)
                    }

                length_gaps[length] = gaps
            possible_placements[label] = (
                length_gaps  # saves everything in a nested dictionary
            )
            # Dict structure: possible_placements[ship label][length of gap]["horiztonal"/"vertical"][row/column number] = [starting index of the gap in the row/col]

        unassigned_ships = [
            ship
            for ship, assigned_length in assigned_length_dict.items()
            if assigned_length is None
        ]
        random_unassigned_ship = random.choice(unassigned_ships) #picks random unassigned ship

        if random_unassigned_ship.length == 0: #if it's not on the board, place it in one of the gaps where there are no other ships
            gaps = possible_placements["H"]
            possible_gaps = []
            gaps = {
                gap_length: orientation_dict
                for (gap_length, orientation_dict) in gaps.items()
                if gap_length not in list(assigned_length_dict.values())
            }
            for gap_length, orientation_dict in gaps.items():
                for orientation, line_dict in orientation_dict.items():
                    for line, start_indices in line_dict.items():
                        for start_index in start_indices:
                            possible_gaps.append(
                                [gap_length, orientation, line, start_index]
                            )
        else: #if it is present on the board, place it in a gap containing the part of the ship that's already on the board
            gaps = possible_placements[random_unassigned_ship.label]

            possible_gaps = []
            gaps = {
                gap_length: orientation_dict
                for (gap_length, orientation_dict) in gaps.items()
                if gap_length >= current_length_dict[random_unassigned_ship.label]
                and gap_length not in list(assigned_length_dict.values())
            }
            for gap_length, orientation_dict in gaps.items():
                if len(random_unassigned_ship.orientation) == 2:
                    random_unassigned_ship.orientation = random.choice(
                        random_unassigned_ship.orientation
                    )
                else:
                    random_unassigned_ship.orientation = (
                        random_unassigned_ship.orientation[0]
                    )

                if random_unassigned_ship.orientation == "H":
                    orientation = "horizontal"
                else:
                    orientation = "vertical"

                line_dict = orientation_dict[orientation]
                for line, start_indices in line_dict.items():
                    for start_index in start_indices:
                        possible_gaps.append(
                            [gap_length, orientation, line, start_index]
                        )

        if len(possible_gaps) != 0: #if there's a possible gap, place the ship there
            random_gap = random.choice(possible_gaps)
            assigned_length_dict[random_unassigned_ship] = random_gap[0]
        else: #otherwise return None -- the sample has failed
            return None

        if random_gap[1] == "horizontal":
            new_board = deepcopy(partial_board)
        if random_gap[1] == "vertical":
            new_board = deepcopy(partial_board.transpose())

        new_board[random_gap[2]] = (
            new_board[random_gap[2]][0 : random_gap[3]].tolist()
            + [random_unassigned_ship.label] * random_gap[0]
            + new_board[random_gap[2]][random_gap[3] + random_gap[0] :].tolist()
        ) #replaces the old row/column with the modified one

        if random_gap[1] == "vertical":
            new_board = new_board.transpose()

        partial_board = new_board

    partial_board[partial_board == "H"] = "W"
    return Board.from_symbolic_array(partial_board)
