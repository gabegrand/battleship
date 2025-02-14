from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from battleship.board import Board
from battleship.fast_sampler import FastSampler
from battleship.game import Decision


class Agent:
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)


class Captain(Agent):
    def decision(
        self,
        state: Board,
        history: List[Dict],
        questions_remaining: int,
        moves_remaining: int,
    ):
        raise NotImplementedError

    def question(self, state: Board, history: List[Dict]):
        raise NotImplementedError

    def move(self, state: Board, history: List[Dict]):
        raise NotImplementedError


class Spotter(Agent):
    def answer(
        self,
        state: Board,
        history: List[Dict],
        question: Tuple[int, int],
    ):
        raise NotImplementedError


class RandomCaptain(Captain):
    def decision(self, *args, **kwargs):
        return Decision.MOVE

    def move(self, state: Board, history: List[Dict]) -> Tuple[int, int]:
        hidden_tiles = np.argwhere(state.board == Board.hidden)
        if len(hidden_tiles) == 0:
            raise ValueError("No hidden tiles left.")
        coords = self.rng.choice(hidden_tiles)
        return tuple(coords)


class MAPCaptain(Captain):
    def __init__(self, seed: int = None, n_samples: int = 10000):
        super().__init__(seed)
        self.n_samples = n_samples

    def decision(self, *args, **kwargs):
        return Decision.MOVE

    def move(self, state: Board, history: List[Dict]) -> Tuple[int, int]:
        sampler = FastSampler(
            board=state,
            ship_lengths=Board.SHIP_LENGTHS,
            ship_labels=Board.SHIP_LABELS,
            seed=self.rng,
        )

        # Compute the raw posterior counts over board positions.
        posterior = sampler.compute_posterior(n_samples=self.n_samples, normalize=False)

        # For tiles that have already been revealed, force their probability to -infinity.
        posterior = posterior.astype(float)
        posterior[state.board != Board.hidden] = -np.inf

        # Select the tile with the maximum posterior probability (MAP estimate).
        flat_idx = int(np.argmax(posterior))

        # Map the flat index back to 2D coordinates.
        move_coords = np.unravel_index(flat_idx, state.board.shape)
        return tuple(move_coords)
