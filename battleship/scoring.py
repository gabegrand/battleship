import itertools
import os
from multiprocessing import Pool

import numpy as np
from eig import compute_eig_fast
from eig.battleship.program import ProgramSyntaxError
from tqdm import tqdm

from battleship.board import Board


def compute_score(program: str, board: Board):
    assert isinstance(program, str)
    assert isinstance(board, Board)

    try:
        score = compute_eig_fast(
            program,
            board.board,
            grid_size=6,
            ship_labels=[1, 2, 3],
            ship_sizes=[2, 3, 4],
            orientations=["V", "H"],
        )
    except ProgramSyntaxError:
        score = 0
    except RuntimeError:
        score = 0

    return np.round(score, decimals=6)


def _compute_score_batch(programs: list, board: Board, show_progress: bool = True):
    """Compute scores for a list of programs."""

    scores = []
    for program in tqdm(programs) if show_progress else programs:
        scores.append(compute_score(program, board))

    return scores


def compute_score_parallel(
    programs: list,
    board: Board,
    processes: int = None,
    n_chunks_per_process: int = 4,
    show_progress: bool = True,
):
    """Compute scores for a list of programs in parallel."""

    if processes is None:
        processes = os.cpu_count()

    # Split programs into chunks using np.array_split
    program_chunks = np.array_split(programs, int(processes * n_chunks_per_process))

    with Pool(processes=processes) as pool:
        job_list = [
            pool.apply_async(_compute_score_batch, (chunk, board, False))
            for chunk in program_chunks
        ]
        if show_progress:
            job_list = tqdm(job_list)
        scores_chunks = [job.get() for job in job_list]

    scores = list(itertools.chain.from_iterable(scores_chunks))

    return scores
