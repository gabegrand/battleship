import itertools
import os
from multiprocessing import get_context
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from battleship.board import Board
from eig import compute_eig_fast
from eig.battleship.program import ProgramSyntaxError


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
        return None
    except RuntimeError:
        return None

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
    chunksize: int = 10,
    show_progress: bool = True,
):
    """Compute scores for a list of programs in parallel.

    `chunksize` is the number of programs to process in each worker (single CPU core).

    To reduce memory overhead, we periodically collect the results from the workers and
    free the memory to create a fresh Pool.
    """

    if processes is None:
        processes = os.cpu_count()

    meta_chunks = split_list(programs, chunksize * processes)
    if show_progress:
        meta_chunks = tqdm(meta_chunks)

    scores_chunks = []
    for meta_chunk in meta_chunks:
        chunks = split_list(meta_chunk, chunksize)
        with get_context("spawn").Pool(processes=processes) as pool:
            job_list = [
                pool.apply_async(_compute_score_batch, (chunk, board, False))
                for chunk in chunks
            ]
            scores_chunks.extend([job.get() for job in job_list])

    scores = list(itertools.chain.from_iterable(scores_chunks))

    return scores


def split_list(arr, chunksize):
    """Split a list into chunks of chunksize."""
    # Ensure the chunksize is at least 1 and not larger than the list itself
    chunksize = max(1, min(chunksize, len(arr)))
    return [arr[i : i + chunksize] for i in range(0, len(arr), chunksize)]
