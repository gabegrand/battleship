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


def compute_score_parallel(
    programs: list, board: Board, processes: int = None, show_progress: bool = True
):
    """Compute scores for a list of programs in parallel."""

    if processes is None:
        processes = os.cpu_count()

    with Pool(processes=processes) as pool:
        job_list = [
            pool.apply_async(compute_score, (program, board)) for program in programs
        ]
        if show_progress:
            job_list = tqdm(job_list)
        scores = [job.get() for job in job_list]

    return scores
