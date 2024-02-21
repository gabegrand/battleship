import argparse
import os
import time
from math import ceil
from typing import List

import pandas as pd
from tqdm import tqdm

from battleship.board import Board
from battleship.board import TRIAL_IDS
from battleship.grammar import BattleshipGrammar
from battleship.scoring import compute_score_parallel


RESULTS_FILENAME = "results.csv"
COMMAND_FILENAME = "command.txt"


def sample_programs_from_grammar(
    n_samples: int = 10000, min_depth: int = 1, max_depth: int = 16
):
    grammar = BattleshipGrammar(include_lambdas=False)
    samples = []
    while len(samples) < n_samples:
        for _ in range(n_samples - len(samples)):
            program_data = grammar.sample(min_depth=min_depth, max_depth=max_depth)
            if program_data is not None:
                samples.append(program_data)
    return samples


def sample_and_score(
    processes=os.cpu_count(),
    n_samples: int = 10000,
    min_depth: int = 2,
    max_depth: int = 16,
    trial_ids: List = TRIAL_IDS,
    output_dir: str = "results",
):
    time_start = time.time()
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        output_dir,
        f"grammar-sampling-{timestamp}",
    )
    os.makedirs(experiment_dir, exist_ok=True)
    results_filepath = os.path.join(experiment_dir, RESULTS_FILENAME)
    print(f"Results will be saved to: {results_filepath}")

    print(f"Sampling {n_samples} programs...")
    all_programs = sample_programs_from_grammar(n_samples, min_depth, max_depth)
    unique_programs = list(set([item[0] for item in all_programs]))
    print(
        f"Finished sampling {n_samples} programs in {time.time() - time_start} seconds"
    )

    results = []
    for trial_id in trial_ids:
        board_start_time = time.time()
        print(f"Running scoring for board {trial_id}")
        program_scores = compute_score_parallel(
            programs=unique_programs,
            board=Board.from_trial_id(trial_id),
            processes=processes,
            show_progress=True,
        )
        program_to_score = dict(zip(unique_programs, program_scores))

        for program, depth in all_programs:
            results.append(
                {
                    "trial_id": trial_id,
                    "program": program,
                    "score": program_to_score[program],
                    "depth": depth,
                }
            )
        print(
            f"Finished scoring in {round(time.time()-board_start_time,2)}s from the start"
        )

    df = pd.DataFrame(results)
    df.to_csv(results_filepath, index=False)
    print(
        f"finished {n_samples}-shot sampling at depth {(min_depth,max_depth)} in time {time.time() - time_start}"
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    df = sample_and_score(
        processes=os.cpu_count(), n_samples=10, min_depth=2, max_depth=16
    )
