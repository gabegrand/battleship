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
from battleship.scoring import _compute_score_batch
from battleship.scoring import compute_score_parallel


RESULTS_FILENAME = "results.csv"
COMMAND_FILENAME = "command.txt"


def main(args):
    time_start = time.time()
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.output_dir,
        f"grammar-sampling-{timestamp}",
    )
    os.makedirs(experiment_dir, exist_ok=True)
    results_filepath = os.path.join(experiment_dir, RESULTS_FILENAME)
    print(f"Results will be saved to: {results_filepath}")

    print(f"Sampling {args.n_samples} programs...")
    all_programs = sample_programs_from_grammar(
        args.n_samples, args.min_depth, args.max_depth
    )
    unique_programs = list(set([item[0] for item in all_programs]))
    print(
        f"Sampled {args.n_samples} programs ({len(unique_programs)} unique) in {time.time() - time_start:.2f}s"
    )

    results = []
    for trial_id in args.trial_ids:
        board_start_time = time.time()
        print(f"Running scoring for board {trial_id}")

        program_scores = compute_score_parallel(
            programs=unique_programs,
            board=Board.from_trial_id(trial_id),
            processes=args.processes,
            chunksize=args.chunksize,
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
            f"Finished scoring board {trial_id} in {time.time()-board_start_time:.2f}s"
        )

    df = pd.DataFrame(results)
    df.to_csv(results_filepath, index=False)
    print(
        f"Finished {args.n_samples}-shot sampling at depth {(args.min_depth, args.max_depth)} in {time.time() - time_start:.2f}s"
    )
    return df


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--min_depth", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--trial_ids", type=str, nargs="+", default=TRIAL_IDS)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--chunksize", type=int, default=16)
    args = parser.parse_args()

    main(args)
