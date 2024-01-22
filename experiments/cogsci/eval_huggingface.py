import argparse
import os
import time
from math import ceil

import numpy as np
import pandas as pd
from eig.battleship import Parser
from tqdm import tqdm

from battleship.board import Board
from battleship.board import TRIAL_IDS
from battleship.grammar import BattleshipGrammar
from battleship.huggingface_llms import HuggingFaceModel
from battleship.prompting import QuestionGenerationPrompt
from battleship.prompting import TranslationPrompt
from battleship.scoring import compute_score
from battleship.scoring import compute_score_parallel


def main(args):
    # rng = np.random.default_rng(args.random_seed)

    # Instantiate LLM
    model = HuggingFaceModel(model_name=args.model_name)

    queries_per_batch = args.n_samples // args.batch_size

    results = []
    for trial_id in args.trial_ids:
        print("-" * 80)
        print(f"TRIAL {trial_id}")
        print("-" * 80)

        questions, programs = [], []

        # Generate questions
        q_prompt = QuestionGenerationPrompt(
            target_trial_id=trial_id,
            board_format=args.board_format,
            n_example_trials=args.q_n_example_trials,
            n_examples_per_trial=args.q_n_examples_per_trial,
            include_system_prompt=args.include_system_prompt,
            include_instructions=args.include_instructions,
            include_board=args.include_board,
            random_seed=args.random_seed,
        )

        for _ in range(queries_per_batch):
            questions.extend(model([str(q_prompt)] * args.batch_size))

        # Translate questions
        translation_prompt = TranslationPrompt(
            target_question=None,
            target_trial_id=trial_id,
            n_example_trials=args.t_n_example_trials,
            n_examples_per_trial=args.t_n_examples_per_trial,
            include_system_prompt=args.include_system_prompt,
            include_instructions=args.include_instructions,
            random_seed=args.random_seed,
        )

        for _ in range(queries_per_batch):
            programs.extend(model([str(translation_prompt)] * args.batch_size))

        # Save results
        for q, p in zip(questions, programs):
            results.append(
                {
                    "trial_id": trial_id,
                    "question": q,
                    "program": p,
                }
            )

        df = pd.DataFrame(results)
        print(df)

        # Readable output filename with timestamp
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        model_name_escaped = "/".split(args.model_name)[0]
        output_path = f"{args.output_dir}/{model_name_escaped}-{timestamp}.csv"

        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--model_name", type=str, default="codellama/CodeLlama-7b-hf")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--trial_ids", type=int, nargs="+", default=TRIAL_IDS)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default="results")
    # Question generation prompt
    parser.add_argument("--board_format", type=str, default="textual")
    parser.add_argument("--q_n_example_trials", type=int, default=3)
    parser.add_argument("--q_n_examples_per_trial", type=int, default=3)
    parser.add_argument(
        "--include_system_prompt", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include_instructions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include_board", action=argparse.BooleanOptionalAction, default=True
    )
    # Translation prompt
    parser.add_argument("--t_n_example_trials", type=int, default=12)
    parser.add_argument("--t_n_examples_per_trial", type=int, default=1)

    main(parser.parse_args())
