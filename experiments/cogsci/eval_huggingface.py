import argparse
import asyncio
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from battleship.board import Board
from battleship.board import TRIAL_IDS
from battleship.models import SingleStepQuestionGenerationModel
from battleship.prompting import QuestionGenerationPrompt
from battleship.prompting import TranslationPrompt
from hfppl.llms import CachedCausalLM


RESULTS_FILENAME = "results.csv"
COMMAND_FILENAME = "command.txt"
PROMPTS_FILENAME = "prompts.json"


async def main(args):
    # Timekeeping
    time_start = time.time()

    # Bookkeeping
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name_escaped = args.model_name.split("/")[1]
    experiment_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.output_dir,
        f"{model_name_escaped}-{timestamp}",
    )
    os.makedirs(experiment_dir, exist_ok=True)
    results_filepath = os.path.join(experiment_dir, RESULTS_FILENAME)
    print(f"Results will be saved to: {results_filepath}")

    # Write the command to a file
    with open(os.path.join(experiment_dir, COMMAND_FILENAME), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # Instantiate LLM
    lm = CachedCausalLM.from_pretrained(args.model_name)
    lm.batch_size = args.batch_size

    # Divide samples into n_queries batches
    assert args.n_samples % args.batch_size == 0
    n_queries = args.n_samples // args.batch_size

    results_trial = []
    question_prompts, translation_prompts = [], []
    for trial_id in args.trial_ids:
        print("-" * 80)
        print(f"TRIAL {trial_id}")
        print("-" * 80)

        # Reset random seed
        rng = np.random.default_rng(args.random_seed)

        # Clear LM Trie cache
        lm.clear_cache()

        for query in range(n_queries):
            print(f"Query {query+1}/{n_queries}")

            # Clear LM vector cache
            lm.clear_kv_cache()

            # Each batch gets a new prompt
            question_prompt = QuestionGenerationPrompt(
                target_trial_id=trial_id,
                board_format=args.board_format,
                n_example_trials=args.q_n_example_trials,
                n_examples_per_trial=args.q_n_examples_per_trial,
                include_system_prompt=args.include_system_prompt,
                include_instructions=args.include_instructions,
                include_board=args.include_board,
                random_seed=rng,
            )

            question_prompt_data = question_prompt.to_dict()
            question_prompt_data["prompt_id"] = query
            question_prompts.append(question_prompt_data)

            translation_prompt = TranslationPrompt(
                target_trial_id=trial_id,
                target_question=None,
                n_example_trials=args.t_n_example_trials,
                n_examples_per_trial=args.t_n_examples_per_trial,
                include_system_prompt=args.include_system_prompt,
                include_instructions=args.include_instructions,
                random_seed=rng,
            )

            translation_prompt_data = translation_prompt.to_dict()
            translation_prompt_data["prompt_id"] = query
            translation_prompts.append(translation_prompt_data)

            # Caching speeds up performance, but may result in CUDA out of memory error.
            if args.q_cache_prompt:
                lm.cache_kv(lm.tokenizer.encode(str(question_prompt)))
            if args.t_cache_prompt:
                # Additionally, this currently degrades the quality of the translations for an unknown reason.
                lm.cache_kv(lm.tokenizer.encode(str(translation_prompt)))

            model = SingleStepQuestionGenerationModel(
                lm=lm,
                board=Board.from_trial_id(trial_id),
                question_prompt=str(question_prompt),
                translation_prompt=str(translation_prompt),
                verbose=args.verbose,
            )

            particles = [copy.deepcopy(model) for _ in range(args.batch_size)]
            results = await asyncio.gather(*[p.step() for p in particles])
            for data in results:
                data["trial_id"] = trial_id
                data["prompt_id"] = query

            results_trial.extend(results)

        df = pd.DataFrame(results_trial)
        print(df)

        df.to_csv(results_filepath, index=False)

        # Save prompt data to JSON file
        with open(os.path.join(experiment_dir, PROMPTS_FILENAME), "w") as f:
            json.dump(
                {
                    "question_prompts": question_prompts,
                    "translation_prompts": translation_prompts,
                },
                f,
                indent=4,
            )

    time_end = time.time()
    print(f"Total time: {time_end - time_start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--model_name", type=str, default="codellama/CodeLlama-7b-hf")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--trial_ids", type=int, nargs="+", default=TRIAL_IDS)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=True
    )
    # Question generation prompt
    parser.add_argument("--board_format", type=str, default="textual")
    parser.add_argument("--q_n_example_trials", type=int, default=3)
    parser.add_argument("--q_n_examples_per_trial", type=int, default=10)
    parser.add_argument(
        "--include_system_prompt", action=argparse.BooleanOptionalAction, default=False
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
    # Caching
    parser.add_argument(
        "--q_cache_prompt", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--t_cache_prompt", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()
    asyncio.run(main(args))
