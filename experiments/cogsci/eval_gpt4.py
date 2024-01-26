import argparse
import asyncio
import json
import os
import sys
import time
from enum import StrEnum

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from battleship.board import Board
from battleship.board import TRIAL_IDS
from battleship.prompting import QuestionGenerationPrompt
from battleship.prompting import TranslationPrompt
from battleship.scoring import compute_score

client = OpenAI()

RESULTS_FILENAME = "results.csv"
COMMAND_FILENAME = "command.txt"
PROMPTS_FILENAME = "prompts.json"


class OpenAIModels(StrEnum):
    TEXT = "gpt-4"
    VISION = "gpt-4-vision-preview"


async def main(args):
    # Bookkeeping
    time_start = time.time()
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name_escaped = args.model_name.split("/")[0]
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

    assert args.n_samples % args.batch_size == 0
    n_queries = args.n_samples // args.batch_size

    results = []
    question_prompts, translation_prompts = [], []
    for trial_id in args.trial_ids:
        rng = np.random.default_rng(args.random_seed)

        print("-" * 80)
        print(f"TRIAL {trial_id}")
        print("-" * 80)

        for query in range(n_queries):
            question_prompt = QuestionGenerationPrompt(
                target_trial_id=trial_id,
                board_format=args.board_format,
                n_example_trials=args.q_n_example_trials,
                n_examples_per_trial=args.q_n_examples_per_trial,
                include_system_prompt=args.include_system_prompt,
                include_instructions=args.include_instructions,
                include_board=args.include_board,
                include_final_prefix=False,
                random_seed=rng,
            )

            question_prompt_data = question_prompt.to_dict()
            question_prompt_data["prompt_id"] = query
            question_prompts.append(str(question_prompt_data))

            translation_prompt = TranslationPrompt(
                target_question=None,
                target_trial_id=trial_id,
                n_example_trials=args.t_n_example_trials,
                n_examples_per_trial=args.t_n_examples_per_trial,
                include_system_prompt=args.include_system_prompt,
                include_instructions=args.include_instructions,
                include_board=args.include_board,
                include_final_prefix=False,
                random_seed=rng,
            )

            translation_prompt_data = translation_prompt.to_dict()
            translation_prompt_data["prompt_id"] = query
            translation_prompts.append(str(translation_prompt_data))

            if args.verbose:
                print("-" * 80)
                print(question_prompt)
                print("-" * 80)
                print(translation_prompt)
                print("-" * 80)

            completion = client.chat.completions.create(
                model=OpenAIModels.VISION
                if args.board_format == "visual"
                else OpenAIModels.TEXT,
                messages=question_prompt.to_chat_format(),
                n=args.batch_size,
                temperature=args.q_temperature,
                stop="\n",
            )

            questions = [
                str(completion.choices[i].message.content)
                .replace(QuestionGenerationPrompt.PREFIX_QUESTION, "")
                .strip()
                for i in range(args.batch_size)
            ]

            for i in range(args.batch_size):
                translation_prompt.target_question = questions[i]

                completion = client.chat.completions.create(
                    model=OpenAIModels.TEXT,
                    messages=translation_prompt.to_chat_format(),
                    n=1,
                    temperature=args.t_temperature,
                    stop="\n",
                )
                program_temp = (
                    str(completion.choices[0].message.content)
                    .replace(TranslationPrompt.PREFIX_CODE, "")
                    .strip()
                )
                score = compute_score(
                    program=program_temp, board=Board.from_trial_id(trial_id)
                )
                result = {
                    "trial_id": trial_id,
                    "prompt_id": query,
                    "question": questions[i],
                    "program": program_temp,
                    "score": score,
                }
                results.append(result)

        df = pd.DataFrame(results)
        print(df)

        df.to_csv(results_filepath, index=False)

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
    parser.add_argument("--model_name", type=str, default="gpt4")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--trial_ids", type=int, nargs="+", default=TRIAL_IDS)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )
    # Temperatures
    parser.add_argument("--q_temperature", type=float, default=1.0)
    parser.add_argument("--t_temperature", type=float, default=0.1)
    # Question generation prompt
    parser.add_argument("--board_format", type=str, default="textual")
    parser.add_argument("--q_n_example_trials", type=int, default=3)
    parser.add_argument("--q_n_examples_per_trial", type=int, default=10)
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

    args = parser.parse_args()
    asyncio.run(main(args))
