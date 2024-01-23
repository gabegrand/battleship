import argparse
import asyncio
import os
import time
from math import ceil

import numpy as np
import pandas as pd
from eig.battleship import Parser
from tqdm import tqdm

from battleship.board import Board
from battleship.board import TRIAL_IDS
from battleship.huggingface_llms import HuggingFaceModel
from battleship.models import SingleStepQuestionGenerationModel
from battleship.prompting import QuestionGenerationPrompt
from battleship.prompting import TranslationPrompt
from battleship.scoring import compute_score
from battleship.scoring import compute_score_parallel
from hfppl.llms import CachedCausalLM


async def main(args):
    # Bookkeeping
    time_start = time.time()
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name_escaped = args.model_name.split("/")[0]
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.output_dir,
        f"{model_name_escaped}-{timestamp}.csv",
    )
    print(f"Script will save results to: {output_path}")

    rng = np.random.default_rng(args.random_seed)

    # Instantiate LLM
    lm = CachedCausalLM.from_pretrained(args.model_name)
    lm.batch_size = args.batch_size

    results = []
    for trial_id in args.trial_ids:
        print("-" * 80)
        print(f"TRIAL {trial_id}")
        print("-" * 80)

        lm.clear_cache()
        lm.clear_kv_cache()

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

        translation_prompt = TranslationPrompt(
            target_trial_id=trial_id,
            target_question=None,
            n_example_trials=args.t_n_example_trials,
            n_examples_per_trial=args.t_n_examples_per_trial,
            include_system_prompt=args.include_system_prompt,
            include_instructions=args.include_instructions,
            random_seed=rng,
        )

        # Caching speeds up performance, but may result in CUDA out of memory error.
        if args.q_cache_prompt:
            lm.cache_kv(lm.tokenizer.encode(str(question_prompt)))
        if args.t_cache_prompt:
            # Additionally, this currently degrades the quality of the translations for an unknown reason.
            lm.cache_kv(lm.tokenizer.encode(str(translation_prompt)))

        for _ in range(args.n_samples):
            model = SingleStepQuestionGenerationModel(
                lm=lm,
                board=Board.from_trial_id(trial_id),
                question_prompt=str(question_prompt),
                translation_prompt=str(translation_prompt),
                verbose=args.verbose,
            )

            result = await model.step()

            results.append(result)

        df = pd.DataFrame(results)
        df.insert(0, "trial_id", trial_id)
        print(df)

        df.to_csv(output_path, index=False)

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
    # Caching
    parser.add_argument(
        "--q_cache_prompt", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--t_cache_prompt", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()
    asyncio.run(main(args))
