#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import sys
import uuid
from copy import deepcopy
from multiprocessing.dummy import Pool
from pathlib import Path
from random import randint
from time import time

import numpy as np
import pandas as pd

from battleship.agents import CACHE_DIR
from battleship.agents import Counter
from battleship.agents import HUMAN_SUMMARY_DIR
from battleship.agents import PROMPTS_DIR
from battleship.agents import RESULTS_DIR
from battleship.agents import STAGE_DIR
from battleship.board import Board
from battleship.captains import AlwaysMoveDecisionStrategy
from battleship.captains import Captain
from battleship.captains import EIGQuestionStrategy
from battleship.captains import LLMDecisionStrategy
from battleship.captains import LLMMoveStrategy
from battleship.captains import LLMQuestionStrategy
from battleship.captains import MAPMoveStrategy
from battleship.captains import ProbabilisticDecisionStrategy
from battleship.captains import RandomMoveStrategy
from battleship.game import BattleshipGame
from battleship.spotters import CodeSpotterModel

HUMAN_MAX_QUESTIONS = 15


def get_human_results(gold_annotations_path, round_data_path):
    stage_df = pd.read_csv(gold_annotations_path)
    round_df = pd.read_csv(round_data_path)

    board_ids = round_df[["id", "board_id", "questionsRemaining"]]
    filtered_stage_df = stage_df[
        [
            "roundID",
            "index",
            "questionID",
            "messageText",
            "messageType",
            "occTiles",
            "gold_answer",
        ]
    ]
    df = filtered_stage_df.merge(
        board_ids, left_on="roundID", right_on="id", how="left"
    )

    question_counts_df = (
        df[df["messageType"] == "question"].groupby("roundID").size().reset_index()
    )

    df = df.merge(question_counts_df, on="roundID", how="left")
    result = df.loc[df.groupby("roundID")["index"].idxmax()][
        ["roundID", "occTiles", "board_id", "questionID", "questionsRemaining"]
    ]
    # GG: Why is this needed?
    result = result[
        result["occTiles"] != str(np.full((8, 8), -1).tolist()).replace(" ", "")
    ]

    data = []
    for roundID, occTiles, board_id in zip(
        result["roundID"], result["occTiles"], result["board_id"]
    ):
        board_true = Board.from_trial_id(board_id)
        board_partial = Board.from_occ_tiles(occTiles)
        scores = board_true.score(board_partial)

        questions_asked = HUMAN_MAX_QUESTIONS - int(
            result[result["roundID"] == roundID]["questionsRemaining"].values[0]
        )

        result_row = {
            "roundId": roundID,
            "captainType": "human",
            "boardId": board_id,
            "hits": int(scores["hits"]),
            "misses": int(scores["misses"]),
            "precision": float(scores["precision"]),
            "recall": float(scores["recall"]),
            "f1_score": float(scores["f1_score"]),
            "is_won": bool(scores["is_won"]),
            "question_count": questions_asked,
        }
        data.append(result_row)

        temp_filename = f"human_{roundID}_{uuid.uuid4()}.json"
        temp_path = os.path.join(HUMAN_SUMMARY_DIR, temp_filename)
        with open(temp_path, "w") as f:
            json.dump(result_row, f, indent=2)

    return data


def create_captain(
    captain_type,
    seed,
    model,
    use_cache,
    map_samples=None,
    prob_q_prob=None,
    eig_samples=None,
    eig_k=None,
):
    # Initialize spotter for EIG captains
    eig_spotter = CodeSpotterModel(
        board_id="B01",
        board_experiment="collaborative",
        model_string=model,
        temperature=None,
        use_cot=True,
    )

    if captain_type == "RandomCaptain":
        return Captain(
            decision_strategy=AlwaysMoveDecisionStrategy(),
            move_strategy=RandomMoveStrategy(rng=np.random.default_rng(seed)),
            question_strategy=None,
            seed=seed,
            use_cache=use_cache,
            round_id=None,
        )

    elif captain_type == "MAPCaptain":
        return Captain(
            decision_strategy=AlwaysMoveDecisionStrategy(),
            move_strategy=MAPMoveStrategy(
                rng=np.random.default_rng(seed), n_samples=map_samples
            ),
            question_strategy=None,
            seed=seed,
            use_cache=use_cache,
            round_id=None,
        )

    elif captain_type == "ProbabilisticCaptain":
        return Captain(
            decision_strategy=ProbabilisticDecisionStrategy(q_prob=prob_q_prob),
            move_strategy=LLMMoveStrategy(model_string=model, use_cot=False),
            question_strategy=LLMQuestionStrategy(model_string=model, use_cot=False),
            seed=seed,
            model_string=model,
            use_cache=use_cache,
            round_id=None,
        )

    elif captain_type == "ProbabilisticCaptain_cot":
        return Captain(
            decision_strategy=ProbabilisticDecisionStrategy(q_prob=prob_q_prob),
            move_strategy=LLMMoveStrategy(model_string=model, use_cot=True),
            question_strategy=LLMQuestionStrategy(model_string=model, use_cot=True),
            seed=seed,
            model_string=model,
            use_cache=use_cache,
            round_id=None,
        )

    elif captain_type == "LLMDecisionCaptain":
        return Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=False),
            move_strategy=LLMMoveStrategy(
                model_string=model,
                use_cot=False,
            ),
            question_strategy=LLMQuestionStrategy(
                model_string=model,
                use_cot=False,
                spotter=eig_spotter,
                rng=np.random.default_rng(seed),
            ),
            seed=seed,
            model_string=model,
            use_cache=use_cache,
            round_id=None,
        )

    elif captain_type == "LLMDecisionCaptain_cot":
        return Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=True),
            move_strategy=LLMMoveStrategy(model_string=model, use_cot=True),
            question_strategy=LLMQuestionStrategy(
                model_string=model,
                use_cot=True,
                spotter=eig_spotter,
                rng=np.random.default_rng(seed),
            ),
            seed=seed,
            model_string=model,
            use_cache=use_cache,
            round_id=None,
        )

    elif captain_type == "EIGCaptain":
        return Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=False),
            move_strategy=LLMMoveStrategy(model_string=model, use_cot=False),
            question_strategy=EIGQuestionStrategy(
                model_string=model,
                spotter=eig_spotter,
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=False,
            ),
            seed=seed,
            model_string=model,
            use_cache=use_cache,
            round_id=None,
        )

    elif captain_type == "EIGCaptain_cot":
        return Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=True),
            move_strategy=LLMMoveStrategy(model_string=model, use_cot=True),
            question_strategy=EIGQuestionStrategy(
                model_string=model,
                spotter=eig_spotter,
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=True,
            ),
            seed=seed,
            model_string=model,
            use_cache=use_cache,
            round_id=None,
        )

    elif captain_type == "MAPEIGCaptain":
        return Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=False),
            move_strategy=MAPMoveStrategy(
                rng=np.random.default_rng(seed), n_samples=eig_samples
            ),
            question_strategy=EIGQuestionStrategy(
                model_string=model,
                spotter=eig_spotter,
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=False,
            ),
            seed=seed,
            model_string=model,
            use_cache=use_cache,
            round_id=None,
        )

    elif captain_type == "MAPEIGCaptain_cot":
        return Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=True),
            move_strategy=MAPMoveStrategy(
                rng=np.random.default_rng(seed), n_samples=eig_samples
            ),
            question_strategy=EIGQuestionStrategy(
                model_string=model,
                spotter=eig_spotter,
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=True,
            ),
            seed=seed,
            model_string=model,
            use_cache=use_cache,
            round_id=None,
        )

    else:
        raise ValueError(f"Unknown captain type: {captain_type}")


def run_single_agent_game(args):
    round_id, cap_name, captain, seed, board_id, max_questions, max_moves, model = args

    # Update spotter board ID for EIG captains
    if hasattr(captain, "question_strategy") and hasattr(
        captain.question_strategy, "spotter"
    ):
        captain.question_strategy.spotter.board_id = board_id

    # Set the captain's seed
    captain.seed = seed

    print(f"{cap_name} started with {board_id} & seed {seed}")
    board = Board.from_trial_id(board_id)

    decision_counter = Counter()
    index_counter = Counter()

    captain.index_counter = index_counter
    captain.decision_counter = decision_counter

    spotter = CodeSpotterModel(
        board_id,
        "collaborative",
        use_cache=True,
        model_string=model,
        temperature=None,
        use_cot=True,
        decision_counter=decision_counter,
        index_counter=index_counter,
        round_id=round_id,
    )

    # Write round information to JSON file
    round_info = {
        "id": round_id,
        "boardId": board_id,
        "seed": seed,
        "captainModel": cap_name,
        "spotterModel": spotter.__class__.__name__,
    }

    # Create a unique filename for the round info
    round_filename = f"round_{round_id}.json"
    round_dir = os.path.join(RESULTS_DIR, "rounds")
    os.makedirs(round_dir, exist_ok=True)

    with open(os.path.join(round_dir, round_filename), "w") as f:
        json.dump(round_info, f, indent=2)

    captain.round_id = round_id

    game = BattleshipGame(
        board_target=board,
        max_questions=max_questions,
        max_moves=max_moves,
        captain=captain,
        spotter=spotter,
    )
    game.play()
    print(f"{cap_name} finished with {board_id} & seed {seed}")

    scores = game.score()

    # Ensure consistent column names in the result data
    result = {
        "roundId": round_id,
        "captainType": cap_name,
        "boardId": board_id,
        "hits": int(game.hits),
        "misses": int(game.misses),
        "is_won": bool(game.is_won()),
        "question_count": game.question_count,
        "precision": float(scores["precision"]),
        "recall": float(scores["recall"]),
        "f1_score": float(scores["f1_score"]),
    }

    # Create a unique filename for each result to avoid race conditions
    safe_model_name = model.replace("/", "-")
    temp_filename = f"{safe_model_name}_{cap_name}_{round_id}"
    temp_path = os.path.join(CACHE_DIR, temp_filename + "_summary.json")

    # Gather all prompt files for this round_id
    prompts_path = os.path.join(CACHE_DIR, temp_filename + "_prompts.json")
    prompt_files = glob.glob(os.path.join(PROMPTS_DIR, f"prompt_{round_id}*.json"))
    prompts = []
    for pf in prompt_files:
        with open(pf, "r") as f:
            prompts.append(json.load(f))
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, indent=2)

    # Gather all stage files for this round_id
    stage_path = os.path.join(CACHE_DIR, temp_filename + "_stage.json")
    stage_files = glob.glob(os.path.join(STAGE_DIR, f"stage_{round_id}*.json"))
    stages = []
    for sf in stage_files:
        with open(sf, "r") as f:
            stages.append(json.load(f))
    with open(stage_path, "w") as f:
        json.dump(stages, f, indent=2)

    # Write individual result to temp file
    with open(temp_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Battleship Captain benchmarks")

    # Data paths
    parser.add_argument(
        "--gold-annotations",
        type=str,
        default="/home/ubuntu/repo_battleship/gold-v2.csv",
        help="Path to gold annotations CSV",
    )
    parser.add_argument(
        "--round-data",
        type=str,
        default="/home/ubuntu/repo_battleship/battleship/experiments/collaborative/battleship-final-data/round.csv",
        help="Path to round data CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cache",
        help="Directory to save results",
    )

    # Captain configuration
    parser.add_argument(
        "--captains",
        nargs="+",
        default=["EIGCaptain_cot"],
        choices=[
            "RandomCaptain",
            "MAPCaptain",
            "ProbabilisticCaptain",
            "EIGCaptain",
            "MAPEIGCaptain",
            "LLMDecisionCaptain",
            "LLMDecisionCaptain_cot",
            "ProbabilisticCaptain_cot",
            "EIGCaptain_cot",
            "MAPEIGCaptain_cot",
        ],
        help="Captain types to benchmark",
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--use-cache", action="store_true", help="Cache results")

    # Experiment settings
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[111, 222, 333],
        help="Random seeds to use",
    )
    parser.add_argument(
        "--board-ids",
        nargs="+",
        default=None,
        help="Board IDs to use (defaults to B01-B18)",
    )
    parser.add_argument(
        "--max-questions", type=int, default=15, help="Maximum questions per game"
    )
    parser.add_argument(
        "--max-moves", type=int, default=40, help="Maximum moves per game"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes to use (defaults to CPU count)",
    )
    parser.add_argument(
        "--include-human", action="store_true", help="Include human results in output"
    )

    # Captain-specific parameters
    parser.add_argument(
        "--map-samples", type=int, default=100, help="Number of samples for MAPCaptain"
    )
    parser.add_argument(
        "--prob-q-prob",
        type=float,
        default=0.7,
        help="Question probability for ProbabilisticCaptain",
    )
    parser.add_argument(
        "--eig-samples",
        type=int,
        default=1000,
        help="Number of samples for EIGCaptain",
    )
    parser.add_argument("--eig-k", type=int, default=10, help="K value for EIGCaptain")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Setup board IDs if not specified
    if args.board_ids is None:
        args.board_ids = [f"B{str(i).zfill(2)}" for i in range(1, 19)]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "individual_results"), exist_ok=True)

    # Get human results
    human_results = []
    if args.include_human:
        human_results = get_human_results(args.gold_annotations, args.round_data)
        print(f"Processed human results for {len(human_results)} games")

    # Prepare a list of tasks (each tuple corresponds to one game run)
    jobs = []
    for seed in args.seeds:
        for board_id in args.board_ids:
            for captain_type in args.captains:
                captain = create_captain(
                    captain_type=captain_type,
                    seed=seed,
                    model=args.model,
                    use_cache=args.use_cache,
                    map_samples=args.map_samples,
                    prob_q_prob=args.prob_q_prob,
                    eig_samples=args.eig_samples,
                    eig_k=args.eig_k,
                )
                jobs.append(
                    (
                        hash(captain_type + str(seed) + board_id + str(args)),
                        captain_type,
                        captain,
                        seed,
                        board_id,
                        args.max_questions,
                        args.max_moves,
                        args.model,
                    )
                )

    print(f"Running {len(jobs)} benchmark games...")

    results = []
    # Run with multiprocessing
    with Pool(processes=args.processes) as pool:
        proc_count = pool._processes
        print(f"Running with {proc_count} processes")
        results = pool.map(run_single_agent_game, jobs)

    total_results = human_results + results
    temp_filename = f"summaries_{time()}.json"
    temp_path = os.path.join(CACHE_DIR, temp_filename)
    with open(temp_path, "w") as f:
        json.dump(total_results, f, indent=2)

    print(f"Completed {len(results)} agent games out of {len(jobs)} jobs")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
