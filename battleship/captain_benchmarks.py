#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from copy import deepcopy
from multiprocessing.dummy import Pool
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd

from battleship.agents import Counter
from battleship.agents import CSV_ROUND_FILE
from battleship.agents import SUMMARY_FILE
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

# Define consistent CSV column names
ROUND_CSV_COLUMNS = ["id", "boardId", "seed", "captainModel", "spotterModel"]
RESULTS_CSV_COLUMNS = [
    "roundId",
    "captainType",
    "boardId",
    "hits",
    "misses",
    "is_won",
    "question_count",
    "precision",
    "recall",
    "f1_score",
]


def get_human_results(gold_annotations_path, round_data_path):
    stage_df = pd.read_csv(gold_annotations_path)
    round_df = pd.read_csv(round_data_path)

    board_ids = round_df[["id", "board_id"]]
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
        df[df["messageType"] == "question"]
        .groupby("roundID")
        .size()
        .reset_index(name="question_number")
    )

    df = df.merge(question_counts_df, on="roundID", how="left")
    result = df.loc[df.groupby("roundID")["index"].idxmax()][
        ["roundID", "occTiles", "board_id", "question_number"]
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

        data.append(
            {
                "roundId": roundID,
                "captainType": "human",
                "boardId": board_id,
                "hits": scores["hits"],
                "misses": scores["misses"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1_score": scores["f1_score"],
                "is_won": scores["is_won"],
                "question_count": result[result["roundID"] == roundID][
                    "question_number"
                ].values[0],
            }
        )

    return pd.DataFrame(data)


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

    # Write round information using consistent column names
    file_exists = os.path.isfile(CSV_ROUND_FILE)
    with open(CSV_ROUND_FILE, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ROUND_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "id": round_id,
                "boardId": board_id,
                "seed": seed,
                "captainModel": cap_name,
                "spotterModel": spotter.__class__.__name__,
            }
        )

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
        "hits": game.hits,
        "misses": game.misses,
        "is_won": game.is_won(),
        "question_count": game.question_count,
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1_score": scores["f1_score"],
    }

    # Write summary information with consistent column order
    summary_df = pd.DataFrame.from_dict([result])
    summary_df.to_csv(
        SUMMARY_FILE,
        mode="a",
        header=not os.path.exists(SUMMARY_FILE),
        index=False,
    )

    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Battleship Captain benchmarks")

    # Data paths
    parser.add_argument(
        "--gold-annotations",
        type=str,
        default="/home/ubuntu/repo_battleship/temp/gold_annotations_partial.csv",
        help="Path to gold annotations CSV",
    )
    parser.add_argument(
        "--round-data",
        type=str,
        default="/home/ubuntu/repo_battleship/battleship/experiments/collaborative/battleship-final-data/round.csv",
        help="Path to round data CSV",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/ubuntu/repo_battleship/temp/total_results.csv",
        help="Path to output results CSV",
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
    output_path = Path(args.output_file)
    os.makedirs(output_path.parent, exist_ok=True)

    # Get human results
    human_results_df = None
    if args.include_human:
        human_results_df = get_human_results(args.gold_annotations, args.round_data)
        # Ensure column names match our defined schema
        human_results_df = human_results_df[RESULTS_CSV_COLUMNS]
        human_results_df.to_csv(args.output_file, index=False)
        print(f"Wrote human results to {args.output_file}")
    else:
        # Initialize empty results file with consistent column names
        pd.DataFrame(columns=RESULTS_CSV_COLUMNS).to_csv(args.output_file, index=False)

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

    # Write all results to CSV
    results_df = pd.DataFrame(results)

    # Ensure all columns match our defined schema
    for col in RESULTS_CSV_COLUMNS:
        if col not in results_df.columns:
            results_df[col] = None
    results_df = results_df[RESULTS_CSV_COLUMNS]

    # Append to existing file
    if human_results_df is not None:
        results_df.to_csv(args.output_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(args.output_file, index=False)

    print(f"Completed {len(results)} agent games out of {len(jobs)} jobs")
    print(f"Results saved to {args.output_file}")

    # Print summary statistics
    print("\nSummary by Captain Type:")
    summary = (
        results_df.groupby("captainType")
        .agg(
            {
                "precision": "mean",
                "recall": "mean",
                "f1_score": "mean",
                "question_count": "mean",
            }
        )
        .reset_index()
    )
    print(summary)


if __name__ == "__main__":
    main()
