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
        "--seeds", nargs="+", type=int, default=[1, 2], help="Random seeds to use"
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


def game_completed(hits, misses, occTiles, board_id):
    def mask(board_array):
        return (board_array != -1) & (board_array != 0)

    if hits + misses > 40:
        return False
    else:
        return np.all(
            mask(occTiles)
            == mask(
                Board.convert_to_numeric(
                    Board.from_trial_id(board_id).to_symbolic_array()
                )
            )
        )


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
            "goldAnswer",
        ]
    ]
    df = filtered_stage_df.merge(
        board_ids, left_on="roundID", right_on="id", how="inner"
    )

    question_counts_df = (
        df[df["messageType"] == "question"]
        .groupby("roundID")
        .size()
        .reset_index(name="question_number")
    )

    df = df.merge(question_counts_df, on="roundID", how="inner")
    result = df.loc[df.groupby("roundID")["index"].idxmax()][
        ["roundID", "occTiles", "board_id", "question_number"]
    ]
    result = result[
        result["occTiles"] != str(np.full((8, 8), -1).tolist()).replace(" ", "")
    ]

    data = []
    for roundID, occTiles, board_id in zip(
        result["roundID"], result["occTiles"], result["board_id"]
    ):
        occTiles = np.array(eval(occTiles))
        misses = np.sum(occTiles == 0)
        hits = np.sum((occTiles != -1) & (occTiles != 0))

        precision = hits / (hits + misses) if (hits + misses) > 0 else 0

        data.append(
            {
                "captainType": "human",
                "boardId": board_id,
                "hits": hits,
                "misses": misses,
                "gameCompleted": game_completed(hits, misses, occTiles, board_id),
                "questionsAsked": result[result["roundID"] == roundID][
                    "question_number"
                ].values[0],
                "precision": precision,
            }
        )

    return pd.DataFrame(data)


def create_captains(args):
    # Initialize spotter for EIG captains
    eig_spotter = CodeSpotterModel(
        board_id="B01",
        board_experiment="collaborative",
        model_string=args.model,
        temperature=None,
        use_cot=True,
    )

    captains = {}

    for captain_type in args.captains:
        if captain_type == "RandomCaptain":
            captains[captain_type] = Captain(
                decision_strategy=AlwaysMoveDecisionStrategy(),
                move_strategy=RandomMoveStrategy(
                    rng=np.random.default_rng(args.seeds[0])
                ),
                question_strategy=None,
                seed=args.seeds[0],
                use_cache=args.use_cache,
                round_id=None,
            )

        elif captain_type == "MAPCaptain":
            captains[captain_type] = Captain(
                decision_strategy=AlwaysMoveDecisionStrategy(),
                move_strategy=MAPMoveStrategy(
                    rng=np.random.default_rng(args.seeds[0]), n_samples=args.map_samples
                ),
                question_strategy=None,
                seed=args.seeds[0],
                use_cache=args.use_cache,
                round_id=None,
            )

        elif captain_type == "ProbabilisticCaptain":
            captains[captain_type] = Captain(
                decision_strategy=ProbabilisticDecisionStrategy(
                    q_prob=args.prob_q_prob
                ),
                move_strategy=LLMMoveStrategy(model_string=args.model, use_cot=False),
                question_strategy=LLMQuestionStrategy(
                    model_string=args.model, use_cot=False
                ),
                seed=args.seeds[0],
                model_string=args.model,
                use_cache=args.use_cache,
                round_id=None,
            )

        elif captain_type == "ProbabilisticCaptain_cot":
            captains[captain_type] = Captain(
                decision_strategy=ProbabilisticDecisionStrategy(
                    q_prob=args.prob_q_prob
                ),
                move_strategy=LLMMoveStrategy(model_string=args.model, use_cot=True),
                question_strategy=LLMQuestionStrategy(
                    model_string=args.model, use_cot=True
                ),
                seed=args.seeds[0],
                model_string=args.model,
                use_cache=args.use_cache,
                round_id=None,
            )

        elif captain_type == "LLMDecisionCaptain":
            captains[captain_type] = Captain(
                decision_strategy=LLMDecisionStrategy(
                    model_string=args.model, use_cot=False
                ),
                move_strategy=LLMMoveStrategy(model_string=args.model, use_cot=False),
                question_strategy=LLMQuestionStrategy(
                    model_string=args.model, use_cot=False
                ),
                seed=args.seeds[0],
                model_string=args.model,
                use_cache=args.use_cache,
                round_id=None,
            )

        elif captain_type == "LLMDecisionCaptain_cot":
            captains[captain_type] = Captain(
                decision_strategy=LLMDecisionStrategy(
                    model_string=args.model, use_cot=True
                ),
                move_strategy=LLMMoveStrategy(model_string=args.model, use_cot=True),
                question_strategy=LLMQuestionStrategy(
                    model_string=args.model, use_cot=True
                ),
                seed=args.seeds[0],
                model_string=args.model,
                use_cache=args.use_cache,
                round_id=None,
            )

        elif captain_type == "EIGCaptain":
            captains[captain_type] = Captain(
                decision_strategy=LLMDecisionStrategy(
                    model_string=args.model, use_cot=False
                ),
                move_strategy=LLMMoveStrategy(model_string=args.model, use_cot=False),
                question_strategy=EIGQuestionStrategy(
                    model_string=args.model,
                    spotter=eig_spotter,
                    rng=np.random.default_rng(args.seeds[0]),
                    samples=args.eig_samples,
                    k=args.eig_k,
                    use_cot=False,
                ),
                seed=args.seeds[0],
                model_string=args.model,
                use_cache=args.use_cache,
                round_id=None,
            )

        elif captain_type == "EIGCaptain_cot":
            captains[captain_type] = Captain(
                decision_strategy=LLMDecisionStrategy(
                    model_string=args.model, use_cot=True
                ),
                move_strategy=LLMMoveStrategy(model_string=args.model, use_cot=True),
                question_strategy=EIGQuestionStrategy(
                    model_string=args.model,
                    spotter=eig_spotter,
                    rng=np.random.default_rng(args.seeds[0]),
                    samples=args.eig_samples,
                    k=args.eig_k,
                    use_cot=True,
                ),
                seed=args.seeds[0],
                model_string=args.model,
                use_cache=args.use_cache,
                round_id=None,
            )

        elif captain_type == "MAPEIGCaptain":
            captains[captain_type] = Captain(
                decision_strategy=LLMDecisionStrategy(
                    model_string=args.model, use_cot=False
                ),
                move_strategy=MAPMoveStrategy(
                    rng=np.random.default_rng(args.seeds[0]), n_samples=args.eig_samples
                ),
                question_strategy=EIGQuestionStrategy(
                    model_string=args.model,
                    spotter=eig_spotter,
                    rng=np.random.default_rng(args.seeds[0]),
                    samples=args.eig_samples,
                    k=args.eig_k,
                    use_cot=False,
                ),
                seed=args.seeds[0],
                model_string=args.model,
                use_cache=args.use_cache,
                round_id=None,
            )

        elif captain_type == "MAPEIGCaptain_cot":
            captains[captain_type] = Captain(
                decision_strategy=LLMDecisionStrategy(
                    model_string=args.model, use_cot=True
                ),
                move_strategy=MAPMoveStrategy(
                    rng=np.random.default_rng(args.seeds[0]), n_samples=args.eig_samples
                ),
                question_strategy=EIGQuestionStrategy(
                    model_string=args.model,
                    spotter=eig_spotter,
                    rng=np.random.default_rng(args.seeds[0]),
                    samples=args.eig_samples,
                    k=args.eig_k,
                    use_cot=True,
                ),
                seed=args.seeds[0],
                model_string=args.model,
                use_cache=args.use_cache,
                round_id=None,
            )

    return captains


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

    file_exists = os.path.isfile(CSV_ROUND_FILE)
    with open(CSV_ROUND_FILE, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["id", "boardId", "seed", "captainModel", "spotterModel"],
        )
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

    result = {
        "roundId": round_id,
        "captainType": cap_name,
        "boardId": board_id,
        "hits": game.hits,
        "misses": game.misses,
        "gameCompleted": game.is_won(),
        "questionsAsked": game.question_count,
        "precision": game.hits / (game.hits + game.misses)
        if (game.hits + game.misses) > 0
        else 0,
    }

    pd.DataFrame.from_dict([result]).to_csv(
        SUMMARY_FILE,
        mode="a",
        header=False,
        index=False,
    )

    return result


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
        human_results_df.to_csv(args.output_file, index=False)
        print(f"Wrote human results to {args.output_file}")
    else:
        # Initialize empty results file
        pd.DataFrame(
            columns=[
                "captainType",
                "boardId",
                "hits",
                "misses",
                "gameCompleted",
                "questionsAsked",
                "precision",
            ]
        ).to_csv(args.output_file, index=False)

    # Create captains using the modular architecture
    captains = create_captains(args)

    # Prepare a list of tasks (each tuple corresponds to one game run)
    jobs = [
        (
            randint(0, 1000000),
            deepcopy(cap_name),
            deepcopy(captain),
            seed,
            board_id,
            args.max_questions,
            args.max_moves,
            args.model,
        )
        for cap_name, captain in captains.items()
        for seed in args.seeds
        for board_id in args.board_ids
    ]

    print(f"Running {len(jobs)} benchmark games...")

    results = []
    # Run with multiprocessing
    with Pool(processes=args.processes) as pool:
        proc_count = pool._processes
        print(f"Running with {proc_count} processes")
        results = pool.map(run_single_agent_game, jobs)

    # Write all results to CSV
    results_df = pd.DataFrame(results)

    # Append to existing file
    if human_results_df is not None:
        results_df.to_csv(args.output_file, mode="a", header=False, index=False)
    else:
        # Write without human results
        results_df.to_csv(args.output_file, index=False)

    print(f"Completed {len(results)} agent games out of {len(jobs)} jobs")
    print(f"Results saved to {args.output_file}")

    # Print summary statistics
    print("\nSummary by Captain Type:")
    summary = (
        results_df.groupby("captainType")
        .agg({"precision": "mean", "gameCompleted": "mean", "questionsAsked": "mean"})
        .reset_index()
    )
    print(summary)


if __name__ == "__main__":
    main()
