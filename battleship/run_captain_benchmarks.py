import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm

from battleship.board import Board
from battleship.captains import create_captain
from battleship.game import BattleshipGame
from battleship.spotters import create_spotter
from battleship.utils import resolve_project_path


def create_experiment_dir(root: str = None) -> str:
    """
    Create experiment directory under the given root and return path.
    Returns the path to the experimental results directory.
    """
    if root is None:
        root = resolve_project_path("experiments/collaborative/captain_benchmarks")

    experiment_dir = os.path.join(root, f"run_{time.strftime('%Y_%m_%d_%H_%M_%S')}")
    os.makedirs(experiment_dir)

    return experiment_dir


def get_human_results(gold_annotations_path, round_data_path, max_questions=15):
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

        questions_asked = max_questions - int(
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

    return data


def run_single_agent_game(args):
    (
        round_id,
        captain_type,
        seed,
        board_id,
        max_questions,
        max_moves,
        captain_llm,
        spotter_llm,
        experiment_dir,
        map_samples,
        prob_q_prob,
        eig_samples,
        eig_k,
    ) = args

    print(f"{captain_type} {round_id} started with {board_id} & seed {seed}")
    board = Board.from_trial_id(board_id)

    # Create round directory and subdirectories
    round_dir = os.path.join(experiment_dir, "rounds", f"round_{round_id}")
    captain_dir = os.path.join(round_dir, "captain")
    spotter_dir = os.path.join(round_dir, "spotter")

    os.makedirs(round_dir, exist_ok=True)
    os.makedirs(captain_dir, exist_ok=True)
    os.makedirs(spotter_dir, exist_ok=True)

    captain = create_captain(
        captain_type=captain_type,
        seed=seed,
        llm=captain_llm,
        board_id=board_id,
        map_samples=map_samples,
        prob_q_prob=prob_q_prob,
        eig_samples=eig_samples,
        eig_k=eig_k,
        json_path=os.path.join(captain_dir, "captain.json"),
    )

    spotter = create_spotter(
        spotter_type="CodeSpotterModel",
        board_id=board_id,
        board_experiment="collaborative",
        llm=spotter_llm,
        temperature=None,
        use_cot=True,
        json_path=os.path.join(spotter_dir, "spotter.json"),
    )

    game = BattleshipGame(
        board_target=board,
        captain=captain,
        spotter=spotter,
        max_questions=max_questions,
        max_moves=max_moves,
        save_dir=round_dir,
    )
    game.play()
    game.save()
    print(f"{captain_type} {round_id} finished with {board_id} & seed {seed}")

    scores = game.score()

    # Ensure consistent column names in the result data
    summary = {
        "roundId": round_id,
        "captainType": captain_type,
        "boardId": board_id,
        "hits": int(game.hits),
        "misses": int(game.misses),
        "is_won": bool(game.is_won()),
        "question_count": game.question_count,
        "precision": float(scores["precision"]),
        "recall": float(scores["recall"]),
        "f1_score": float(scores["f1_score"]),
        "seed": seed,
        "spotterModel": spotter.__class__.__name__,
    }

    return summary


def main():
    args = parse_arguments()

    experiment_dir = create_experiment_dir()

    # Save the command used to run the script
    command = " ".join(["python"] + sys.argv)
    command_path = os.path.join(experiment_dir, "command.txt")
    with open(command_path, "w") as f:
        f.write(command)
    print(f"Command saved to {command_path}")

    # Create rounds directory
    rounds_dir = os.path.join(experiment_dir, "rounds")
    os.makedirs(rounds_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(experiment_dir, "run.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Resolve paths relative to project root
    gold_annotations_path = resolve_project_path(args.gold_annotations)
    round_data_path = resolve_project_path(args.round_data)

    # Setup board IDs if not specified
    if args.board_ids is None:
        args.board_ids = [f"B{str(i).zfill(2)}" for i in range(1, 19)]

    # Get human results
    human_results = []
    if args.include_human:
        human_results = get_human_results(
            str(gold_annotations_path), str(round_data_path)
        )
        print(f"Processed human results for {len(human_results)} games")

    # Prepare a list of tasks (each tuple corresponds to one game run)
    jobs = []
    # Determine which models to use
    if args.captain_llm is None or args.spotter_llm is None:
        raise ValueError("You must provide both --captain-llm and --spotter-llm flags.")
    captain_llm = args.captain_llm
    spotter_llm = args.spotter_llm
    for seed in args.seeds:
        for board_id in args.board_ids:
            for captain_type in args.captains:
                # Generate a short unique ID for the round
                round_id = uuid.uuid4().hex[:8]
                jobs.append(
                    (
                        round_id,
                        captain_type,
                        seed,
                        board_id,
                        args.max_questions,
                        args.max_moves,
                        captain_llm,
                        spotter_llm,
                        experiment_dir,
                        args.map_samples,
                        args.prob_q_prob,
                        args.eig_samples,
                        args.eig_k,
                    )
                )

    print(f"Running {len(jobs)} benchmark games...")

    # Run with ThreadPoolExecutor (concurrent.futures)
    max_workers = args.max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        logging.info(f"Running with max {executor._max_workers} worker threads...")
        # executor.map() preserves order - results will be in same order as jobs
        results = list(
            tqdm(
                executor.map(run_single_agent_game, jobs),
                total=len(jobs),
                desc="Running captain benchmark games",
            )
        )

    # Prepare data structures
    summaries_data = human_results.copy() if human_results else []
    summaries_data.extend(results)

    # Write summary file
    summary_path = os.path.join(experiment_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries_data, f, indent=2)

    print(f"Completed {len(results)} agent games out of {len(jobs)} jobs")
    print(f"Results saved to {experiment_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Battleship Captain benchmarks")

    # Data paths
    parser.add_argument(
        "--gold-annotations",
        type=str,
        default="experiments/collaborative/data/battleship-final-data/gold-v2/gold-v2.csv",
        help="Path to gold annotations CSV",
    )
    parser.add_argument(
        "--round-data",
        type=str,
        default="experiments/collaborative/data/battleship-final-data/round.csv",
        help="Path to round data CSV",
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
    parser.add_argument(
        "--captain-llm", type=str, default=None, help="LLM model to use for captain"
    )
    parser.add_argument(
        "--spotter-llm", type=str, default=None, help="LLM model to use for spotter"
    )

    # Experiment settings
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
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
        "--max-workers",
        type=int,
        default=None,
        help="Number of worker threads to use for parallelism (defaults to Python's ThreadPoolExecutor default)",
    )
    parser.add_argument(
        "--include-human", action="store_true", help="Include human results in output"
    )

    # Captain-specific parameters
    parser.add_argument(
        "--map-samples", type=int, default=1000, help="Number of samples for MAPCaptain"
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


if __name__ == "__main__":
    main()
