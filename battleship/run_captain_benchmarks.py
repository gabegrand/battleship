import argparse
import glob
import json
import os
import sys
import time
import uuid
from multiprocessing.dummy import Pool
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from battleship.agents import Counter
from battleship.board import Board
from battleship.captains import create_captain
from battleship.game import BattleshipGame
from battleship.spotters import CodeSpotterModel
from battleship.utils import resolve_project_path


def create_experiment_dir(root: str = None) -> Tuple:
    """
    Create experiment directories under the given cache root and return paths.
    Returns (cache_dir, results_dir, round_results_dir, experimental_results_dir,
             stage_dir, prompts_dir, human_summary_dir)
    """
    if root is None:
        root = resolve_project_path("experiments/collaborative/captain_benchmarks")
    root = Path(root)

    root.mkdir(exist_ok=True)
    experimental_results_dir = root / f"run_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    experimental_results_dir.mkdir(exist_ok=True)

    individual_results_dir = experimental_results_dir / "individual_results"
    individual_results_dir.mkdir(exist_ok=True)
    round_results_dir = experimental_results_dir / "round_results"
    round_results_dir.mkdir(exist_ok=True)

    stage_dir = individual_results_dir / "stages"
    prompts_dir = individual_results_dir / "prompts"
    stage_dir.mkdir(exist_ok=True)
    prompts_dir.mkdir(exist_ok=True)
    human_summary_dir = individual_results_dir / "human_summaries"
    human_summary_dir.mkdir(exist_ok=True)
    return (
        root,
        individual_results_dir,
        round_results_dir,
        experimental_results_dir,
        stage_dir,
        prompts_dir,
        human_summary_dir,
    )


## Experiment directory setup
HUMAN_MAX_QUESTIONS = 15

COMMAND_FILE_NAME = "command.txt"


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

    return data


def run_single_agent_game(args):
    (
        round_id,
        captain_type,
        seed,
        board_id,
        max_questions,
        max_moves,
        model,
        EXPERIMENTAL_RESULTS_DIR,
        ROUND_RESULTS_DIR,
        STAGE_DIR,
        PROMPTS_DIR,
        use_cache,
        map_samples,
        prob_q_prob,
        eig_samples,
        eig_k,
    ) = args

    print(f"{captain_type} started with {board_id} & seed {seed}")
    board = Board.from_trial_id(board_id)

    captain = create_captain(
        captain_type=captain_type,
        seed=seed,
        model=model,
        board_id=board_id,
        use_cache=use_cache,
        map_samples=map_samples,
        prob_q_prob=prob_q_prob,
        eig_samples=eig_samples,
        eig_k=eig_k,
        round_id=round_id,
        stage_dir=STAGE_DIR,
        prompts_dir=PROMPTS_DIR,
    )

    decision_counter = Counter()
    index_counter = Counter()

    # Set runtime counters
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
        stage_dir=STAGE_DIR,
        prompts_dir=PROMPTS_DIR,
    )

    # Write round information to JSON file
    round_info = {
        "id": round_id,
        "boardId": board_id,
        "seed": seed,
        "captainModel": captain_type,
        "spotterModel": spotter.__class__.__name__,
    }

    # Setup game save directory for history
    game_save_dir = EXPERIMENTAL_RESULTS_DIR / f"game_{round_id}"

    # Initialize game with save_dir
    game = BattleshipGame(
        board_target=board,
        captain=captain,
        spotter=spotter,
        max_questions=max_questions,
        max_moves=max_moves,
        save_dir=str(game_save_dir),
    )
    game.play()
    game.save()
    print(f"{captain_type} finished with {board_id} & seed {seed}")

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
    }

    # Create a unique filename for each result to avoid race conditions
    safe_model_name = model.replace("/", "-")
    temp_filename = f"{safe_model_name}_{captain_type}_{round_id}"

    results = {}
    for result in ["prompt", "stage"]:
        home_dir = PROMPTS_DIR if result == "prompt" else STAGE_DIR
        result_path = os.path.join(ROUND_RESULTS_DIR, temp_filename + f"_{result}.json")
        source_files = glob.glob(os.path.join(home_dir, f"{result}_{round_id}*.json"))
        results[result] = []
        for f in source_files:
            with open(f, "r") as f:
                results[result].append(json.load(f))
        with open(result_path, "w") as f:
            json.dump(results[result], f, indent=2)

    return summary, results["stage"], results["prompt"], round_info


def main():
    args = parse_arguments()

    (
        CACHE_DIR,
        RESULTS_DIR,
        ROUND_RESULTS_DIR,
        EXPERIMENTAL_RESULTS_DIR,
        STAGE_DIR,
        PROMPTS_DIR,
        HUMAN_SUMMARY_DIR,
    ) = create_experiment_dir()

    # Save the command used to run the script
    command = " ".join(["python"] + sys.argv)
    command_path = os.path.join(EXPERIMENTAL_RESULTS_DIR, COMMAND_FILE_NAME)
    with open(command_path, "w") as f:
        f.write(command)
    print(f"Command saved to {command_path}")

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
    for seed in args.seeds:
        for board_id in args.board_ids:
            for captain_type in args.captains:
                round_id = uuid.uuid4().hex[
                    :8
                ]  # Generate a short unique ID for the round
                jobs.append(
                    (
                        round_id,
                        captain_type,
                        seed,
                        board_id,
                        args.max_questions,
                        args.max_moves,
                        args.model,
                        EXPERIMENTAL_RESULTS_DIR,
                        ROUND_RESULTS_DIR,
                        STAGE_DIR,
                        PROMPTS_DIR,
                        args.use_cache,
                        args.map_samples,
                        args.prob_q_prob,
                        args.eig_samples,
                        args.eig_k,
                    )
                )

    print(f"Running {len(jobs)} benchmark games...")

    results = []
    # Run with multiprocessing
    with Pool(processes=args.processes) as pool:
        proc_count = pool._processes
        print(f"Running with {proc_count} processes")
        results = pool.map(run_single_agent_game, jobs)

    # Prepare data structures
    summaries_data = human_results.copy() if human_results else []
    stages_data = []
    prompts_data = []
    rounds_data = []

    for summary, stage, prompt, round_info in results:
        summaries_data.append(summary)
        stages_data.append(stage)
        prompts_data.append(prompt)
        rounds_data.append(round_info)

    # Write all files
    file_pairs = [
        ("summary.json", summaries_data),
        ("stages.json", stages_data),
        ("prompts.json", prompts_data),
        ("rounds.json", rounds_data),
    ]

    for filename, data in file_pairs:
        filepath = os.path.join(EXPERIMENTAL_RESULTS_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Completed {len(results)} agent games out of {len(jobs)} jobs")
    print(f"Results saved to {EXPERIMENTAL_RESULTS_DIR}")


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
