"""
# Start new captain experiment
python run_captain_benchmarks.py --captains EIGCaptain_cot --captain-llm gpt-4o-mini --spotter-llm gpt-4o-mini

# Resume interrupted captain experiment
python run_captain_benchmarks.py --resume --experiment-dir {EXPERIMENT_DIR}

# Force restart captain experiment (clear existing results)
python run_captain_benchmarks.py --force-restart --experiment-dir {EXPERIMENT_DIR}

# Resume with additional configurations
python run_captain_benchmarks.py --resume --experiment-dir {EXPERIMENT_DIR} --captains EIGCaptain_cot RandomCaptain --seeds 42 123
"""
import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from battleship.board import Board
from battleship.captains import create_captain
from battleship.game import BattleshipGame
from battleship.spotters import create_spotter
from battleship.utils import resolve_project_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("captain_benchmark.log"),
        logging.StreamHandler(),
    ],
)

# Suppress HTTP request logs from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


@dataclass
class CaptainBenchmarkConfig:
    """Configuration for captain benchmark experiments."""

    captain_type: str
    seed: int
    captain_llm: str
    spotter_llm: str
    max_questions: int
    max_moves: int
    experiment_dir: str
    map_samples: int = 1000
    prob_q_prob: float = 0.7
    eig_samples: int = 1000
    eig_k: int = 10
    experiment_name: str = None

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.captain_type}_seed{self.seed}"


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


# ---------------------
# Resilience Functions
# ---------------------


def save_game_checkpoint(result: Dict, config: CaptainBenchmarkConfig) -> None:
    """Save a lightweight checkpoint after each game completion."""
    checkpoint_dir = os.path.join(
        config.experiment_dir, "checkpoints", config.experiment_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    completed_file = os.path.join(checkpoint_dir, "completed_games.json")

    # Load existing completed games
    try:
        with open(completed_file, "r") as f:
            completed = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        completed = []

    # Add new completion
    completed.append(
        {
            "captain_type": result["captainType"],
            "seed": result["seed"],
            "board_id": result["boardId"],
            "timestamp": time.time(),
            "is_won": result["is_won"],
            "round_id": result["roundId"],
        }
    )

    # Save updated list
    with open(completed_file, "w") as f:
        json.dump(completed, f, indent=2)


def save_game_error_checkpoint(
    captain_type: str,
    seed: int,
    board_id: str,
    config: CaptainBenchmarkConfig,
    error: Exception,
) -> None:
    """Save details of failed games."""
    checkpoint_dir = os.path.join(
        config.experiment_dir, "checkpoints", config.experiment_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    error_file = os.path.join(checkpoint_dir, "errors.json")

    try:
        with open(error_file, "r") as f:
            errors = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        errors = []

    errors.append(
        {
            "captain_type": captain_type,
            "seed": seed,
            "board_id": board_id,
            "error": str(error),
            "timestamp": time.time(),
        }
    )

    with open(error_file, "w") as f:
        json.dump(errors, f, indent=2)


def get_completed_games(config: CaptainBenchmarkConfig) -> Set[Tuple[str, int, str]]:
    """Get set of already completed (captain_type, seed, board_id) tuples."""
    checkpoint_file = os.path.join(
        config.experiment_dir,
        "checkpoints",
        config.experiment_name,
        "completed_games.json",
    )

    try:
        with open(checkpoint_file, "r") as f:
            completed = json.load(f)
        return {
            (item["captain_type"], item["seed"], item["board_id"]) for item in completed
        }
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_captain_configuration_results(
    results: List[Dict], config: CaptainBenchmarkConfig
) -> None:
    """Save complete results for a captain configuration."""
    results_dir = os.path.join(config.experiment_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"{config.experiment_name}.json")

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Saved {len(results)} results for {config.experiment_name}")


def load_existing_captain_results(config: CaptainBenchmarkConfig) -> List[Dict]:
    """Load existing results for a captain configuration if they exist."""
    results_file = os.path.join(
        config.experiment_dir, "results", f"{config.experiment_name}.json"
    )

    try:
        with open(results_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _extract_game_id_from_job(job: Tuple) -> Tuple[str, int, str]:
    """Extract (captain_type, seed, board_id) from a job tuple."""
    return job[1], job[2], job[3]  # captain_type, seed, board_id


def is_captain_configuration_complete(
    config: CaptainBenchmarkConfig, jobs_for_config: List[Tuple]
) -> bool:
    """Check if a captain configuration has been fully completed for the specific jobs requested."""
    completed_games = get_completed_games(config)

    # Check if all specific games in jobs_for_config have been completed
    return all(
        _extract_game_id_from_job(job) in completed_games for job in jobs_for_config
    )


def get_remaining_captain_work(
    all_jobs: List[Tuple], config: CaptainBenchmarkConfig
) -> List[Tuple]:
    """Filter out already completed games."""
    completed_games = get_completed_games(config)

    return [
        job for job in all_jobs if _extract_game_id_from_job(job) not in completed_games
    ]


def rebuild_captain_summary_from_results(experiment_dir: str) -> List[Dict]:
    """Rebuild summary.json from individual captain result files."""
    results_dir = os.path.join(experiment_dir, "results")

    if not os.path.exists(results_dir):
        return []

    all_results = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r") as f:
                    results = json.load(f)
                all_results.extend(results)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.warning(f"Could not load results from {filepath}: {e}")

    return all_results


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


# ---------------------
# Resilient Core Functions
# ---------------------


def run_single_agent_game_wrapper(args) -> Optional[Dict]:
    """Run a single agent game with error handling and checkpointing."""

    # Extract parameters from args tuple
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

    # Create config for this specific captain+seed combination
    config = CaptainBenchmarkConfig(
        captain_type=captain_type,
        seed=seed,
        captain_llm=captain_llm,
        spotter_llm=spotter_llm,
        max_questions=max_questions,
        max_moves=max_moves,
        experiment_dir=experiment_dir,
        map_samples=map_samples,
        prob_q_prob=prob_q_prob,
        eig_samples=eig_samples,
        eig_k=eig_k,
    )

    try:
        # Check if already completed
        completed = get_completed_games(config)
        if (captain_type, seed, board_id) in completed:
            logging.debug(
                f"Skipping already completed: {captain_type} seed{seed} {board_id}"
            )
            return None

        # Original game processing logic
        result = run_single_agent_game(args)

        # Save checkpoint on success
        save_game_checkpoint(result, config)

        return result

    except Exception as e:
        logging.error(
            f"Failed to process game: {captain_type} seed{seed} {board_id}: {e}"
        )
        save_game_error_checkpoint(captain_type, seed, board_id, config, e)
        return None


def run_captain_configuration(
    jobs_for_config: List[Tuple],
    config: CaptainBenchmarkConfig,
    max_workers: int = None,
) -> List[Dict]:
    """Run all games for a single captain configuration with resume capability."""

    # Check if configuration is already complete for the specific jobs requested
    if is_captain_configuration_complete(config, jobs_for_config):
        logging.info(
            f"Configuration {config.experiment_name} already complete for requested jobs, loading existing results"
        )
        return load_existing_captain_results(config)

    logging.info(f"Running/resuming benchmark: {config.experiment_name}")

    # Filter out completed work
    remaining_jobs = get_remaining_captain_work(jobs_for_config, config)

    if not remaining_jobs:
        logging.info(f"All work already completed for {config.experiment_name}")
        return load_existing_captain_results(config)

    logging.info(
        f"Processing {len(remaining_jobs)} remaining games out of {len(jobs_for_config)} total"
    )

    # Process remaining games
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        logging.info(f"Running with max {executor._max_workers} worker threads...")
        new_results = list(
            tqdm(
                executor.map(run_single_agent_game_wrapper, remaining_jobs),
                total=len(remaining_jobs),
                desc=f"Processing {config.captain_type} seed{config.seed}",
            )
        )

    # Filter out None results (failures/skips)
    new_results = [r for r in new_results if r is not None]

    # Combine with existing results
    existing_results = load_existing_captain_results(config)
    all_results = existing_results + new_results

    # Save complete configuration results
    save_captain_configuration_results(all_results, config)

    logging.info(f"Completed {len(new_results)} new games for {config.experiment_name}")
    logging.info(f"Total results for configuration: {len(all_results)}")

    return all_results


def run_all_captain_experiments(
    captains: List[str],
    seeds: List[int],
    board_ids: List[str],
    captain_llm: str,
    spotter_llm: str,
    max_questions: int,
    max_moves: int,
    experiment_dir: str,
    map_samples: int,
    prob_q_prob: float,
    eig_samples: int,
    eig_k: int,
    max_workers: int = None,
) -> List[Dict]:
    """Run experiments for all captain configurations, supporting resume from partial completion."""

    all_results = []

    # Group jobs by configuration (captain_type + seed)
    for captain_type in captains:
        for seed in seeds:
            # Create configuration
            config = CaptainBenchmarkConfig(
                captain_type=captain_type,
                seed=seed,
                captain_llm=captain_llm,
                spotter_llm=spotter_llm,
                max_questions=max_questions,
                max_moves=max_moves,
                experiment_dir=experiment_dir,
                map_samples=map_samples,
                prob_q_prob=prob_q_prob,
                eig_samples=eig_samples,
                eig_k=eig_k,
            )

            # Create jobs for this configuration
            jobs_for_config = []
            for board_id in board_ids:
                round_id = uuid.uuid4().hex[:8]
                jobs_for_config.append(
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
                    )
                )

            try:
                results = run_captain_configuration(
                    jobs_for_config, config, max_workers
                )
                all_results.extend(results)
            except Exception as e:
                logging.error(
                    f"Configuration {config.experiment_name} failed completely: {e}"
                )
                # Continue with other configurations rather than failing entirely
                continue

    return all_results


def main():
    """Main entry point for resilient captain benchmarks."""
    start_time = time.time()
    args = parse_arguments()

    # Handle experiment directory creation/resumption
    if args.resume and args.experiment_dir:
        experiment_dir = resolve_project_path(args.experiment_dir)
        if not os.path.exists(experiment_dir):
            raise ValueError(
                f"Cannot resume: experiment directory {experiment_dir} does not exist"
            )
        logging.info(f"Resuming experiment in {experiment_dir}")
    elif args.experiment_dir:
        experiment_dir = create_experiment_dir(
            resolve_project_path(args.experiment_dir)
        )
    else:
        experiment_dir = create_experiment_dir()

    # Clear existing results if force restart
    if args.force_restart:
        checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        results_dir = os.path.join(experiment_dir, "results")
        if os.path.exists(checkpoints_dir):
            shutil.rmtree(checkpoints_dir)
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        logging.info("Cleared existing results due to --force-restart")

    # Save the command used to run the script (only if not resuming or file doesn't exist)
    command_path = os.path.join(experiment_dir, "command.txt")
    if not args.resume or not os.path.exists(command_path):
        command = " ".join(["python"] + sys.argv)
        with open(command_path, "w") as f:
            f.write(command)
        print(f"Command saved to {command_path}")

    # Create rounds directory
    rounds_dir = os.path.join(experiment_dir, "rounds")
    os.makedirs(rounds_dir, exist_ok=True)

    action = "Resuming" if args.resume else "Starting"
    logging.info(f"{action} captain benchmark experiment in {experiment_dir}")

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

    # Determine which models to use
    if args.captain_llm is None or args.spotter_llm is None:
        raise ValueError("You must provide both --captain-llm and --spotter-llm flags.")

    # Run experiments with resilience
    results = run_all_captain_experiments(
        captains=args.captains,
        seeds=args.seeds,
        board_ids=args.board_ids,
        captain_llm=args.captain_llm,
        spotter_llm=args.spotter_llm,
        max_questions=args.max_questions,
        max_moves=args.max_moves,
        experiment_dir=experiment_dir,
        map_samples=args.map_samples,
        prob_q_prob=args.prob_q_prob,
        eig_samples=args.eig_samples,
        eig_k=args.eig_k,
        max_workers=args.max_workers,
    )

    # Rebuild and save final summary from all results
    final_results = rebuild_captain_summary_from_results(experiment_dir)
    if final_results:
        results = final_results  # Use rebuilt results if available

    # Prepare data structures
    summaries_data = human_results.copy() if human_results else []
    summaries_data.extend(results)

    # Write summary file
    summary_path = os.path.join(experiment_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries_data, f, indent=2)

    # Create/update metadata file for experiment info
    metadata = {
        "experiment_dir": str(experiment_dir),
        "total_results": len(results),
        "experiment_args": vars(args),
        "experiments_run": len(args.captains) * len(args.seeds),
        "resumed": args.resume,
        "force_restarted": args.force_restart,
    }

    metadata_path = os.path.join(experiment_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Experiment completed!")
    logging.info(f"Results saved to: {experiment_dir}")
    logging.info(f"- summary.json: {len(results)} game results")
    logging.info(f"- metadata.json: experiment configuration")
    logging.info(f"- results/: individual configuration result files")
    logging.info(f"- checkpoints/: incremental progress checkpoints")

    # Log overall runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    logging.info(
        f"Overall runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)"
    )

    print(f"Completed {len(results)} agent games")
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

    # Experiment configuration
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Full path to experiment directory. If not provided, will create a timestamped directory in experiments/collaborative/captain_benchmarks.",
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

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing experiment directory, skipping completed work",
    )

    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing results and restart experiment from scratch",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
