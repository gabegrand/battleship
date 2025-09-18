"""
# Start new captain experiment
python run_captain_benchmarks.py --captains LLMDecisionCaptain --captain-llm gpt-4o-mini --spotter-llm gpt-4o-mini

# Resume interrupted captain experiment
python run_captain_benchmarks.py --resume --experiment-dir {EXPERIMENT_DIR}

# Force restart captain experiment (clear existing results)
python run_captain_benchmarks.py --force-restart --experiment-dir {EXPERIMENT_DIR}

# Resume with additional configurations
python run_captain_benchmarks.py --resume --experiment-dir {EXPERIMENT_DIR} --captains LLMDecisionCaptain RandomCaptain --seeds 42 123
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

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

from battleship.board import Board
from battleship.captains import create_captain
from battleship.game import BattleshipGame
from battleship.spotters import create_spotter
from battleship.utils import resolve_project_path

# ------------------------------------------------------------------
# Logging configuration: Use Rich's RichHandler for coordinated console output
# that works harmoniously with Rich progress displays. Also mirror to file.
# ------------------------------------------------------------------
# Create a shared console instance for coordination between logging and progress
shared_console = Console()

# Use Rich's logging handler to coordinate with Rich progress displays
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            console=shared_console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        ),
    ],
    force=True,
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
    board_id: str
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
            self.experiment_name = (
                f"{self.captain_type}_seed{self.seed}_{self.board_id}"
            )


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


def get_completed_games(experiment_dir: str) -> Set[Tuple[str, int, str]]:
    """Get set of already completed (captain_type, seed, board_id) tuples by scanning result files."""
    completed = set()
    results_dir = os.path.join(experiment_dir, "results")

    if not os.path.exists(results_dir):
        return completed

    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r") as f:
                    results = json.load(f)
                for result in results:
                    completed.add(
                        (result["captain_type"], result["seed"], result["board_id"])
                    )
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.warning(f"Could not load results from {filepath}: {e}")

    return completed


def save_captain_configuration_results(
    results: List[Dict], config: CaptainBenchmarkConfig
) -> None:
    """Save complete results for a captain configuration."""
    results_dir = os.path.join(config.experiment_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"{config.experiment_name}.json")

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"✅ Configuration completed: {config.experiment_name}")


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


def get_remaining_games(all_jobs: List[Tuple], experiment_dir: str) -> List[Tuple]:
    """Filter out already completed games."""
    completed_games = get_completed_games(experiment_dir)

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

    scores = game.score()

    # Ensure consistent column names in the result data
    summary = {
        "captain_type": captain_type,
        "spotter_type": spotter.__class__.__name__,
        "round_id": round_id,
        "board_id": board_id,
        "seed": seed,
        "hits": int(game.hits),
        "misses": int(game.misses),
        "is_won": bool(game.is_won()),
        "question_count": game.question_count,
        "precision": float(scores["precision"]),
        "recall": float(scores["recall"]),
        "f1_score": float(scores["f1_score"]),
    }

    return summary


def run_single_agent_game_wrapper(args, retry_count=0) -> Optional[Dict]:
    """Run a single agent game with error handling."""

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

    try:
        # Run the game
        result = run_single_agent_game(args)
        return result
    except Exception as e:
        retry_msg = f" (retry {retry_count})" if retry_count > 0 else ""
        logging.error(
            f"Failed to process game: {captain_type} seed{seed} {board_id}{retry_msg}: {e}"
        )
        return None


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
    resume_mode: bool = False,
) -> List[Dict]:
    """Run experiments for all captain configurations with Rich progress display."""

    # ---------------------------------------------------------------------
    # 1. Generate all configuration objects up-front (flat list)
    # ---------------------------------------------------------------------
    all_configs: List[CaptainBenchmarkConfig] = []
    all_jobs: List[Tuple] = []

    for captain_type in captains:
        for seed in seeds:
            for board_id in board_ids:
                # Create configuration for each (captain_type, seed, board_id)
                config = CaptainBenchmarkConfig(
                    captain_type=captain_type,
                    seed=seed,
                    board_id=board_id,
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
                all_configs.append(config)

                # Create job for this configuration (one job per config now)
                round_id = uuid.uuid4().hex[:8]
                job = (
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
                all_jobs.append(job)

    logging.info(f"Preparing {len(all_configs)} total configurations")

    # Build a nice multi-row progress UI: each config gets its own row.
    progress_columns = [
        SpinnerColumn(),
        TextColumn("{task.description}", style="bold"),
        BarColumn(bar_width=None),
        TaskProgressColumn(
            text_format="[progress.completed]{task.completed}/[progress.total]{task.total} games"
        ),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    # ---------------------------------------------------------------------
    # 2. Filter out already completed games
    # ---------------------------------------------------------------------
    # Filter out already completed games
    remaining_jobs = get_remaining_games(all_jobs, experiment_dir)

    # Group configurations and jobs by captain type for progress tracking
    captain_types = list(set(captains))
    captain_type_to_idx = {
        captain_type: idx for idx, captain_type in enumerate(captain_types)
    }

    # Count total and completed games per captain type
    games_per_captain = {captain_type: 0 for captain_type in captain_types}
    completed_per_captain = {captain_type: 0 for captain_type in captain_types}

    completed_games = get_completed_games(experiment_dir)

    # Count total games per captain type
    for config in all_configs:
        games_per_captain[config.captain_type] += 1

    # Count completed games per captain type
    for captain_type, seed, board_id in completed_games:
        if captain_type in completed_per_captain:
            completed_per_captain[captain_type] += 1

    # Create mapping from job to captain type index for progress tracking
    job_to_captain_idx = {}
    for job in remaining_jobs:
        captain_type = job[1]  # Extract captain type from job tuple
        job_to_captain_idx[id(job)] = captain_type_to_idx[captain_type]

    # ---------------------------------------------------------------------
    # 3. Execute with Rich progress display
    # ---------------------------------------------------------------------
    with Progress(
        *progress_columns, transient=False, console=shared_console
    ) as progress:
        # Create one progress bar per captain type
        task_ids: List[int] = []
        for captain_type in captain_types:
            total_games = games_per_captain[captain_type]
            completed_games = completed_per_captain[captain_type]
            task_id = progress.add_task(
                captain_type, total=total_games, completed=completed_games
            )
            task_ids.append(task_id)

        def process_job_with_progress(job):
            """Process a job, save result immediately, and update progress bar."""
            result = run_single_agent_game_wrapper(job)
            if result is not None:
                captain_type, seed, board_id = job[1], job[2], job[3]

                # Find the config for this specific job and save result
                for config in all_configs:
                    if (
                        config.captain_type == captain_type
                        and config.seed == seed
                        and config.board_id == board_id
                    ):
                        save_captain_configuration_results([result], config)
                        break

                # Update progress for this captain type
                captain_idx = job_to_captain_idx[id(job)]
                progress.update(task_ids[captain_idx], advance=1)
            return result

        def process_job_with_retry(job, max_retries=10):
            """Process a job with retry logic for failed games."""
            for retry_count in range(max_retries + 1):
                result = run_single_agent_game_wrapper(job, retry_count)
                if result is not None:
                    captain_type, seed, board_id = job[1], job[2], job[3]

                    # Find the config for this specific job and save result
                    for config in all_configs:
                        if (
                            config.captain_type == captain_type
                            and config.seed == seed
                            and config.board_id == board_id
                        ):
                            save_captain_configuration_results([result], config)
                            break

                    # Update progress for this captain type
                    captain_idx = job_to_captain_idx[id(job)]
                    progress.update(task_ids[captain_idx], advance=1)
                    return result

                # If failed and we have more retries available, wait before retrying
                if retry_count < max_retries:
                    wait_time = min(
                        2**retry_count, 60
                    )  # Exponential backoff capped at 60s
                    captain_type, seed, board_id = job[1], job[2], job[3]
                    logging.info(
                        f"Retrying {captain_type} seed{seed} {board_id} in {wait_time} seconds (attempt {retry_count + 2}/{max_retries + 1})"
                    )
                    time.sleep(wait_time)
                else:
                    # Final failure after all retries
                    captain_type, seed, board_id = job[1], job[2], job[3]
                    logging.error(
                        f"Failed {captain_type} seed{seed} {board_id} after {max_retries + 1} attempts"
                    )

            return None

        # Process remaining jobs in parallel
        if remaining_jobs:
            process_function = (
                process_job_with_retry if resume_mode else process_job_with_progress
            )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                retry_info = " with automatic retry" if resume_mode else ""
                logging.info(
                    f"Running {len(remaining_jobs)} games{retry_info} in parallel with up to {executor._max_workers} workers"
                )

                new_results = list(
                    filter(None, executor.map(process_function, remaining_jobs))
                )
        else:
            logging.info("No remaining games to process")
            new_results = []

    # ---------------------------------------------------------------------
    # 4. Collect final results from all configurations
    # ---------------------------------------------------------------------
    all_results = []

    for config in all_configs:
        config_results = load_existing_captain_results(config)
        all_results.extend(config_results)

    logging.info(f"Total results across all configurations: {len(all_results)}")
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
    elif args.experiment_dir:
        experiment_dir = create_experiment_dir(
            resolve_project_path(args.experiment_dir)
        )
    else:
        experiment_dir = create_experiment_dir()

    # Clear existing results if force restart
    if args.force_restart:
        results_dir = os.path.join(experiment_dir, "results")
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

    # Setup board IDs if not specified
    if args.board_ids is None:
        args.board_ids = [f"B{str(i).zfill(2)}" for i in range(1, 19)]

    # Setup logging to experiment directory
    log_handler = logging.FileHandler(
        os.path.join(experiment_dir, "captain_benchmark.log")
    )
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(log_handler)

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
        resume_mode=args.resume,
    )

    # Rebuild and save final summary from all results
    final_results = rebuild_captain_summary_from_results(experiment_dir)
    if final_results:
        results = final_results  # Use rebuilt results if available

    # Prepare data structures
    summaries_data = results

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
            "ProbabilisticCaptain_cot",
            "LLMDecisionCaptain",
            "LLMDecisionCaptain_cot",
            "EIGCaptain",
            "EIGCaptain_cot",
            "MAPEIGCaptain",
            "MAPEIGCaptain_cot",
            "ConditionalEIGCaptain",
            "ConditionalEIGCaptain_cot",
            "PlannerCaptain",
            "PlannerCaptain_cot",
        ],
        help="Captain types to benchmark",
    )

    # Model configuration
    parser.add_argument(
        "--captain-llm",
        type=str,
        required=True,
        default=None,
        help="LLM model to use for captain",
    )
    parser.add_argument(
        "--spotter-llm",
        type=str,
        required=True,
        default=None,
        help="LLM model to use for spotter",
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
