import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from battleship.agents import Answer
from battleship.agents import EIGCalculator
from battleship.agents import Question
from battleship.board import Board
from battleship.spotters import create_spotter
from battleship.utils import PROJECT_ROOT
from battleship.utils import resolve_project_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("spotter_benchmark.log"),
        logging.StreamHandler(),
    ],
)

# Suppress HTTP request logs from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

GOLD_ANNOTATIONS = ["discourse", "stateful", "vague", "ambiguous", "unanswerable"]


@dataclass
class SpotterBenchmarkConfig:
    """Configuration for spotter benchmark experiments."""

    spotter_type: str
    llm: str
    temperature: Optional[float]
    use_history: bool
    use_cot: bool
    max_rounds: int
    max_questions: int
    experiment_dir: str
    eig_samples: int = 1000
    experiment_name: str = None

    def __post_init__(self):
        if self.experiment_name is None:
            safe_model = self.llm.replace("/", "-")
            self.experiment_name = (
                f"{safe_model}_{self.spotter_type}{'_cot' if self.use_cot else ''}"
            )


@dataclass
class QuestionContext:
    """Context data for a question being processed."""

    question_text: str
    board_id: str
    gold_answer_text: str
    gold_answer_value: Optional[bool]
    board_state: str
    gold_annotations: Dict[str, bool]
    history: List[Dict[str, str]]
    round_id: str
    question_id: int


def create_experiment_dir(experiment_dir: str = None) -> str:
    """Create experiment directory and return path."""
    if experiment_dir is None:
        experiment_dir = resolve_project_path(
            os.path.join(
                "experiments",
                "collaborative",
                "spotter_benchmarks",
                f"run_{time.strftime('%Y_%m_%d_%H_%M_%S')}",
            )
        )

    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def load_benchmark_data(
    stages_path: str, rounds_path: str, gold_annotations: List[str] = None
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """Load and filter benchmark data."""
    if gold_annotations is None:
        gold_annotations = []

    # Resolve paths relative to project root
    stages_path = resolve_project_path(stages_path)
    rounds_path = resolve_project_path(rounds_path)

    stage_df = pd.read_csv(str(stages_path))
    round_df = pd.read_csv(str(rounds_path))

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
            "gold_discourse",
            "gold_stateful",
            "gold_vague",
            "gold_ambiguous",
            "gold_unanswerable",
        ]
    ]
    df = filtered_stage_df.merge(
        board_ids, left_on="roundID", right_on="id", how="inner"
    )

    # Build rounds-questions mapping from the merged dataframe
    rounds_questions_dict = defaultdict(set)

    for (round_id, question_id), group in df.groupby(["roundID", "questionID"]):
        # Ensure that the question has a question row
        if "question" not in group["messageType"].values:
            continue

        # Ensure that the question has an answer row
        answer_data = group[group["messageType"] == "answer"]
        if answer_data.empty:
            continue
        answer_data = answer_data.iloc[0]

        # Ensure that the gold answer is not NaN
        if pd.isna(answer_data["gold_answer"]):
            continue

        # Filter out questions where the gold annotation field is True
        for annotation in gold_annotations:
            if answer_data[f"gold_{annotation}"] == True:
                continue

        rounds_questions_dict[str(round_id)].add(question_id)

    logging.warning(
        f"Found {df['roundID'].nunique() - len(rounds_questions_dict)} rounds with no valid questions - these will be skipped."
    )

    logging.info(
        f"Found {len(rounds_questions_dict)} rounds with {sum([len(qids) for qids in rounds_questions_dict.values()])} total valid questions"
    )

    return df, rounds_questions_dict


def safe_extract_value(df: pd.DataFrame, message_type: str, column: str, default=None):
    """Safely extract a value from a filtered DataFrame."""
    filtered = df[df["messageType"] == message_type]
    if filtered.empty:
        return default
    try:
        return filtered[column].iloc[0]
    except (IndexError, KeyError):
        return default


def extract_gold_annotations(question_data: pd.DataFrame) -> Dict[str, bool]:
    """Extract gold annotations from question data."""
    gold_annotations = {}
    answer_data = question_data[question_data["messageType"] == "answer"]

    if answer_data.empty:
        return {annotation: None for annotation in GOLD_ANNOTATIONS}

    gold_annotations["answer"] = answer_data["gold_answer"].iloc[0]

    for annotation in GOLD_ANNOTATIONS:
        try:
            gold_annotations[annotation] = answer_data[f"gold_{annotation}"].iloc[0]
        except (IndexError, KeyError):
            gold_annotations[annotation] = None

    return gold_annotations


def extract_history_entry(id_actions: pd.DataFrame) -> Optional[Dict[str, str]]:
    """Extract a single history entry from action data."""
    # Get decision type
    decision = safe_extract_value(id_actions, "decision", "messageText")
    if decision is None:
        return None

    # Remap "fire" -> "move"
    if decision == "fire":
        decision = "move"

    example = {
        "decision": decision,
        "question": None,
        "answer": None,
        "move": None,
    }

    if decision == "question":
        example["question"] = safe_extract_value(id_actions, "question", "messageText")
        example["answer"] = safe_extract_value(
            id_actions, "answer", "messageText", default="Not answered"
        )
        # Skip entries without valid question data
        if example["question"] is None or example["answer"] is None:
            return None
    else:
        # Skip entries without valid move data
        example["move"] = safe_extract_value(id_actions, "move", "messageText")
        if example["move"] is None:
            return None

    return example


def extract_question_context(
    question_id: int, round_data: pd.DataFrame, round_id: str
) -> QuestionContext:
    """Extract context for a specific question."""
    question_history_data = round_data[round_data["questionID"] < question_id]
    question_data = round_data[round_data["questionID"] == question_id]

    # Extract question details using safe extraction
    question_text = safe_extract_value(question_data, "question", "messageText")
    question_captain_board = safe_extract_value(question_data, "question", "occTiles")
    question_board_id = safe_extract_value(question_data, "question", "board_id")
    ground_truth_answer = safe_extract_value(question_data, "answer", "gold_answer")

    # Extract gold annotations
    gold_annotations = extract_gold_annotations(question_data)

    # Extract history
    previous_decision_ids = sorted(
        list(set(question_history_data["questionID"].tolist()))
    )

    history = []
    for decision_id in previous_decision_ids:
        id_actions = question_history_data[
            question_history_data["questionID"] == decision_id
        ]

        history_entry = extract_history_entry(id_actions)
        if history_entry is not None:
            history.append(history_entry)

    return QuestionContext(
        question_text=question_text,
        board_id=question_board_id,
        gold_answer_text=ground_truth_answer,
        gold_answer_value=Answer.parse(ground_truth_answer),
        board_state=question_captain_board,
        gold_annotations=gold_annotations,
        history=history,
        round_id=round_id,
        question_id=question_id,
    )


def run_single_question(
    context: QuestionContext, config: SpotterBenchmarkConfig, round_data: pd.DataFrame
) -> Dict:
    """Process a single question and return results."""

    # Create round directory structure
    spotter_dir = os.path.join(
        config.experiment_dir,
        "spotters",
        config.experiment_name,
        f"round_{context.round_id}",
    )
    json_path = os.path.join(
        spotter_dir, f"question_{str(context.question_id).zfill(2)}.json"
    )
    os.makedirs(spotter_dir, exist_ok=True)

    # Initialize spotter model
    spotter_model = create_spotter(
        spotter_type=config.spotter_type,
        board_id=context.board_id,
        board_experiment="collaborative",
        llm=config.llm,
        temperature=config.temperature,
        use_cot=config.use_cot,
        json_path=json_path,
    )

    # Process question
    question = Question(text=context.question_text)
    result = spotter_model.answer(
        question,
        board=Board.from_occ_tiles(context.board_state),
        history=context.history if config.use_history else None,
    )

    # Calculate EIG if we have a code question
    eig_value = None
    if result is not None and result.code_question:
        try:
            calculator = EIGCalculator(seed=0, samples=config.eig_samples)
            eig_value = calculator(
                result.code_question, Board.from_occ_tiles(context.board_state)
            )
        except Exception as e:
            logging.warning(f"Failed to calculate EIG: {e}")

    # Create result summary with parsed values
    result_summary = {
        "llm": config.llm,
        "use_cot": bool(config.use_cot),
        "spotter_type": config.spotter_type,
        "round_id": str(context.round_id),
        "question_id": int(context.question_id),
        "question": context.question_text,
        "program": result.code_question.fn_str
        if result is not None and result.code_question
        else None,
        "board_state": context.board_state,
        "answer_text": result.text if result is not None else None,
        "answer_value": result.value if result is not None else None,
        "eig_value": float(eig_value) if eig_value is not None else None,
        "gold_answer_text": context.gold_answer_text,
        "gold_answer_value": context.gold_answer_value,
        "is_correct": bool(result.value == context.gold_answer_value)
        if result is not None
        else False,
        "spotter_json": os.path.relpath(json_path, PROJECT_ROOT),
    }

    # Add gold annotations (ensure proper type conversion)
    for annotation, value in context.gold_annotations.items():
        if value is not None:
            result_summary[f"gold_{annotation}"] = (
                bool(value) if isinstance(value, (bool, np.bool_)) else value
            )
        else:
            result_summary[f"gold_{annotation}"] = None

    return result_summary


def run_single_experiment(
    df: pd.DataFrame,
    rounds_questions_dict: Dict[str, List[int]],
    config: SpotterBenchmarkConfig,
    max_workers: int = None,
) -> List[Dict]:
    """Run benchmark for a single spotter configuration."""

    logging.info(f"Running benchmark: {config.experiment_name}")

    # Prepare all questions for processing
    all_contexts = []
    round_list = list(rounds_questions_dict.keys())
    if config.max_rounds is not None:
        round_list = round_list[: config.max_rounds]

    for round_id in round_list:
        round_data = df[df["roundID"] == round_id]
        question_ids = sorted(rounds_questions_dict[round_id])

        if config.max_questions is not None:
            question_ids = question_ids[: config.max_questions]

        for question_id in question_ids:
            context = extract_question_context(question_id, round_data, round_id)
            all_contexts.append((context, config, round_data))

    def process_wrapper(args):
        context, config, round_data = args
        return run_single_question(context, config, round_data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        logging.info(f"Running with max {executor._max_workers} worker threads...")
        # executor.map() preserves order - results will be in same order as all_contexts
        results = list(
            tqdm(
                executor.map(process_wrapper, all_contexts),
                total=len(all_contexts),
                desc=f"Processing {config.spotter_type} with {config.llm}",
            )
        )

    logging.info(f"Completed {len(results)} questions for {config.experiment_name}")
    return results


def run_all_experiments(
    df: pd.DataFrame,
    rounds_questions_dict: Dict[str, List[int]],
    language_models: List[str] = ["gpt-4o-mini"],
    spotter_models: List[str] = ["DirectSpotterModel", "CodeSpotterModel"],
    cot_options: List[bool] = [True, False],
    max_rounds: int = None,
    max_questions: int = None,
    use_history: bool = True,
    temperature: float = None,
    experiment_dir: str = None,
    eig_samples: int = 1000,
    max_workers: int = None,
) -> List[Dict]:
    """Run experiments with all combinations of models and options."""

    all_results = []

    for llm in language_models:
        for spotter in spotter_models:
            for cot_option in cot_options:
                config = SpotterBenchmarkConfig(
                    spotter_type=spotter,
                    llm=llm,
                    temperature=temperature,
                    use_history=use_history,
                    use_cot=cot_option,
                    max_rounds=max_rounds,
                    max_questions=max_questions,
                    experiment_dir=experiment_dir,
                    eig_samples=eig_samples,
                )

                results = run_single_experiment(
                    df, rounds_questions_dict, config, max_workers
                )

                all_results.extend(results)

    return all_results


def main():
    """Main entry point for spotter benchmarks."""
    start_time = time.time()
    args = parse_arguments()

    # Create experiment directory
    if args.experiment_dir:
        experiment_dir = create_experiment_dir(
            resolve_project_path(args.experiment_dir)
        )
    else:
        experiment_dir = create_experiment_dir()

    # Save the command used to run the script
    command = " ".join(["python"] + sys.argv)
    command_path = os.path.join(experiment_dir, "command.txt")
    with open(command_path, "w") as f:
        f.write(command)
    print(f"Command saved to {command_path}")

    # Setup logging to experiment directory
    log_handler = logging.FileHandler(os.path.join(experiment_dir, "benchmark.log"))
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(log_handler)

    logging.info(f"Starting spotter benchmark experiment in {experiment_dir}")

    # Load data
    df, rounds_questions_dict = load_benchmark_data(
        stages_path=args.stages,
        rounds_path=args.rounds,
        gold_annotations=args.gold_annotations,
    )

    # Run experiments
    all_results = run_all_experiments(
        df=df,
        rounds_questions_dict=rounds_questions_dict,
        language_models=args.llms,
        spotter_models=args.spotter_models,
        cot_options=args.cot_options,
        max_rounds=args.max_rounds,
        max_questions=args.max_questions,
        use_history=args.use_history,
        temperature=args.temperature,
        experiment_dir=experiment_dir,
        eig_samples=args.eig_samples,
        max_workers=args.max_workers,
    )

    # Save results to summary.json (following captain benchmarks pattern)
    summary_path = os.path.join(experiment_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Create metadata file for experiment info
    metadata = {
        "experiment_dir": str(experiment_dir),
        "total_results": len(all_results),
        "experiment_args": vars(args),
        "experiments_run": len(args.llms)
        * len(args.spotter_models)
        * len(args.cot_options),
    }

    metadata_path = os.path.join(experiment_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Experiment completed!")
    logging.info(f"Results saved to: {experiment_dir}")
    logging.info(f"- summary.json: {len(all_results)} question results")
    logging.info(f"- metadata.json: experiment configuration")
    logging.info(f"- rounds/: individual spotter.json files with ActionData")

    # Log overall runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    logging.info(
        f"Overall runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)"
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks on spotter models with customizable options."
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Full path to experiment directory. If not provided, will create a timestamped directory in experiments/collaborative/spotter_benchmarks.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="experiments/collaborative/data/battleship-final-data/gold-v2/gold-v2.csv",
        help="Path to the stages CSV file containing question data.",
    )
    parser.add_argument(
        "--rounds",
        type=str,
        default="experiments/collaborative/data/battleship-final-data/round.csv",
        help="Path to the rounds CSV file containing round data.",
    )
    parser.add_argument(
        "--llms",
        type=str,
        nargs="+",
        default=["gpt-4o-mini"],
        help="Space-separated list of language models to test.",
    )
    parser.add_argument(
        "--spotter-models",
        type=str,
        nargs="+",
        default=["CodeSpotterModel", "DirectSpotterModel"],
        help="Space-separated list of spotter models to test.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="The model temperature to use.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum number of rounds to process (default: None, process all rounds).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions per round to process (default: None, process all questions).",
    )
    parser.add_argument(
        "--use-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include history in the spotter context.",
    )

    parser.add_argument(
        "--gold-annotations",
        type=str,
        nargs="+",
        choices=GOLD_ANNOTATIONS,
        default=[],
        help="Space-separated list of gold annotations to filter on.",
    )
    parser.add_argument(
        "--cot-options",
        type=str_to_bool,
        nargs="+",
        default=[False, True],
        help="Space-separated list of chain-of-thought options. Use 'true'/'false'.",
    )
    parser.add_argument(
        "--eig-samples",
        type=int,
        default=1000,
        help="Number of samples for EIG calculations. Only used for CodeSpotterModel.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of worker threads to use for parallelism (defaults to Python's ThreadPoolExecutor default)",
    )

    return parser.parse_args()


def str_to_bool(value: str) -> bool:
    """Convert string representation of boolean to actual boolean."""
    if value.lower() in ("true", "1"):
        return True
    elif value.lower() in ("false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


if __name__ == "__main__":
    main()
