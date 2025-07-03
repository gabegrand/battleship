import argparse
import json
import logging
import multiprocessing.dummy as mp
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from battleship.agents import ActionData
from battleship.agents import EIGCalculator
from battleship.agents import Question
from battleship.board import Board
from battleship.spotters import CodeSpotterModel
from battleship.spotters import DirectSpotterModel
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

ALL_ANNOTATIONS = ["discourse", "stateful", "vague", "ambiguous", "unanswerable"]


@dataclass
class SpotterBenchmarkConfig:
    """Configuration for spotter benchmark experiments."""

    model_class: type
    model_string: str
    temperature: Optional[float]
    use_history: bool
    use_cot: bool
    use_captain_board: bool
    max_rounds: int
    max_questions: int
    output_dir: str
    experiment_name: str = None

    def __post_init__(self):
        if self.experiment_name is None:
            safe_model = self.model_string.replace("/", "-")
            self.experiment_name = (
                f"{safe_model}_{self.model_class.__name__}_{self.use_cot}"
            )


@dataclass
class QuestionContext:
    """Context data for a question being processed."""

    question_text: str
    board_id: str
    true_answer: str
    occ_tiles: str
    gold_annotations: Dict[str, bool]
    history: List[Dict[str, str]]
    round_id: str
    question_id: int


def create_experiment_dir(root: str = None) -> str:
    """Create experiment directory and return path."""
    if root is None:
        root = resolve_project_path("experiments/collaborative/spotter_benchmarks")

    experiment_dir = os.path.join(root, f"run_{time.strftime('%Y_%m_%d_%H_%M_%S')}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(experiment_dir, "rounds"), exist_ok=True)

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

    logging.info(f"Initial data size: {len(stage_df)}")

    # Apply gold annotation filters
    for annotation in gold_annotations:
        if annotation == "answer":
            stage_df = stage_df[(stage_df["gold_answer"].notna())]
        elif annotation in ALL_ANNOTATIONS:
            stage_df = stage_df[stage_df[f"gold_{annotation}"] == False]
        else:
            raise ValueError(f"Invalid annotation: {annotation}")

    logging.info(f"Filtered data size: {len(stage_df)}")

    # Build rounds-questions mapping
    rounds_questions_list = list(zip(stage_df["roundID"], stage_df["questionID"]))
    rounds_questions_list = [
        item for item in rounds_questions_list if item[0] in round_df["id"].tolist()
    ]
    rounds_questions_dict = {key: [] for key, _ in rounds_questions_list}
    for key, value in rounds_questions_list:
        rounds_questions_dict[key].append(value)

    return df, rounds_questions_dict


def extract_question_context(
    question_id: int, round_data: pd.DataFrame, round_id: str
) -> QuestionContext:
    """Extract context for a specific question."""
    question_history_data = round_data[round_data["questionID"] < question_id]
    question_data = round_data[round_data["questionID"] == question_id]

    # Extract question details
    question_text = question_data[question_data["messageType"] == "question"][
        "messageText"
    ].iloc[0]
    question_captain_board = question_data[question_data["messageType"] == "question"][
        "occTiles"
    ].iloc[0]
    question_board_id = question_data[question_data["messageType"] == "question"][
        "board_id"
    ].iloc[0]
    ground_truth_answer = question_data[question_data["messageType"] == "answer"][
        "gold_answer"
    ].iloc[0]

    # Extract gold annotations
    gold_annotations = {}
    for annotation in ALL_ANNOTATIONS:
        try:
            gold_annotations[annotation] = question_data[
                question_data["messageType"] == "answer"
            ][f"gold_{annotation}"].iloc[0]
        except (IndexError, KeyError):
            gold_annotations[annotation] = None

    # Extract history
    previous_decision_ids = sorted(
        list(set(question_history_data["questionID"].tolist()))
    )

    history = []
    for decision_id in previous_decision_ids:
        id_actions = question_history_data[
            question_history_data["questionID"] == decision_id
        ]
        try:
            decision = id_actions[id_actions["messageType"] == "decision"][
                "messageText"
            ].iloc[0]

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
                example["question"] = id_actions[
                    id_actions["messageType"] == "question"
                ]["messageText"].iloc[0]
                try:
                    example["answer"] = id_actions[
                        id_actions["messageType"] == "answer"
                    ]["messageText"].iloc[0]
                except (IndexError, KeyError):
                    example["answer"] = "Not answered"
            else:
                try:
                    example["move"] = id_actions[id_actions["messageType"] == "move"][
                        "messageText"
                    ].iloc[0]
                except (IndexError, KeyError):
                    continue

            history.append(example)
        except (IndexError, KeyError):
            continue

    return QuestionContext(
        question_text=question_text,
        board_id=question_board_id,
        true_answer=ground_truth_answer,
        occ_tiles=question_captain_board,
        gold_annotations=gold_annotations,
        history=history,
        round_id=round_id,
        question_id=question_id,
    )


def calculate_question_eig(question, board: Board) -> float:
    """Calculate Expected Information Gain for a question."""
    calculator = EIGCalculator(seed=0, spotter=None)
    return calculator.calculate_eig(
        None, board, pregenerated_question=question, samples=100
    )


def process_single_question(
    context: QuestionContext, config: SpotterBenchmarkConfig, round_data: pd.DataFrame
) -> Dict:
    """Process a single question and return results."""

    # Create round directory structure
    round_dir = os.path.join(
        config.output_dir, "rounds", f"round_{context.round_id}", context.question_id
    )
    spotter_dir = os.path.join(round_dir, "spotter")
    os.makedirs(spotter_dir, exist_ok=True)

    # Initialize spotter model
    spotter_model = config.model_class(
        board_id=context.board_id,
        board_experiment="collaborative",
        model_string=config.model_string,
        temperature=config.temperature,
        use_cot=config.use_cot,
        spotter_benchmark=True,
        round_id=f"{context.round_id}_{context.question_id}",
        json_path=os.path.join(spotter_dir, "spotter.json"),
    )

    # Process question
    question = Question(text=context.question_text)
    result = spotter_model.answer(
        question,
        occ_tiles=Board.from_occ_tiles(context.occ_tiles).to_numpy(),
        history=context.history if config.use_history else None,
    )

    # Extract answer
    answer_text = result.text.lower() if isinstance(result.text, str) else None

    # Calculate EIG if we have a code question
    eig_value = None
    if result.code_question:
        try:
            eig_value = calculate_question_eig(
                result.code_question, Board.from_occ_tiles(context.occ_tiles)
            )
        except Exception as e:
            logging.warning(f"Failed to calculate EIG: {e}")

    # Create result summary
    result_summary = {
        "model": config.model_string,
        "CoT": bool(config.use_cot),
        "spotterModel": config.model_class.__name__,
        "roundID": str(context.round_id),
        "questionID": int(context.question_id),
        "question": context.question_text,
        "program": result.code_question.fn_str if result.code_question else None,
        "occTiles": context.occ_tiles,
        "answer": answer_text,
        "EIG": float(eig_value) if eig_value is not None else None,
        "true_answer": context.true_answer,
        "is_correct": bool(answer_text == context.true_answer)
        if answer_text
        else False,
        "round_dir": round_dir,
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


def run_spotter_benchmark(
    df: pd.DataFrame,
    rounds_questions_dict: Dict[str, List[int]],
    config: SpotterBenchmarkConfig,
    processes: int = None,
) -> List[Dict]:
    """Run benchmark for a single spotter configuration."""

    if processes is None:
        processes = os.cpu_count()

    logging.info(f"Running benchmark: {config.experiment_name}")

    # Prepare all questions for processing
    all_contexts = []
    round_list = list(rounds_questions_dict.keys())[: config.max_rounds]

    for round_id in round_list:
        round_data = df[df["roundID"] == round_id]
        question_ids = sorted(rounds_questions_dict[round_id])

        # Filter to questions that actually have question messages
        valid_question_ids = [
            qid
            for qid in question_ids
            if "question"
            in round_data[round_data["questionID"] == qid]["messageType"].values
        ]

        valid_question_ids = valid_question_ids[: config.max_questions]

        for question_id in valid_question_ids:
            context = extract_question_context(question_id, round_data, round_id)
            all_contexts.append((context, config, round_data))

    # Process questions in parallel
    logging.info(f"Processing {len(all_contexts)} questions with {processes} processes")

    def process_wrapper(args):
        context, config, round_data = args
        return process_single_question(context, config, round_data)

    with mp.Pool(processes=processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_wrapper, all_contexts),
                total=len(all_contexts),
                desc=f"Processing {config.model_class.__name__} with {config.model_string}",
            )
        )

    # Save combined results
    results_file = os.path.join(config.output_dir, f"{config.experiment_name}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Saved {len(results)} results to {results_file}")
    return results


def run_all_experiments(
    df: pd.DataFrame,
    rounds_questions_dict: Dict[str, List[int]],
    language_models: List[str] = None,
    spotter_models: List[type] = None,
    cot_options: List[bool] = None,
    max_rounds: int = 20,
    max_questions: int = 20,
    use_history: bool = True,
    use_captain_board: bool = False,
    temperature: float = None,
    output_dir: str = "benchmark_results",
    processes: int = None,
) -> List[str]:
    """Run experiments with all combinations of models and options."""

    if language_models is None:
        language_models = ["gpt-4o"]
    if spotter_models is None:
        spotter_models = [DirectSpotterModel, CodeSpotterModel]
    if cot_options is None:
        cot_options = [True, False]

    experiment_files = []

    for llm in language_models:
        for spotter in spotter_models:
            for cot_option in cot_options:
                config = SpotterBenchmarkConfig(
                    model_class=spotter,
                    model_string=llm,
                    temperature=temperature,
                    use_history=use_history,
                    use_cot=cot_option,
                    use_captain_board=use_captain_board,
                    max_rounds=max_rounds,
                    max_questions=max_questions,
                    output_dir=output_dir,
                )

                results = run_spotter_benchmark(
                    df, rounds_questions_dict, config, processes
                )

                experiment_files.append(f"{config.experiment_name}.json")

    return experiment_files


def main():
    """Main entry point for spotter benchmarks."""
    args = parse_arguments()

    # Create experiment directory
    experiment_dir = create_experiment_dir()

    # Save command used to run the script
    command = " ".join(
        ["python"]
        + ["run_spotter_benchmarks.py"]
        + [
            f"--{arg.replace('_', '-')} {getattr(args, arg)}"
            for arg in vars(args)
            if getattr(args, arg) is not None
        ]
    )
    with open(os.path.join(experiment_dir, "command.txt"), "w") as f:
        f.write(command)

    # Setup logging to experiment directory
    log_handler = logging.FileHandler(os.path.join(experiment_dir, "benchmark.log"))
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(log_handler)

    logging.info(f"Starting spotter benchmark experiment in {experiment_dir}")

    # Convert spotter model names to classes
    spotter_model_map = {
        "CodeSpotterModel": CodeSpotterModel,
        "DirectSpotterModel": DirectSpotterModel,
    }

    spotter_models = [
        spotter_model_map[name]
        for name in args.spotter_models
        if name in spotter_model_map
    ]

    # Convert COT options
    cot_options = [opt.lower() == "true" for opt in args.cot_options]

    # Load data
    df, rounds_questions_dict = load_benchmark_data(
        stages_path=args.stages,
        rounds_path=args.rounds,
        gold_annotations=args.gold_annotations,
    )

    # Run experiments
    result_files = run_all_experiments(
        df=df,
        rounds_questions_dict=rounds_questions_dict,
        language_models=args.models,
        spotter_models=spotter_models,
        cot_options=cot_options,
        max_rounds=args.max_rounds,
        max_questions=args.max_questions,
        use_history=args.use_history,
        use_captain_board=args.use_captain_board,
        temperature=args.temperature,
        output_dir=experiment_dir,
        processes=args.processes,
    )

    # Create summary
    summary = {
        "experiment_dir": experiment_dir,
        "total_experiments": len(result_files),
        "result_files": result_files,
        "args": vars(args),
    }

    with open(os.path.join(experiment_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Experiment completed!")
    logging.info(f"Results saved to: {experiment_dir}")
    for result_file in result_files:
        logging.info(f"- {result_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks on spotter models with customizable options."
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
        "--models",
        type=str,
        nargs="+",
        default=["gpt-4o"],
        help="Space-separated list of language models to test.",
    )
    parser.add_argument(
        "--spotter_models",
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
        "--max_rounds",
        type=int,
        default=20,
        help="Maximum number of rounds to process.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=20,
        help="Maximum number of questions per round to process.",
    )
    parser.add_argument(
        "--use_history",
        action="store_true",
        help="Flag to include history in the benchmark.",
    )
    parser.add_argument(
        "--use_captain_board",
        action="store_true",
        help="Flag to use captain board in the model.",
    )
    parser.add_argument(
        "--gold_annotations",
        type=str,
        nargs="+",
        choices=ALL_ANNOTATIONS + ["answer"],
        default=[],
        help="Space-separated list of gold annotations to filter on.",
    )
    parser.add_argument(
        "--cot_options",
        type=str,
        nargs="+",
        default=["True", "False"],
        help="Space-separated list of chain-of-thought options (True/False).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel processes to use.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
