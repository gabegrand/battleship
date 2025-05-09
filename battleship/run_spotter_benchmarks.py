import argparse
import csv
import logging
import multiprocessing.dummy as mp
import os
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from battleship.agents import Counter
from battleship.agents import EIGCalculator
from battleship.agents import Question
from battleship.board import Board
from battleship.spotters import CodeSpotterModel
from battleship.spotters import DirectSpotterModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("spotter_benchmark.log"),
        logging.StreamHandler(),
    ],
)

# LOAD DATA

ALL_ANNOTATIONS = ["discourse", "stateful", "vague", "ambiguous", "unanswerable"]


def load_data(
    stages_path="/home/ubuntu/repo_battleship/gold-v2.csv",
    rounds_path="/home/ubuntu/repo_battleship/battleship/experiments/collaborative/battleship-final-data/round.csv",
    goldAnnotations=[],
):
    stage_df = pd.read_csv(stages_path)
    round_df = pd.read_csv(rounds_path)

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

    print(len(stage_df))

    for annotation in goldAnnotations:
        if annotation == "answer":
            stage_df = stage_df[(stage_df["gold_answer"].notna())]
        if annotation in ALL_ANNOTATIONS:
            stage_df = stage_df[stage_df[f"gold_{annotation}"] == False]

    print(len(stage_df))

    rounds_questions_list = list(zip(stage_df["roundID"], stage_df["questionID"]))
    rounds_questions_list = [
        i for i in rounds_questions_list if i[0] in round_df["id"].tolist()
    ]
    rounds_questions_dict = {key: [] for key, _ in rounds_questions_list}
    for key, value in rounds_questions_list:
        rounds_questions_dict[key].append(value)

    return df, rounds_questions_dict


def calculate_EIG(question, board):
    """EIG Calculator"""
    calculator = EIGCalculator(seed=0, spotter=None)
    return calculator.calculate_eig(
        None, board, pregenerated_question=question, samples=100
    )


def retrieve_context(question_id, round_data):
    question_history_data = round_data[round_data["questionID"] < question_id]
    question_data = round_data[round_data["questionID"] == question_id]

    # QUESTION AND ITS BOARD
    question_text = question_data[question_data["messageType"] == "question"][
        "messageText"
    ].tolist()[0]
    question_captain_board = question_data[question_data["messageType"] == "question"][
        "occTiles"
    ].tolist()[0]
    question_board_id = question_data[question_data["messageType"] == "question"][
        "board_id"
    ].tolist()[0]
    ground_truth_answer = question_data[question_data["messageType"] == "answer"][
        "gold_answer"
    ].tolist()[0]

    # Extract all gold annotations
    gold_annotations = {}
    for annotation in ALL_ANNOTATIONS:
        try:
            gold_annotations[annotation] = question_data[
                question_data["messageType"] == "answer"
            ][f"gold_{annotation}"].tolist()[0]
        except (IndexError, KeyError):
            gold_annotations[annotation] = None

    # QUESTION HISTORY decision, question, answer, move
    previous_decision_ids = sorted(
        list(set(question_history_data["questionID"].tolist()))
    )

    examples = []
    for id in previous_decision_ids:
        decision, question, answer, move = None, None, None, None
        id_actions = question_history_data[question_history_data["questionID"] == id]
        try:
            decision = id_actions[id_actions["messageType"] == "decision"][
                "messageText"
            ].tolist()[0]

            # Remap "fire" -> "move"
            if decision == "fire":
                decision = "move"

            if decision == "question":
                question = id_actions[id_actions["messageType"] == "question"][
                    "messageText"
                ].tolist()[0]
                try:
                    answer = id_actions[id_actions["messageType"] == "answer"][
                        "messageText"
                    ].tolist()[0]
                except:
                    answer = "Not answered"
            else:
                move = id_actions[id_actions["messageType"] == "move"][
                    "messageText"
                ].tolist()[0]

            dict_example = {
                "decision": decision,
                "question": question,
                "answer": answer,
                "move": move,
            }
            examples.append(dict_example)
        except IndexError:
            continue

    return {
        "context": {
            "text": question_text,
            "board_id": question_board_id,
            "true_answer": ground_truth_answer,
            "occ_tiles": question_captain_board,
            "gold_annotations": gold_annotations,
        },
        "history": examples,
    }


def process_question_data(question_data):
    """Process a single question with pre-extracted data"""

    (
        round_data,
        round_id,
        question_id,
        model_class,
        model_string,
        temperature,
        use_history,
        use_cache,
        use_cot,
        use_captain_board,
        output_dir,
    ) = question_data

    question_context = retrieve_context(question_id, round_data)

    examples = question_context["history"]
    question_board = question_context["context"]["board_id"]
    question_text = question_context["context"]["text"]
    question_captain_board = question_context["context"]["occ_tiles"]
    ground_truth_answer = question_context["context"]["true_answer"]
    gold_annotations = question_context["context"]["gold_annotations"]

    decision_counter = Counter()
    index_counter = Counter()

    spotter_model = model_class(
        board_id=question_board,
        board_experiment="collaborative",
        use_cache=use_cache,
        model_string=model_string,
        temperature=temperature,
        use_cot=use_cot,
        decision_counter=decision_counter,
        index_counter=index_counter,
        spotter_benchmark=True,
    )

    question = Question(text=question_text)
    result, answer_cache = spotter_model.answer(
        question,
        occ_tiles=Board.from_occ_tiles(question_captain_board).to_numpy(),
        history=examples if use_history else None,
    )

    # Get answer from result
    if isinstance(result.text, str):
        answer_text = result.text.lower()
    else:
        answer_text = None

    # Extract necessary data for CSV
    data_row = {
        "model": model_string,
        "CoT": use_cot,
        "spotterModel": model_class.__name__,
        "roundID": round_id,
        "questionID": question_id,
        "question": question_text,
        "program": result.code_question.fn_str if result.code_question else None,
        "occTiles": question_captain_board,
        "answer": answer_text,
        "EIG": calculate_EIG(
            result.code_question, Board.from_occ_tiles(question_captain_board)
        )
        if result.code_question
        else None,
        "full_completion": answer_cache.prompts[-1].full_completion,
        "prompt": answer_cache.prompts[-1].prompt,
        "true_answer": ground_truth_answer,
        "is_correct": answer_text == ground_truth_answer,
    }

    # Add all gold annotations to the data row
    for annotation, value in gold_annotations.items():
        data_row[f"gold_{annotation}"] = value

    # Replace slashes with a safe character for filenames (e.g., underscores or hyphens)
    safe_model_string = model_string.replace("/", "-")

    # Append to model-specific CSV file using the safe model string
    csv_path = os.path.join(
        output_dir, f"{safe_model_string}_{model_class.__name__}_{use_cot}.csv"
    )

    # Use a lock to prevent race conditions when writing to the same file
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(data_row.keys()))
        writer.writerow(data_row)

    return data_row["is_correct"]


def prepare_question_data(
    df,
    rounds_question_ids,
    model,
    model_string,
    temperature,
    use_history,
    max_rounds,
    max_questions,
    use_cache,
    use_cot,
    use_captain_board,
    output_dir,
):
    """Prepare all question data for parallel processing"""
    all_question_data = []
    round_list = list(rounds_question_ids.keys())[:max_rounds]

    for round_id in round_list:
        round_data = df[df["roundID"] == round_id]
        question_ids = sorted(rounds_question_ids[round_id])

        clean_q_ids = list(
            set(
                [
                    i
                    for i in question_ids
                    if "question"
                    in round_data[round_data["questionID"] == i]["messageType"].tolist()
                ]
            )
        )

        clean_q_ids = clean_q_ids[:max_questions]

        for question_id in clean_q_ids:
            all_question_data.append(
                (
                    round_data,
                    round_id,
                    question_id,
                    model,
                    model_string,
                    temperature,
                    use_history,
                    use_cache,
                    use_cot,
                    use_captain_board,
                    output_dir,
                )
            )

    return all_question_data


def benchmark_on_rounds(
    df,
    rounds_question_ids,
    model,
    model_string,
    temperature=1.0,
    use_history=False,
    max_rounds=10,
    max_questions=10,
    use_cache=True,
    use_cot=False,
    use_captain_board=False,
    output_dir="benchmark_results",
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    safe_model_string = model_string.replace("/", "-")

    # Create the model-specific CSV file with headers
    csv_path = os.path.join(
        output_dir, f"{safe_model_string}_{model.__name__}_{use_cot}.csv"
    )

    # Get a sample question to determine all the fieldnames including gold annotations
    sample_question_data = next(
        iter(
            prepare_question_data(
                df,
                rounds_question_ids,
                model,
                model_string,
                temperature,
                use_history,
                1,
                1,
                use_cache,
                use_cot,
                use_captain_board,
                output_dir,
            )
        ),
        None,
    )

    if sample_question_data:
        sample_context = retrieve_context(
            sample_question_data[2], sample_question_data[0]
        )
        gold_annotation_keys = [
            f"gold_{k}" for k in sample_context["context"]["gold_annotations"].keys()
        ]
    else:
        gold_annotation_keys = [f"gold_{k}" for k in ALL_ANNOTATIONS]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = [
                "model",
                "CoT",
                "spotterModel",
                "roundID",
                "questionID",
                "question",
                "program",
                "occTiles",
                "answer",
                "EIG",
                "full_completion",
                "prompt",
                "true_answer",
                "is_correct",
            ] + gold_annotation_keys

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Prepare all question data for parallel processing
    all_question_data = prepare_question_data(
        df,
        rounds_question_ids,
        model,
        model_string,
        temperature,
        use_history,
        max_rounds,
        max_questions,
        use_cache,
        use_cot,
        use_captain_board,
        output_dir,
    )

    # Process all questions in parallel with max 10 workers
    with mp.Pool(processes=args.processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_question_data, all_question_data),
                total=len(all_question_data),
                desc=f"Processing {model.__name__} with {model_string}",
            )
        )

    # Calculate accuracy
    if results:
        accuracy = sum(1 for r in results if r) / len(results)
    else:
        accuracy = 0.0

    return accuracy


def combine_results(output_dir="benchmark_results"):
    """Combine all individual CSV files into a single results file"""
    all_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".csv")
        and f != "combined_results.csv"
        and f != "accuracy_summary.csv"
    ]

    if not all_files:
        print("No result files found to combine.")
        return None

    # Read all CSVs and determine all possible columns
    all_columns = set()
    dfs = []

    for file_path in all_files:
        df = pd.read_csv(file_path)
        all_columns.update(df.columns)
        dfs.append(df)

    # Create a combined dataframe with all columns
    combined_df = pd.DataFrame(columns=list(all_columns))

    # Append each dataframe, filling missing columns with NaN
    for df in dfs:
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_path = os.path.join(output_dir, "combined_results.csv")
    combined_df.to_csv(combined_path, index=False)

    print(f"Combined results saved to {combined_path}")
    return combined_df


def run_experiments(
    df,
    rounds_questions_dict,
    language_models=["gpt-4o"],
    spotter_models=[DirectSpotterModel, CodeSpotterModel],
    cot_options=[True, False],
    max_rounds=20,
    max_questions=20,
    use_history=True,
    use_cache=False,
    use_captain_board=False,
    output_dir="benchmark_results",
):
    """Run experiments with all combinations of models and options"""
    results = {"language_model": [], "spotter_model": [], "accuracy": [], "cot": []}

    for llm in language_models:
        for spotter in spotter_models:
            for cot_option in cot_options:
                print(
                    f"Benchmarking {spotter.__name__} with language model {llm}, COT: {cot_option}"
                )

                accuracy = benchmark_on_rounds(
                    df=df,
                    rounds_question_ids=rounds_questions_dict,
                    model=spotter,
                    model_string=llm,
                    max_rounds=max_rounds,
                    max_questions=max_questions,
                    use_cache=use_cache,
                    use_cot=cot_option,
                    use_history=use_history,
                    use_captain_board=use_captain_board,
                    output_dir=output_dir,
                )

                results["language_model"].append(llm)
                results["spotter_model"].append(spotter.__name__)
                results["cot"].append(cot_option)
                results["accuracy"].append(accuracy)

                print(f"Accuracy: {accuracy * 100:.2f}%")

    # Convert the results to a df for easier manipulation
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "accuracy_summary.csv"), index=False)

    # Combine all individual result files
    combine_results(output_dir)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarks on rounds with customizable options."
    )
    parser.add_argument(
        "--stages",
        type=str,
        help="Path to the stage.csv file.",
    )
    parser.add_argument(
        "--rounds",
        type=str,
        help="Path to the round.csv file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results.",
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
        "--use_cache",
        action="store_true",
        help="Flag to enable cache usage.",
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
        help="Space-separated list of gold annotations from [answer, ambiguous, contextual, unanswerable].",
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

    args = parser.parse_args()

    # Convert string spotter model names to actual classes
    spotter_model_map = {
        "CodeSpotterModel": CodeSpotterModel,
        "DirectSpotterModel": DirectSpotterModel,
    }

    spotter_models = [
        spotter_model_map[name]
        for name in args.spotter_models
        if name in spotter_model_map
    ]

    # Convert string cot options to boolean
    cot_options = [opt.lower() == "true" for opt in args.cot_options]

    df, rounds_questions_dict = load_data(
        stages_path=args.stages,
        rounds_path=args.rounds,
        goldAnnotations=args.gold_annotations,
    )

    # Run the experiments
    results_df = run_experiments(
        df=df,
        rounds_questions_dict=rounds_questions_dict,
        language_models=args.models,
        spotter_models=spotter_models,
        cot_options=cot_options,
        max_rounds=args.max_rounds,
        max_questions=args.max_questions,
        use_history=args.use_history,
        use_cache=args.use_cache,
        use_captain_board=args.use_captain_board,
        output_dir=args.output_dir,
    )

    print("Experiment completed!")
    print(f"Results summary:\n{results_df}")
