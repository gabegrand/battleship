import argparse

import pandas as pd
from tqdm import tqdm

from battleship.agents import CacheMode
from battleship.agents import CodeSpotterModel
from battleship.agents import DirectSpotterModel
from battleship.agents import Question

# LOAD DATA


def load_data(
    stages_path="/home/ubuntu/repo_battleship/temp/gold_annotations_partial.csv",
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
            "goldAnswer",
        ]
    ]
    df = filtered_stage_df.merge(
        board_ids, left_on="roundID", right_on="id", how="inner"
    )

    for annotation in goldAnnotations:
        if annotation == "answer":
            stage_df = stage_df[(stage_df["goldAnswer"].isin(["yes", "no"]))]
        if annotation in ["ambiguous", "contextual", "unanswerable"]:
            stage_df = stage_df[stage_df[f"gold{annotation.capitalize()}"] == False]

    rounds_questions_list = list(zip(stage_df["roundID"], stage_df["questionID"]))
    rounds_questions_list = [
        i for i in rounds_questions_list if i[0] in round_df["id"].tolist()
    ]
    rounds_questions_dict = {key: [] for key, value in rounds_questions_list}
    for key, value in rounds_questions_list:
        rounds_questions_dict[key].append(value)

    return df, rounds_questions_dict


# -----------------------------------------------------------------------


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
        "goldAnswer"
    ].tolist()[0]

    # QUESTION HISTORY decision, question, answer, move
    # {"question":question, "code": q_ex.codestr, "answer": a_ex.text, "board_id": board}

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
            if int(id) != 0:
                print(f"Error with decision id {id}")

    return {
        "context": {
            "text": question_text,
            "board_id": question_board_id,
            "true_answer": ground_truth_answer,
            "occ_tiles": question_captain_board,
        },
        "history": examples,
    }


def benchmark_on_rounds(
    df,
    rounds_question_ids,
    model,
    model_string,
    temperature=None,
    use_history=False,
    max_rounds=10,
    max_questions=10,
    cache_mode=CacheMode.NO_CACHE,
    use_cot=False,
    use_captain_board=False,
):
    correct = []
    failed_cache_ids = []
    round_list = list(rounds_question_ids.keys())
    for idx, round in enumerate(round_list[:max_rounds]):
        round_data = df[df["roundID"] == round]
        question_ids = sorted(rounds_question_ids[round])

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

        for question_id in tqdm(
            clean_q_ids[:max_questions],
            desc=f"Round {str(round)}, {idx+1}/{max_rounds}",
        ):
            question_context = retrieve_context(question_id, round_data)

            examples = question_context["history"]
            question_board = question_context["context"]["board_id"]
            question_text = question_context["context"]["text"]
            question_captain_board = question_context["context"]["occ_tiles"]

            ground_truth_answer = question_context["context"]["true_answer"]

            spotter_model = model(
                board_id=question_board,
                board_experiment="collaborative",
                cache_mode=cache_mode,
                model_string=model_string,
                temperature=temperature,
                use_cot=use_cot,
            )

            question = Question(text=question_text)
            model_answer = spotter_model.answer(
                question,
                history=examples if use_history else None,
                occ_tiles=question_captain_board if use_captain_board else None,
            )

            if model_answer.text is not None:
                try:
                    lowercase_bool = model_answer.text.lower()
                except AttributeError:
                    print("Non-string answer:", model_answer.text)
                    failed_cache_ids.append(
                        [
                            question.get_cache_key(question_board),
                            (model_answer.text, ground_truth_answer),
                        ]
                    )
                    continue

                correct_bool = lowercase_bool == ground_truth_answer

                if not correct_bool:
                    failed_cache_ids.append(
                        [
                            question.get_cache_key(question_board),
                            (lowercase_bool, ground_truth_answer),
                        ]
                    )

                correct.append(correct_bool)
            else:
                pass
    return sum(correct) / len(correct), failed_cache_ids


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
        "--model",
        type=str,
        default="CodeSpotterModel",
        choices=["CodeSpotterModel", "DirectSpotterModel"],
        help="The model type to use.",
    )
    parser.add_argument(
        "--model_string",
        type=str,
        default="gpt-4o",
        help="The underlying AI model to call.",
    )
    parser.add_argument(
        "--model_temperature",
        type=int,
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
        "--cache_mode",
        type=str,
        default=CacheMode.NO_CACHE,
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
        choices=["answer", "ambiguous", "contextual", "unanswerable"],
        help="Space-separated list of gold annotations from [answer, ambiguous, contextual, unanswerable].",
    )

    args = parser.parse_args()

    # Select model class based on argument
    if args.model == "CodeSpotterModel":
        model_class = CodeSpotterModel
    else:
        model_class = DirectSpotterModel

    df, rounds_questions_dict = load_data(
        stages_path=args.stages,
        rounds_path=args.rounds,
        goldAnnotations=args.gold_annotations,
    )

    accuracy, failed = benchmark_on_rounds(
        df=df,
        rounds_question_ids=rounds_questions_dict,
        model=model_class,
        model_string=args.model_string,
        temperature=args.model_temperature,
        max_rounds=args.max_rounds,
        max_questions=args.max_questions,
        use_history=args.use_history,
        cache_mode=args.cache_mode,
        use_captain_board=args.use_captain_board,
    )

    print(f"Benchmark Accuracy: {accuracy * 100:.2f}%")
    if failed:
        print("Failed cache ids and mismatches:")
        for item in failed:
            print(item)
