import json
import os
import warnings
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from battleship.board import Board


MODEL_DISPLAY_NAMES = {
    "claude-3-5-haiku-latest": "claude-3.5-haiku",
    "claude-3-7-sonnet-latest": "claude-3.7-sonnet",
    "deepseek/deepseek-chat-v3-0324": "deepseek-chat-v3",
    "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct": "llama-3.1-405b-instruct",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "meta-llama/llama-4-maverick": "llama-4-maverick",
    "meta-llama/llama-4-scout": "llama-4-scout",
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o",
    "o3": "o3",
    "o4-mini": "o4-mini",
}

GOLD_ANSWER_LABEL = "gold_answer"

GOLD_CATEGORY_LABELS = {
    "gold_discourse": "Discourse",
    "gold_stateful": "Stateful",
    "gold_vague": "Vague",
    "gold_ambiguous": "Ambiguous",
    "gold_unanswerable": "Unanswerable",
}


def load_dataset(
    experiment_path: str, use_gold: bool = False, drop_incomplete: bool = False
) -> pd.DataFrame:
    PATH_PLAYER = os.path.join(experiment_path, "player.csv")
    PATH_ROUND = os.path.join(experiment_path, "round.csv")
    PATH_STAGE = os.path.join(
        experiment_path, "gold-v2/gold-v2.csv" if use_gold else "stage.csv"
    )

    df_stage = pd.read_csv(PATH_STAGE)
    df_round = pd.read_csv(PATH_ROUND)
    df_player = pd.read_csv(PATH_PLAYER)

    # rename id to stageID
    df_stage = df_stage.rename(mapper={"id": "stageID"}, axis=1)
    df_round = df_round.rename(mapper={"id": "roundID"}, axis=1)

    # drop all columns that end with LastChangedAt
    df_stage = df_stage.loc[:, ~df_stage.columns.str.endswith("LastChangedAt")]
    df_round = df_round.loc[:, ~df_round.columns.str.endswith("LastChangedAt")]

    df_ended = df_player[["gameID", "ended", "timeoutGameEnd"]].drop_duplicates()
    df_ended["gameCompleted"] = (df_ended["ended"] == "game ended") & (
        df_ended["timeoutGameEnd"] == False
    )

    if not (df_ended["gameCompleted"]).all():
        print("WARNING: Some games were not completed.")
        print(df_ended.loc[~df_ended["gameCompleted"]])
        if drop_incomplete:
            print("These will be dropped from the dataset.")
        else:
            print("These will be kept in the dataset.")

    # Merge stage, round, and player dataframes
    ROUND_COLUMNS = ["roundID"] + ["board_id", "trueTiles"]
    df = df_stage.merge(df_round[ROUND_COLUMNS], on="roundID")

    # drop all rows where game was not completed
    df = df.merge(df_ended, on="gameID")
    if drop_incomplete:
        df = df[df["gameCompleted"]]

    # drop all rows where messageType is not in (fire, question, answer, decision)
    df = df[df["messageType"].isin(["move", "question", "answer", "decision"])]

    # Convert occTiles and trueTiles to numpy arrays
    df["occTiles"] = df["occTiles"].apply(json.loads)
    df["trueTiles"] = df["trueTiles"].apply(json.loads)

    # Convert board_id to int
    # df["board_id"] = df["board_id"].astype(int)

    # Map each gameID to a unique pairID (pair_01, pair_02, ...)
    df["pairID"] = df["gameID"].map(
        {
            gameID: f"pair_{i:02}"
            for i, gameID in enumerate(sorted(df["gameID"].unique()))
        }
    )

    # Sort by pairID and roundID
    df = df.sort_values(by=["pairID", "roundID"])

    # Use Board.score to calculate metrics for each row
    def calculate_metrics(row):
        true_board = Board(np.array(row["trueTiles"]))
        partial_board = Board(np.array(row["occTiles"]))

        # Calculate scores using the Board class
        scores = true_board.score(partial_board)

        # Return scores with exact keys from Board.score
        return pd.Series(scores)

    # Apply the calculation to each row and merge the results
    scores_df = df.apply(calculate_metrics, axis=1)
    df = pd.concat([df, scores_df], axis=1)

    return df


def get_gold_answer_dataset(df_gold) -> Tuple[List[bool], List[bool]]:
    df = df_gold.copy()
    df = df[~pd.isnull(df["gold_answer"])]

    gold_labels = df["gold_answer"].apply(parse_answer).tolist()
    human_labels = df["messageText"].apply(parse_answer).tolist()

    assert all(label in [True, False] for label in gold_labels)
    assert all(label in [True, False, None] for label in human_labels)

    dropped_count = sum(label is None for label in human_labels)
    print(f"Warning: Dropped {dropped_count} instances where human label is None.")
    valid_idxs = [i for i, label in enumerate(human_labels) if label is not None]
    gold_labels = [gold_labels[i] for i in valid_idxs]
    human_labels = [human_labels[i] for i in valid_idxs]
    assert all(
        label in [True, False] for label in gold_labels
    ), "Gold labels should be True or False"
    assert all(
        label in [True, False] for label in human_labels
    ), "Human labels should be True or False"

    assert len(gold_labels) == len(
        human_labels
    ), "Mismatch in lengths of gold and human labels"

    return gold_labels, human_labels


def load_spotter_results(path: str):
    """Load all JSON files in a directory and concatenate them into a single DataFrame."""
    # List of JSON file paths
    json_paths = sorted(
        [
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if filename.endswith(".json")
        ]
    )

    # Concatenate DataFrames from all JSON files
    df = pd.concat([pd.read_json(path) for path in json_paths], ignore_index=True)

    df["answer"] = df["answer"].map(parse_answer)
    df["true_answer"] = df["true_answer"].map(parse_answer)
    df["is_correct"] = df["answer"] == df["true_answer"]

    def _get_spotter_type_short(spotter_type, cot):
        if spotter_type == "DirectSpotterModel" and not cot:
            return "Base"
        elif spotter_type == "DirectSpotterModel" and cot:
            return "+ CoT"
        elif spotter_type == "CodeSpotterModel" and not cot:
            return "+ Code"
        elif spotter_type == "CodeSpotterModel" and cot:
            return "+ CoT + Code"
        else:
            raise ValueError((spotter_type, cot))

    spotter_type_short_order = [
        "Base",
        "+ CoT",
        "+ Code",
        "+ CoT + Code",
    ]
    df["spotter_type_short"] = pd.Categorical(
        df[["spotterModel", "CoT"]].apply(
            lambda x: _get_spotter_type_short(x["spotterModel"], x["CoT"]), axis=1
        ),
        categories=spotter_type_short_order,
        ordered=True,
    )

    df["model_display_name"] = pd.Categorical(
        df["model"].map(MODEL_DISPLAY_NAMES),
        categories=list(MODEL_DISPLAY_NAMES.values()),
        ordered=True,
    )

    df = df.sort_values(by=["model_display_name", "spotter_type_short"])

    return df


def parse_answer(answer: str) -> bool:
    if isinstance(answer, bool):
        return answer

    if pd.isnull(answer):
        return None

    assert isinstance(answer, str), f"Answer should be a string, got {type(answer)}"
    answer = answer.lower()
    if answer == "true":
        return True
    elif answer == "false":
        return False
    elif answer == "yes":
        return True
    elif answer == "no":
        return False
    elif answer == "(captain timed out)":
        return None
    elif answer == "(answer timed out)":
        return None
    elif answer == "(no question asked)":
        return None
    elif answer == "none":
        return None
    else:
        warnings.warn(f"Unknown answer will be parsed as `null`: {answer}")
        return None
