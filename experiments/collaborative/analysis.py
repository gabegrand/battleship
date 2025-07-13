import json
import os
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd

from battleship.agents import Answer
from battleship.board import Board


MODEL_DISPLAY_NAMES = {
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-opus-4": "claude-opus-4",
    "deepseek/deepseek-chat-v3-0324": "deepseek-chat-v3",
    "deepseek/deepseek-r1-0528": "deepseek-r1",
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct": "llama-3.1-405b-instruct",
    "meta-llama/llama-4-maverick": "llama-4-maverick",
    "meta-llama/llama-4-scout": "llama-4-scout",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4.1-nano": "gpt-4.1-nano",
    "openai/gpt-4.1-mini": "gpt-4.1-mini",
    "openai/gpt-4.1": "gpt-4.1",
    "openai/o3": "o3",
    "openai/o4-mini": "o4-mini",
}

GOLD_ANSWER_LABEL = "gold_answer_text"

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

    gold_labels = df["gold_answer"].apply(Answer.parse).tolist()
    human_labels = df["messageText"].apply(Answer.parse).tolist()

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


def get_spotter_type_short(spotter_type: str, cot: bool) -> str:
    """Get short spotter type label for categorization."""
    if spotter_type == "DirectSpotterModel" and not cot:
        return "Base"
    elif spotter_type == "DirectSpotterModel" and cot:
        return "+ CoT"
    elif spotter_type == "CodeSpotterModel" and not cot:
        return "+ Code"
    elif spotter_type == "CodeSpotterModel" and cot:
        return "+ CoT + Code"
    else:
        raise ValueError(f"Unknown spotter type combination: {spotter_type}, {cot}")
