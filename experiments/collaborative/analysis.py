import json
import os
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd

from battleship.agents import Answer
from battleship.board import Board
from battleship.game import BattleshipGame


MODEL_DISPLAY_NAMES = {
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-opus-4": "claude-opus-4",
    "deepseek/deepseek-chat-v3-0324": "deepseek-chat-v3",
    "deepseek/deepseek-r1-0528": "deepseek-r1",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
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
    "openai/gpt-5": "gpt-5",
}

GOLD_ANSWER_LABEL = "gold_answer"

GOLD_CATEGORY_LABELS = {
    "gold_discourse": "Discourse",
    "gold_stateful": "Stateful",
    "gold_vague": "Vague",
    "gold_ambiguous": "Ambiguous",
    "gold_unanswerable": "Unanswerable",
}

CAPTAIN_TYPE_LABELS = {
    "RandomCaptain": "Random",
    "MAPCaptain": "MAP",
    "LLMDecisionCaptain": "LLM",
    "LLMDecisionCaptain_cot": "LLM (CoT)",
    "EIGCaptain": "EIG",
    "EIGCaptain_cot": "EIG (CoT)",
    "MAPEIGCaptain": "MAP + EIG",
    "MAPEIGCaptain_cot": "MAP + EIG (CoT)",
    "human": "Human",
}


def load_dataset(
    experiment_path: str,
    use_gold: bool = False,
    drop_incomplete: bool = False,
    filter_exceeded_max_moves: bool = True,
) -> pd.DataFrame:
    """Load and normalize collaborative Battleship experiment data at the event level.

    This function is the single source of truth for assembling the analysis
    dataset. It:
    - Reads `stage.csv` (or `gold-v2/gold-v2.csv` when `use_gold=True`),
      `round.csv`, and `player.csv` from `experiment_path`.
    - Renames identifiers, drops timestamp churn columns, and joins tables.
    - Optionally drops games that did not complete (`drop_incomplete`).
    - Keeps only messages of interest: "move", "question", "answer", "decision".
    - Parses JSON fields like `occTiles` and `trueTiles` into Python lists.
    - Computes per-row metrics by scoring each partial board against the true board
      via `Board.score` (hits, misses, precision, recall, f1_score, is_won, etc.).
    - Adds convenience columns such as `pairID` and `hits_pct`.
    - Optionally filters rows where hits + misses exceed `BattleshipGame.MAX_MOVES`.

    Parameters
    ----------
    experiment_path : str
        Path to a single experiment directory containing `stage.csv` (or
        `gold-v2/gold-v2.csv`), `round.csv`, and `player.csv`.
    use_gold : bool, optional
        If True, uses `gold-v2/gold-v2.csv` instead of `stage.csv` for stage data.
    drop_incomplete : bool, optional
        If True, removes games that did not end normally (e.g., timed out).
    filter_exceeded_max_moves : bool, optional
        If True, drops rows where hits + misses exceed `BattleshipGame.MAX_MOVES`.

    Returns
    -------
    pandas.DataFrame
        Event-level dataframe sorted by `pairID` and `roundID`, including
        per-row board metrics and `questionsRemaining` from `round.csv`.

    Notes
    -----
    - This function is intentionally generic and model-agnostic. Higher-level
      result builders (e.g., human summaries) should build on this to avoid
      duplicating IO, parsing, or scoring logic.
    """
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
    ROUND_COLUMNS = ["roundID"] + ["board_id", "trueTiles", "questionsRemaining"]
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

    df["hits_pct"] = df["hits"] / df["total_ship_tiles"]

    # Filter out rows where hits + misses exceed the maximum allowed moves (optional)
    exceeded_max_moves = df["hits"] + df["misses"] > BattleshipGame.MAX_MOVES
    if exceeded_max_moves.any():
        if filter_exceeded_max_moves:
            print(
                f"Warning: Dropped {exceeded_max_moves.sum()} rows where hits + misses exceeded {BattleshipGame.MAX_MOVES}."
            )
            df = df[~exceeded_max_moves]
        else:
            print(
                f"Warning: Found {exceeded_max_moves.sum()} rows where hits + misses exceeded {BattleshipGame.MAX_MOVES} (not dropped)."
            )

    return df


def human_round_summaries(
    experiment_path: str,
    use_gold: bool = True,
    drop_incomplete: bool = False,
    filter_exceeded_max_moves: bool = True,
    max_questions: int = BattleshipGame.MAX_QUESTIONS,
):
    """Summarize human performance per round using the normalized dataset.

    This function builds on `load_dataset` and produces one record per round
    representing the final observed state for human play. It selects the last
    stage entry per `roundID` (by `index`), filters rounds where the board is
    still completely unknown, derives `question_count` from
    `max_questions - questionsRemaining`, and returns a compact list of
    dictionaries suitable for plotting or tabular reporting.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory containing CSVs.
    use_gold : bool, optional
        Passed through to `load_dataset` to choose gold annotations.
    drop_incomplete : bool, optional
        Passed through to `load_dataset` to drop incomplete games.
    filter_exceeded_max_moves : bool, optional
        Passed through to `load_dataset` to drop rows exceeding max moves.
    max_questions : int, optional
        Total question budget per round used to compute `question_count`.

    Returns
    -------
    list[dict]
        One dict per round with keys: `captain_type`, `spotter_type`,
        `round_id`, `board_id`, `hits`, `misses`, `precision`, `recall`,
        `f1_score`, `is_won`, and `question_count`.

    Notes
    -----
    - Use this when you need human per-round summaries. For lower-level or
      mixed analyses, prefer calling `load_dataset` directly.
    """
    # Center around load_dataset to unify parsing, merging and scoring
    df = load_dataset(
        experiment_path=experiment_path,
        use_gold=use_gold,
        drop_incomplete=drop_incomplete,
        filter_exceeded_max_moves=filter_exceeded_max_moves,
    )

    # Select the final stage entry per round (max index)
    last_indices = df.groupby("roundID")["index"].idxmax()
    result = df.loc[last_indices].copy()

    # Filter out rows where occTiles are entirely unknown (-1 everywhere)
    def is_all_unknown(tiles):
        arr = np.asarray(tiles)
        return arr.size > 0 and (arr == -1).all()

    result = result[~result["occTiles"].apply(is_all_unknown)]

    data = []
    for _, row in result.iterrows():
        questions_remaining = (
            int(row["questionsRemaining"])
            if not pd.isnull(row["questionsRemaining"])
            else 0
        )
        questions_asked = max_questions - questions_remaining

        result_row = {
            "captain_type": "human",
            "spotter_type": "human",
            "round_id": row["roundID"],
            "board_id": row["board_id"],
            "hits": int(row["hits"]),
            "misses": int(row["misses"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1_score": float(row["f1_score"]),
            "is_won": bool(row["is_won"]),
            "question_count": questions_asked,
        }
        data.append(result_row)

    return data


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
        return "CoT"
    elif spotter_type == "CodeSpotterModel" and not cot:
        return "Code"
    elif spotter_type == "CodeSpotterModel" and cot:
        return "CoT + Code"
    else:
        raise ValueError(f"Unknown spotter type combination: {spotter_type}, {cot}")
