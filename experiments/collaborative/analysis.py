import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from battleship.board import Board


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
