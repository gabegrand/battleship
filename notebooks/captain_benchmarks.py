import sys

sys.path.insert(0, "..")

import numpy as np
from io import StringIO
import pandas as pd
import os

from battleship.agents import (
    RandomCaptain,
    MAPCaptain,
    ProbabilisticCaptain,
    CodeSpotterModel,
    LLMDecisionCaptain,
    EIGAutoCaptain,
)
from battleship.board import Board
from battleship.game import BattleshipGame
from battleship.agents import CacheMode
from multiprocessing.dummy import Pool

stage_df = pd.read_csv("/home/ubuntu/repo_battleship/temp/gold_annotations_partial.csv")
round_df = pd.read_csv(
    "/home/ubuntu/repo_battleship/battleship/experiments/collaborative/battleship-final-data/round.csv"
)
goldAnnotations = ["answer", "ambiguous", "contextual", "unanswerable"]

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
df = filtered_stage_df.merge(board_ids, left_on="roundID", right_on="id", how="inner")


def game_completed(hits, misses, occTiles, board_id):
    def mask(board_array):
        return (board_array != -1) & (board_array != 0)

    if hits + misses > 40:
        return False
    else:
        return np.all(
            mask(occTiles)
            == mask(
                Board.convert_to_numeric(
                    Board.from_trial_id(board_id).to_symbolic_array()
                )
            )
        )


question_counts_df = (
    df[df["messageType"] == "question"]
    .groupby("roundID")
    .size()
    .reset_index(name="question_number")
)

df = df.merge(question_counts_df, on="roundID", how="inner")
result = df.loc[df.groupby("roundID")["index"].idxmax()][
    ["roundID", "occTiles", "board_id", "question_number"]
]
result = result[
    result["occTiles"] != str(np.full((8, 8), -1).tolist()).replace(" ", "")
]  # ugly!
data = []
for roundID, occTiles, board_id in zip(
    result["roundID"], result["occTiles"], result["board_id"]
):
    occTiles = np.array(eval(occTiles))
    misses = np.sum(occTiles == 0)
    hits = np.sum((occTiles != -1) & (occTiles != 0))
    data.append(
        {
            "captainType": "human",
            "boardId": board_id,
            "hits": hits,
            "misses": misses,
            "gameCompleted": game_completed(hits, misses, occTiles, board_id),
            "questionsAsked": result[result["roundID"] == roundID][
                "question_number"
            ].values[0],
        }
    )

human_results_df = pd.DataFrame(data)

board_ids = ["B" + str(i).zfill(2) for i in range(1, 19)]

eig_spotter = CodeSpotterModel(
    board_id="B01",
    board_experiment="collaborative",
    model_string="gpt-4o",
    temperature=None,
    use_cot=True,
)

captains = {
    "RandomCaptain": RandomCaptain(seed=42),
    "MAPCaptain": MAPCaptain(seed=42, n_samples=100),
    "ProbabilisticCaptain": ProbabilisticCaptain(
        seed=42,
        model_string="gpt-4o",
        q_prob=0.7,
        use_cot=False,
        cache_mode=CacheMode.WRITE_ONLY,
    ),
    "EIGAutoCaptain": EIGAutoCaptain(
        seed=42,
        samples=1000,
        model_string="gpt-4o",
        spotter=eig_spotter,
        use_cot=False,
        k=10,
        cache_mode=CacheMode.WRITE_ONLY,
    ),
    "ProbabilisticCaptain_cot": ProbabilisticCaptain(
        seed=42,
        model_string="gpt-4o",
        q_prob=0.7,
        use_cot=True,
        cache_mode=CacheMode.WRITE_ONLY,
    ),
    "EIGAutoCaptain_cot": EIGAutoCaptain(
        seed=42,
        samples=1000,
        model_string="gpt-4o",
        spotter=eig_spotter,
        use_cot=True,
        k=10,
        cache_mode=CacheMode.WRITE_ONLY,
    ),
}

captains = {
    "EIGAutoCaptain_cot": EIGAutoCaptain(
        seed=42,
        samples=1000,
        model_string="gpt-4o",
        spotter=eig_spotter,
        use_cot=True,
        k=10,
        cache_mode=CacheMode.WRITE_ONLY,
    )
}

seeds = range(1, 7 + 1)

seeds = [1, 2]
board_ids = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B13",
    "B14",
    "B15",
    "B16",
    "B17",
    "B18",
]

# Define output file
output_file = "/home/ubuntu/repo_battleship/temp/total_results.csv"

# First write the human results to initialize the file
human_results_df.to_csv(output_file, index=False)


def run_single_agent_game(args):
    cap_name, captain, seed, board_id = args
    if cap_name == "EIGAutoCaptain":
        eig_spotter.board_id = board_id
    captain.seed = seed

    print(f"{cap_name} started with {board_id} & seed {seed}")
    board = Board.from_trial_id(board_id)
    game = BattleshipGame(
        board_target=board,
        max_questions=15,
        max_moves=40,
        captain=captain,
        spotter=CodeSpotterModel(
            board_id,
            "collaborative",
            cache_mode=CacheMode.WRITE_ONLY,
            model_string="gpt-4o",
            temperature=None,
            use_cot=True,
        ),
    )
    game.play()
    print(f"{cap_name} finished with {board_id} & seed {seed}")

    result = {
        "captainType": cap_name,
        "boardId": board_id,
        "hits": game.hits,
        "misses": game.misses,
        "gameCompleted": game.is_won(),
        "questionsAsked": game.question_count,
    }

    # Calculate precision
    result["precision"] = (
        result["hits"] / (result["hits"] + result["misses"])
        if (result["hits"] + result["misses"]) > 0
        else 0
    )

    # Append result to CSV
    result_df = pd.DataFrame([result])
    result_df.to_csv(output_file, mode="a", header=False, index=False)

    return result


# Prepare a list of tasks (each tuple corresponds to one game run)
jobs = [
    (cap_name, captain, seed, board_id)
    for cap_name, captain in captains.items()
    for seed in seeds
    for board_id in board_ids
]

# Run with multiprocessing
with Pool() as pool:
    print(f"Running with {pool._processes} processes")
    results = pool.map(run_single_agent_game, jobs)

results_df = pd.read_csv(output_file)
print(
    f"Completed {len(results_df) - len(human_results_df)} agent games out of {len(jobs)} jobs"
)
