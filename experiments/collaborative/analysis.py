import colorsys
import json
import os
from itertools import combinations
from itertools import product
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

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

# NOTE: This maps both CoT and non-CoT captain types to the same label
CAPTAIN_TYPE_LABELS = {
    "human": "Human",
    "RandomCaptain": "Random",
    "MAPCaptain": "MAP",
    "LLMDecisionCaptain": "LLM",
    "LLMDecisionCaptain_cot": "LLM",
    "EIGCaptain": "EIG",
    "EIGCaptain_cot": "EIG",
    "MAPEIGCaptain": "MAP + EIG",
    "MAPEIGCaptain_cot": "MAP + EIG",
}


def load_dataset(
    experiment_path: str,
    use_gold: bool = True,
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

    # Compute per-round contiguous turn_index based on questionID.
    # Note: questionID values are non-unique across rows within a round (multiple
    # actions can share the same questionID) and may be non-contiguous (e.g., 1,3,5,...)
    # We map the ORDER OF FIRST APPEARANCE of each distinct questionID within a round
    # to 0,1,2,... so that all rows sharing a questionID share the same turn_index.
    def _factorize_within_round(s: pd.Series) -> np.ndarray:
        # Factorize preserves first-seen order; missing values => code -1.
        codes, _ = pd.factorize(s, sort=False)
        # Convert to float so we can set missing codes to NaN.
        codes = codes.astype(float)
        codes[codes < 0] = np.nan  # optional: keep NaN for missing questionID
        return codes

    # Use transform so the returned array aligns exactly with the original index.
    df["turn_index"] = df.groupby("roundID")["questionID"].transform(
        _factorize_within_round
    )

    # stage_completion: proportion of the turn sequence completed (0 at first turn, 1 at last turn)
    turn_max = df.groupby("roundID")["turn_index"].transform("max")
    df["stage_completion"] = np.where(
        turn_max.isna() | (turn_max == 0), 0.0, df["turn_index"] / turn_max
    )

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


# ---------------------------------------------------------------------------
# Win rate computation & visualization helpers (extracted from notebook)
# ---------------------------------------------------------------------------


def build_competitor_column(
    df: pd.DataFrame,
    llm_col: str = "llm_display",
    captain_col: str = "captain_type_display",
    out_col: str = "competitor",
    sep: str = " | ",
) -> pd.DataFrame:
    """Construct a categorical competitor column (LLM primary, then captain).

    The order is derived by iterating over the llm categorical order (if present)
    and, within each llm, the captain categorical order, **including only**
    observed combinations.

    Returns the mutated DataFrame (copy) with a categorical `out_col`.
    """
    df = df.copy()
    if llm_col not in df.columns or captain_col not in df.columns:
        raise KeyError("Required columns missing to build competitor column.")

    # Determine ordered categories
    if pd.api.types.is_categorical_dtype(df[llm_col]):
        llm_categories = list(df[llm_col].cat.categories)
    else:
        llm_categories = sorted(df[llm_col].dropna().unique())

    if pd.api.types.is_categorical_dtype(df[captain_col]):
        cap_categories = list(df[captain_col].cat.categories)
    else:
        cap_categories = sorted(df[captain_col].dropna().unique())

    competitor_order = []
    for llm in llm_categories:
        present_caps = [
            c
            for c in cap_categories
            if ((df[llm_col] == llm) & (df[captain_col] == c)).any()
        ]
        for cap in present_caps:
            competitor_order.append(f"{llm}{sep}{cap}")

    df[out_col] = (df[llm_col].astype(str) + sep + df[captain_col].astype(str)).astype(
        str
    )
    df[out_col] = pd.Categorical(df[out_col], categories=competitor_order, ordered=True)
    return df


def compute_board_level_win_rate(a_vals, b_vals, higher_is_better=True):
    """Compute win rate of A over B for all pairwise comparisons of metric values.

    Ties count as 0.5 wins. Returns (win_rate, wins, comparisons).
    """
    wins = 0.0
    comparisons = 0
    for a, b in product(a_vals, b_vals):
        if pd.isna(a) or pd.isna(b):
            continue
        comparisons += 1
        if higher_is_better:
            if a > b:
                wins += 1
            elif a == b:
                wins += 0.5
        else:
            if a < b:
                wins += 1
            elif a == b:
                wins += 0.5
    if comparisons == 0:
        return np.nan, wins, comparisons
    return wins / comparisons, wins, comparisons


def compute_pairwise_win_rates(
    df: pd.DataFrame,
    metric: str = "f1_score",
    higher_is_better: bool = True,
    competitor_col: str = "competitor",
    board_col: str = "board_id",
) -> dict:
    """Compute pairwise win rates between all competitors.

    For each pair (A,B) and each board, use the cross product of their per-round
    metric values to compute a board-level win rate (ties = 0.5). Two aggregate
    summaries per pair:
      * mean_board_win_rate: mean of board-level win rates (unweighted).
      * weighted_all_pairs_win_rate: total wins / total comparisons (pooled).

    Returns a dict with detailed & aggregate DataFrames and symmetric matrices.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in DataFrame.")
    if competitor_col not in df.columns:
        raise ValueError(f"Competitor column '{competitor_col}' not in DataFrame.")
    if board_col not in df.columns:
        raise ValueError(f"Board column '{board_col}' not in DataFrame.")

    competitors = [c for c in df[competitor_col].dropna().unique()]
    boards = sorted(df[board_col].dropna().unique())

    grouped = df.groupby([competitor_col, board_col])
    records = []

    for ca, cb in combinations(competitors, 2):
        board_results = []
        total_wins = 0.0
        total_comparisons = 0
        for board in boards:
            try:
                a_group = grouped.get_group((ca, board))
            except KeyError:
                a_group = pd.DataFrame(columns=df.columns)
            try:
                b_group = grouped.get_group((cb, board))
            except KeyError:
                b_group = pd.DataFrame(columns=df.columns)

            a_vals = a_group[metric].dropna().values
            b_vals = b_group[metric].dropna().values
            if len(a_vals) == 0 or len(b_vals) == 0:
                continue
            board_wr, wins, comps = compute_board_level_win_rate(
                a_vals, b_vals, higher_is_better=higher_is_better
            )
            if not np.isnan(board_wr):
                board_results.append((board, board_wr, wins, comps))
                total_wins += wins
                total_comparisons += comps

        if len(board_results) == 0:
            mean_board_win_rate = np.nan
            weighted_all_pairs_win_rate = np.nan
        else:
            mean_board_win_rate = np.nanmean([br for _, br, _, _ in board_results])
            weighted_all_pairs_win_rate = (
                total_wins / total_comparisons if total_comparisons > 0 else np.nan
            )

        for board, br, wins, comps in board_results:
            records.append(
                {
                    "competitor_a": ca,
                    "competitor_b": cb,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "board_id": board,
                    "board_win_rate": br,
                    "board_wins": wins,
                    "board_comparisons": comps,
                }
            )
        records.append(
            {
                "competitor_a": ca,
                "competitor_b": cb,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "board_id": None,
                "board_win_rate": mean_board_win_rate,
                "board_wins": total_wins,
                "board_comparisons": total_comparisons,
                "weighted_all_pairs_win_rate": weighted_all_pairs_win_rate,
                "boards_considered": len(board_results),
            }
        )

    detailed_df = pd.DataFrame(records)
    aggregate_df = detailed_df[detailed_df["board_id"].isna()].copy()
    aggregate_df = aggregate_df.assign(
        mean_board_win_rate=aggregate_df["board_win_rate"],
        weighted_all_pairs_win_rate=aggregate_df["weighted_all_pairs_win_rate"].fillna(
            aggregate_df["board_wins"]
            / aggregate_df["board_comparisons"].replace({0: np.nan})
        ),
    )

    # Maintain original categorical ordering if present
    if pd.api.types.is_categorical_dtype(df[competitor_col]):
        competitors_sorted = [
            c for c in df[competitor_col].cat.categories if c in competitors
        ]
    else:
        competitors_sorted = sorted(competitors)

    mean_matrix = pd.DataFrame(
        np.nan, index=competitors_sorted, columns=competitors_sorted
    )
    weighted_matrix = pd.DataFrame(
        np.nan, index=competitors_sorted, columns=competitors_sorted
    )

    for _, row in aggregate_df.iterrows():
        ca, cb = row["competitor_a"], row["competitor_b"]
        mean_ab = row["mean_board_win_rate"]
        weighted_ab = row["weighted_all_pairs_win_rate"]
        if ca in mean_matrix.index and cb in mean_matrix.columns:
            mean_matrix.loc[ca, cb] = mean_ab
            weighted_matrix.loc[ca, cb] = weighted_ab
            if not pd.isna(mean_ab):
                mean_matrix.loc[cb, ca] = 1 - mean_ab
            if not pd.isna(weighted_ab):
                weighted_matrix.loc[cb, ca] = 1 - weighted_ab
            mean_matrix.loc[ca, ca] = 0.5
            weighted_matrix.loc[ca, ca] = 0.5

    # Ensure all diagonals filled
    for c in competitors_sorted:
        mean_matrix.loc[c, c] = 0.5
        weighted_matrix.loc[c, c] = 0.5

    return {
        "detailed": detailed_df,
        "aggregate": aggregate_df,
        "mean_board_win_rate_matrix": mean_matrix,
        "weighted_win_rate_matrix": weighted_matrix,
    }


def _lighten(hex_color: str, factor: float = 0.9) -> str:
    try:
        hex_color = hex_color.lstrip("#")
        r, g, b = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
        h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        l = 1 - (1 - l) * factor
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        return f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
    except Exception:
        return "#f0f0f0"


def plot_grouped_winrate_heatmap(
    win_matrix: pd.DataFrame,
    llm_palette: dict,
    cmap: str | None = "cividis",
    annotate: bool = True,
    annotate_fmt: str = ".2f",
    annot_min: int = 6,
    annot_max: int = 12,
    captain_tick_fontsize: int = 6,
    row_alpha: float = 0.30,
    col_alpha: float = 0.18,
    show_group_separators: bool = True,
    separator_width: float = 4,
    shade_rows: bool = True,
    shade_cols: bool = True,
    group_label_rotation: int = 90,
    group_label_fontsize: int = 10,
    group_label_offset: float = 1.5,
    output_path: str | None = None,
    title: str | None = None,
    dpi: int = 300,
):
    """Plot a grouped win-rate heatmap given a competitor-indexed matrix.

    Assumes competitor labels are of the form "<LLM> | <Captain>".
    Returns (fig, ax).
    """
    mat = win_matrix.copy().astype(float)
    competitors = list(mat.index)

    def split_comp(c):
        parts = c.split("|", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return c.strip(), ""

    competitor_llm = {c: split_comp(c)[0] for c in competitors}
    competitor_captain = {c: split_comp(c)[1] for c in competitors}

    # Build contiguous blocks
    blocks = []
    current_llm, current_block = None, []
    for c in competitors:
        llm = competitor_llm[c]
        if llm != current_llm:
            if current_block:
                blocks.append((current_llm, current_block))
            current_llm = llm
            current_block = [c]
        else:
            current_block.append(c)
    if current_block:
        blocks.append((current_llm, current_block))

    new_order = [c for _, comps in blocks for c in comps]
    mat = mat.loc[new_order, new_order]

    # Color shades per LLM
    row_shade_colors = {
        llm: _lighten(llm_palette.get(llm, "#888888"), 0.93) for llm, _ in blocks
    }

    n = len(mat)
    figsize_base = 0.50
    fig, ax = plt.subplots(
        figsize=(min(24, 1.4 + n * figsize_base), min(24, 1.4 + n * figsize_base))
    )
    annot_font_size = max(annot_min, min(annot_max, 120 / max(n, 1)))

    hm = sns.heatmap(
        mat,
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        annot=annotate,
        fmt=annotate_fmt,
        annot_kws={"size": annot_font_size} if annotate else None,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.6, "pad": 0.02},
        ax=ax,
    )

    captain_labels = [competitor_captain[c] for c in new_order]
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(
        captain_labels,
        rotation=90,
        ha="center",
        va="top",
        fontsize=captain_tick_fontsize,
    )
    ax.set_yticklabels(
        captain_labels,
        rotation=0,
        ha="right",
        va="center",
        fontsize=captain_tick_fontsize,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Shading
    if shade_rows:
        for llm, comps in blocks:
            start = new_order.index(comps[0])
            end = new_order.index(comps[-1])
            ax.add_patch(
                Rectangle(
                    (0, start),
                    width=n,
                    height=end - start + 1,
                    facecolor=row_shade_colors[llm],
                    edgecolor="none",
                    alpha=row_alpha,
                    zorder=0,
                )
            )
    if shade_cols:
        for llm, comps in blocks:
            start = new_order.index(comps[0])
            end = new_order.index(comps[-1])
            ax.add_patch(
                Rectangle(
                    (start, 0),
                    width=end - start + 1,
                    height=n,
                    facecolor=row_shade_colors[llm],
                    edgecolor="none",
                    alpha=col_alpha,
                    zorder=0,
                )
            )

    # Separators
    if show_group_separators:
        cum = 0
        for llm, comps in blocks:
            cum += len(comps)
            if cum < n:
                ax.axhline(cum, color="white", linewidth=separator_width)
                ax.axvline(cum, color="white", linewidth=separator_width)

    # Group labels
    cum_start = 0
    for llm, comps in blocks:
        size = len(comps)
        midpoint = cum_start + size / 2
        ax.text(
            midpoint,
            n + group_label_offset,
            llm,
            ha="center",
            va="top",
            rotation=0,
            fontsize=group_label_fontsize,
            fontweight="bold",
            color=llm_palette.get(llm, "#222222"),
            clip_on=False,
        )
        ax.text(
            -group_label_offset,
            midpoint,
            llm,
            ha="right",
            va="center",
            rotation=group_label_rotation,
            fontsize=group_label_fontsize,
            fontweight="bold",
            color=llm_palette.get(llm, "#222222"),
            clip_on=False,
        )
        cum_start += size

    cbar = hm.collections[0].colorbar
    cbar.ax.set_title("Win rate\n(row > col)", fontsize=9, pad=6, loc="left")
    cbar.ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=14, pad=12)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig, ax
