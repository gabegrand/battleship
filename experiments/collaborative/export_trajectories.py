"""Utility script to export representative trajectories for Battleship runs."""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from datetime import UTC
from pathlib import Path
from typing import Iterable
from typing import Sequence

from battleship.board import Board
from battleship.utils import resolve_project_path


DEFAULT_RUN_DIR = "experiments/collaborative/captain_benchmarks/run_2025_08_25_22_02_29"
DEFAULT_CONTEXTS_DIR = "experiments/collaborative/contexts"
DEFAULT_OUTPUT_PATH = "battleship.github.io/static/data/_tmp_trajectory_samples.json"


@dataclass(frozen=True)
class ExportEntry:
    round_id: str
    captain_type: str
    data: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export representative trajectories from a captain benchmark run"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=DEFAULT_RUN_DIR,
        help="Path to the benchmark run directory (default: %(default)s)",
    )
    parser.add_argument(
        "--contexts-dir",
        type=str,
        default=DEFAULT_CONTEXTS_DIR,
        help=(
            "Path to the contexts directory used to infer the experiment name. "
            "Must live under the project experiments/ tree."
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Optional experiment identifier to override the value inferred from "
            "--contexts-dir when loading boards via Board.from_trial_id."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON path for the exported trajectories (default: %(default)s)",
    )
    parser.add_argument(
        "--include-ties",
        action="store_true",
        help="Keep multiple rounds when F1 scores tie for extremes (default: drop duplicates).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def load_summary(run_dir: Path) -> list[dict]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file at {summary_path}")
    logging.debug("Reading summary from %s", summary_path)
    return json.loads(summary_path.read_text())


def normalize_event(event: dict) -> dict:
    decision = event.get("decision")
    normalized: dict[str, object] = {
        "stage": event.get("stage"),
        "decision": decision,
        "board": event.get("state"),
    }

    coords = event.get("coords")
    if coords is not None:
        normalized["move"] = {
            "coords": coords,
            "tile": coords_to_tile(coords),
        }

    question = event.get("question")
    if question:
        normalized["question"] = {"text": question.get("text")}

    answer = event.get("answer")
    if answer:
        normalized["answer"] = {
            "text": answer.get("text"),
            "value": answer.get("value"),
        }

    return normalized


def coords_to_tile(coords: Sequence[int] | None) -> str | None:
    if coords is None:
        return None
    row, col = coords
    return f"{chr(ord('A') + row)}{col + 1}"


def select_extreme_entries(
    summary: Iterable[dict], include_ties: bool = False
) -> list[ExportEntry]:
    by_captain: dict[str, list[dict]] = defaultdict(list)
    for record in summary:
        by_captain[record["captain_type"]].append(record)

    selected: list[ExportEntry] = []
    for captain_type, entries in sorted(by_captain.items()):
        if not entries:
            continue
        sorted_entries = sorted(entries, key=lambda e: e["f1_score"])
        lowest = sorted_entries[0]
        highest = sorted_entries[-1]

        selected.append(
            ExportEntry(
                round_id=str(lowest["round_id"]),
                captain_type=captain_type,
                data={"extreme": "lowest", **lowest},
            )
        )

        if include_ties:
            tied_high = [
                entry
                for entry in sorted_entries[::-1]
                if entry["f1_score"] == highest["f1_score"]
            ]
        else:
            tied_high = [highest]

        for entry in tied_high:
            if str(entry["round_id"]) == str(lowest["round_id"]):
                continue
            selected.append(
                ExportEntry(
                    round_id=str(entry["round_id"]),
                    captain_type=captain_type,
                    data={"extreme": "highest", **entry},
                )
            )

    unique_rounds: dict[str, ExportEntry] = {}
    for entry in selected:
        if entry.round_id not in unique_rounds:
            unique_rounds[entry.round_id] = entry

    return list(unique_rounds.values())


def infer_experiment_from_contexts(contexts_dir: Path) -> str:
    experiments_root = resolve_project_path("experiments").resolve()
    try:
        relative = contexts_dir.resolve().relative_to(experiments_root)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            "--contexts-dir must live under the experiments/ directory"
        ) from exc

    parts = relative.parts
    if len(parts) < 2 or parts[-1] != "contexts":
        raise ValueError(
            "--contexts-dir should point to .../experiments/<experiment>/contexts"
        )

    experiment = Path(*parts[:-1]).as_posix()
    if not experiment:
        raise ValueError("Unable to infer experiment name from contexts directory")

    return experiment


def parse_trial_id(board_id: str) -> int:
    match = re.search(r"(\d+)$", str(board_id))
    if not match:
        raise ValueError(f"Unable to infer numeric trial id from board_id={board_id}")
    return int(match.group(1))


def load_true_board(trial_id: int, experiment: str) -> list[list[int]]:
    board = Board.from_trial_id(trial_id, experiment=experiment)
    return board.to_numpy().astype(int).tolist()


def build_games(
    entries: Iterable[ExportEntry],
    run_dir: Path,
    experiment: str,
) -> list[dict]:
    games: list[dict] = []
    for entry in entries:
        round_dir = run_dir / "rounds" / f"round_{entry.round_id}"
        game_path = round_dir / "game.json"
        if not game_path.exists():
            logging.warning("Skipping round %s (missing %s)", entry.round_id, game_path)
            continue

        raw_events = json.loads(game_path.read_text())
        events = [normalize_event(evt) for evt in raw_events]
        question_count = sum(1 for evt in events if evt.get("decision") == "question")
        move_count = sum(1 for evt in events if evt.get("decision") == "move")

        try:
            trial_id = parse_trial_id(entry.data["board_id"])
            true_board = load_true_board(trial_id, experiment)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(
                "Failed to load board %s for round %s: %s",
                entry.data.get("board_id"),
                entry.round_id,
                exc,
            )
            continue

        games.append(
            {
                "round_id": entry.round_id,
                "captain_type": entry.captain_type,
                "spotter_type": entry.data.get("spotter_type"),
                "board_id": entry.data.get("board_id"),
                "seed": entry.data.get("seed"),
                "f1_score": entry.data.get("f1_score"),
                "hits": entry.data.get("hits"),
                "misses": entry.data.get("misses"),
                "question_count": question_count,
                "move_count": move_count,
                "is_won": entry.data.get("is_won"),
                "precision": entry.data.get("precision"),
                "recall": entry.data.get("recall"),
                "true_board": true_board,
                "events": events,
                "extreme": entry.data.get("extreme"),
            }
        )

    return games


def export_data(games: list[dict], run_dir: Path, output_path: Path) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_run": run_dir.name,
        "game_count": len(games),
        "games": games,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logging.info("Wrote %s games to %s", len(games), output_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    run_dir = resolve_project_path(args.run_dir)
    contexts_dir = resolve_project_path(args.contexts_dir)
    output_path = resolve_project_path(args.output_path)

    experiment = args.experiment or infer_experiment_from_contexts(contexts_dir)
    logging.debug(
        "Resolved experiment '%s' from contexts dir %s", experiment, contexts_dir
    )

    summary = load_summary(run_dir)
    logging.info("Loaded %s summary entries", len(summary))

    entries = select_extreme_entries(summary, include_ties=args.include_ties)
    logging.info("Selected %s unique rounds for export", len(entries))

    games = build_games(entries, run_dir, experiment)
    logging.info("Constructed %s games with trajectories", len(games))

    export_data(games, run_dir, output_path)


if __name__ == "__main__":
    main()
