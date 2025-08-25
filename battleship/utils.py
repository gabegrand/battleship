#!/usr/bin/env python3
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ANSWER_STRING_BOOL_MAP = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "(captain timed out)": None,
    "(answer timed out)": None,
    "(no question asked)": None,
    "none": None,
}

ANSWER_BOOL_STRING_MAP = {
    True: "yes",
    False: "no",
    None: "none",
}


def _parse_answer(answer: (str | bool), parse_map: dict) -> bool | str:
    """
    Parse the answer string to a boolean value or vice versa.
    """
    if pd.isnull(answer):
        return None

    if answer in parse_map:
        return parse_map[answer]
    else:
        warnings.warn(f"Unknown answer will be parsed as `null`: {answer}")
        return None


def parse_answer_to_bool(answer: str) -> bool:
    """
    Parse the answer string to a boolean value.
    """
    if isinstance(answer, str):
        answer = answer.lower()
    return _parse_answer(answer, ANSWER_STRING_BOOL_MAP)


def parse_answer_to_str(answer: bool) -> str:
    """
    Parse the answer string to a string value.
    """
    if isinstance(answer, np.bool_):
        answer = bool(answer)
    return _parse_answer(answer, ANSWER_BOOL_STRING_MAP)


# Project root is one level up from this file (battleship/utils.py -> project_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(relative_path):
    """
    Resolve a path relative to the project root.

    Args:
        relative_path: Path relative to project root (str or Path)

    Returns:
        Path: Absolute path resolved from project root
    """
    if os.path.isabs(relative_path):
        return Path(relative_path)

    return PROJECT_ROOT / relative_path
