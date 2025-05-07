import csv
import os
import re
from abc import ABC
from dataclasses import dataclass
from math import log2
from pathlib import Path
from time import time

import numpy as np
from openai import OpenAI

from battleship.board import Board
from battleship.fast_sampler import FastSampler

CACHE_DIR = Path(f"./cache")
CACHE_DIR.mkdir(exist_ok=True)

CSV_STAGE_FILE = CACHE_DIR / f"stage.csv"
CSV_ROUND_FILE = CACHE_DIR / f"round.csv"
CSV_PROMPTS_FILE = CACHE_DIR / f"prompts.csv"
SUMMARY_FILE = CACHE_DIR / f"summary.csv"


MOVE_PATTERN = lambda size: re.compile(f"^{config_move_regex(size)}$")
DECISION_PATTERN = re.compile("^(Question|Move)$")
MOVE_COT_PATTERN = lambda size: re.compile(
    rf"\s*<answer>\s*({config_move_regex(size)})\s*</answer>\s*"
)
BOOL_ANSWER_PATTERN = re.compile(r"\s*<answer>\s*(Yes|No)\s*</answer>\s*")
ANSWER_MATCH_PATTERN = re.compile(r"\s*<answer>\s*(.*?)\s*</answer>\s*")
CODE_ANSWER_PATTERN = re.compile("```python(.*?)```", re.DOTALL)

client = OpenAI()


@dataclass
class Question:
    text: str

    def get_cache_key(self, board_id):
        return f"{self.text.lower().replace(' ','_').replace('/','_')}_{board_id}"


@dataclass
class Prompt:
    prompt: str = None
    full_completion: str = None
    extracted_completion: str = None
    eig: float = None
    map_prob: float = None
    occ_tiles: np.ndarray = None


@dataclass
class CacheData:
    message_text: str = None
    eig: float = None
    map_prob: float = None
    occ_tiles: np.ndarray = None
    prompts: list[Prompt] = None


class CodeQuestion:
    def __init__(
        self,
        question,
        fn,
        fn_string,
        translation_prompt,
        full_completion,
        traceback=None,
    ):
        self.question = question
        self.fn = fn
        self.fn_str = fn_string
        self.translation_prompt = translation_prompt
        self.traceback = traceback
        self.full_completion = full_completion

    def __call__(self, board):
        try:
            result = self.fn(board)
            return result
        except:
            return None


@dataclass
class Answer:
    text: str
    code_question: CodeQuestion = None


# ---------------------
# Abstract Agent Classes
# ---------------------


class Counter:
    def __init__(self):
        self.counter = 0

    def increment_counter(self):
        self.counter += 1
        return self.counter


class Agent(ABC):
    # Class variable for global counter
    action_counter = 0

    def increment_counter(self):
        self.action_counter += 1
        return self.action_counter

    def __init__(
        self,
        seed: int = None,
        model_string: str = None,
        use_cot: bool = False,
        use_cache: bool = False,
        decision_counter: Counter = None,
        index_counter: Counter = None,
        round_id: str = None,
    ):
        self.round_id = round_id
        self.use_cot = use_cot
        self.rng = np.random.default_rng(seed)
        self.model_string = model_string
        self.use_cache = use_cache
        self.decision_counter = decision_counter
        self.index_counter = index_counter

    def write_cache(self, message_type: str = None, cache_data: CacheData = None):
        """Append a new entry to the CSV cache."""

        def option_to_str(option):
            return option if option is not None else ""

        occ_tiles_str = (
            np.array2string(cache_data.occ_tiles)
            if cache_data.occ_tiles is not None
            else ""
        )
        exists = os.path.isfile(self.csv_stage_file)
        with open(self.csv_stage_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "round_id",
                    "index",
                    "messageType",
                    "messageText",
                    "mapProb",
                    "eig",
                    "occTiles",
                    "question_id",
                    "modelBackend",  # Add the model backend
                ],
            )
            if not exists:
                writer.writeheader()
            stage_id = self.index_counter.increment_counter()
            writer.writerow(
                {
                    "round_id": self.round_id,
                    "index": stage_id,
                    "messageType": message_type,
                    "messageText": option_to_str(cache_data.message_text),
                    "mapProb": option_to_str(cache_data.map_prob),
                    "eig": option_to_str(cache_data.eig),
                    "occTiles": option_to_str(occ_tiles_str),
                    "question_id": self.decision_counter.counter,
                    "modelBackend": self.model_string,  # Include model backend
                }
            )
            if cache_data.prompts is not None:
                with open(self.csv_prompts_file, "a", newline="") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=[
                            "round_id",
                            "stage_index",
                            "prompt_index",
                            "prompt",
                            "full_completion",
                            "extracted_completion",
                            "eig",
                            "map_prob",
                            "occ_tiles",
                            "modelBackend",  # Add the model backend
                        ],
                    )
                    if not exists:
                        writer.writeheader()
                    prompt_counter = Counter()
                    for prompt in cache_data.prompts:
                        writer.writerow(
                            {
                                "round_id": self.round_id,
                                "stage_index": stage_id,
                                "prompt_index": prompt_counter.increment_counter(),
                                "prompt": prompt.prompt,
                                "full_completion": prompt.full_completion,
                                "extracted_completion": prompt.extracted_completion,
                                "eig": prompt.eig,
                                "map_prob": prompt.map_prob,
                                "occ_tiles": prompt.occ_tiles,
                                "modelBackend": self.model_string,  # Include model backend
                            }
                        )


class EIGCalculator:
    def __init__(self, seed, spotter):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.spotter = spotter

    def calculate_eig(self, question, state, pregenerated_question=None, samples=100):
        if pregenerated_question is None:
            code_question = self.spotter.translate(question, [], state.board)
        else:
            code_question = pregenerated_question

        sampler = FastSampler(
            board=state,
            ship_lengths=Board.SHIP_LENGTHS,
            ship_labels=Board.SHIP_LABELS,
            seed=self.rng,
        )

        results = {"Yes": 0, "No": 0}
        curr_time = time()
        while sum(results.values()) < samples:
            if time() - curr_time > 15:
                # print("EIG calculation timed out")
                return float("nan")
            board = None
            while not board:
                board = sampler.populate_board()
            board = board.to_symbolic_array()
            result = code_question(board)
            if type(result) == str:
                try:
                    results[result] += 1
                except:
                    break

        if any(v == 0 for v in results.values()):
            return 0
        else:
            return np.log2(samples) - sum(
                [p / samples * np.log2(p) for p in results.values()]
            )


def config_move_regex(size):
    """Generate a regex pattern for move validation based on board size."""
    # max_letter/number are required so black can format the return statement properly
    max_letter = chr(ord("A") + size - 1)
    max_number = str(size)
    return f"[A-{max_letter}]{{1}}[1-{max_number}]{{1}}"
