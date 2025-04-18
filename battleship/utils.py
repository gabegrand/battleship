import json
import os
import re
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from math import log2
from pathlib import Path
from time import time

import numpy as np
from openai import OpenAI

from battleship.board import Board
from battleship.fast_sampler import FastSampler

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

MOVE_PATTERN = lambda size: re.compile(f"^{config_move_regex(size)}$")
DECISION_PATTERN = re.compile("^(Question|Move)$")
MOVE_COT_PATTERN = lambda size: re.compile(
    rf"\s*<answer>\s*({config_move_regex(size)})\s*</answer>\s*"
)
BOOL_ANSWER_PATTERN = re.compile(r"\s*<answer>\s*(Yes|No)\s*</answer>\s*")
ANSWER_MATCH_PATTERN = re.compile(r"\s*<answer>\s*(.*?)\s*</answer>\s*")
CODE_ANSWER_PATTERN = re.compile("```(.*?)```", re.DOTALL)

client = OpenAI()


@dataclass
class Question:
    text: str

    def get_cache_key(self, board_id):
        return f"{self.text.lower().replace(' ','_').replace('/','_')}_{board_id}"


class CodeQuestion:
    def __init__(self, question, fn, fn_string, translation_prompt, traceback=None):
        self.question = question
        self.fn = fn
        self.fn_str = fn_string
        self.translation_prompt = translation_prompt
        self.traceback = traceback

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


class CacheMode(Enum):
    NO_CACHE = "no_cache"  # Don't use cache at all
    READ_ONLY = "read_only"  # Only read from cache, don't write
    WRITE_ONLY = "write_only"  # Only write to cache, don't read
    READ_WRITE = "read_write"  # Read from cache and generate+cache new responses


# ---------------------
# Abstract Agent Classes
# ---------------------


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
        cache_mode: CacheMode = CacheMode.NO_CACHE,
    ):
        self.use_cot = use_cot
        self.rng = np.random.default_rng(seed)
        self.model_string = model_string
        self.cache_mode = cache_mode

        folder_name = self.__class__.__name__
        if model_string is not None:
            # Extract model name after the provider (e.g. "openai/gpt-4o" -> "gpt-4o")
            if "/" in model_string:
                folder_name = model_string.split("/")[1]
            else:
                folder_name = model_string

        self.cache_path = CACHE_DIR / self.__class__.__name__ / folder_name
        os.makedirs(self.cache_path, exist_ok=True)

    def get_cache_key(self, decision_type):
        """Generate a cache key based on the global counter and decision type"""
        return f"{Agent.action_counter}_{decision_type}_{str(time())}"

    def read_cache(self, decision_type):
        """Read from cache based on cache mode using the counter-based key"""
        # Don't read if NO_CACHE or WRITE_ONLY
        if self.cache_mode in [CacheMode.NO_CACHE, CacheMode.WRITE_ONLY]:
            return None

        cache_key = self.get_cache_key(decision_type)
        cache_file = os.path.join(self.cache_path, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache_dict = json.load(f)

            # For READ_WRITE, mark that we had a cache hit but still want to generate new
            if self.cache_mode == CacheMode.READ_WRITE:
                # We've read it, but we'll still generate a new response
                return "GENERATE_NEW"

            # For READ_ONLY, return the cached result
            return cache_dict["result"]
        return None

    def write_cache(self, decision_type, prompt, code, result, traceback=None):
        """Write to cache based on cache mode using the counter-based key"""
        # Don't write if NO_CACHE or READ_ONLY
        if self.cache_mode in [CacheMode.NO_CACHE, CacheMode.READ_ONLY]:
            return None

        cache_key = self.get_cache_key(decision_type)
        cache_file = os.path.join(self.cache_path, f"{cache_key}.json")
        cache_data = {
            "counter": Agent.action_counter,
            "decision_type": decision_type,
            "prompt": prompt,
            "code": str(code),
            "result": str(result),
            "traceback": traceback,
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=4)


class EIGCalculator:
    def __init__(self, seed, spotter):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.spotter = spotter

    def calculate_eig(self, question, state, samples=100, move=False):
        def safe_log2(x):
            if x == 0:
                return 0
            return log2(x)

        code_question = self.spotter.translate(question, [], state.board)
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
                print("EIG calculation timed out")
                return 0
            board = None
            while not board:
                board = sampler.populate_board()
            board = board.to_symbolic_array()
            result = code_question(board)
            if result:
                results[result] += 1
        print("eig results calculated")

        if results == {"Yes": samples, "No": 0} and move:
            return 1000

        eig = safe_log2(samples) - sum(
            [p / samples * safe_log2(p) for p in results.values()]
        )
        # print("eig calculated as", eig, results)
        return eig


def config_move_regex(size):
    """Generate a regex pattern for move validation based on board size."""
    # max_letter/number are required so black can format the return statement properly
    max_letter = chr(ord("A") + size - 1)
    max_number = str(size)
    return f"[A-{max_letter}]{{1}}[1-{max_number}]{{1}}"
