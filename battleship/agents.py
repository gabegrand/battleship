import json
import os
import re
import traceback
import uuid
from abc import ABC
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from battleship.board import Board
from battleship.fast_sampler import FastSampler
from battleship.utils import parse_answer_to_str

# MOVE_PATTERN = lambda size: re.compile(f"^{config_move_regex(size)}$")
DECISION_PATTERN = re.compile(
    r"\s*<answer>\s*(Question|Move)\s*(<answer>|</answer>)\s*"
)
MOVE_PATTERN = lambda size: re.compile(
    rf"\s*<answer>\s*({config_move_regex(size)})\s*(<answer>|</answer>)\s*"
)
BOOL_ANSWER_PATTERN = re.compile(r"\s*<answer>\s*(Yes|No)\s*(<answer>|</answer>)\s*")
ANSWER_MATCH_PATTERN = re.compile(r"\s*<answer>\s*(.*?)\s*(<answer>|</answer>)\s*")
CODE_ANSWER_PATTERN = re.compile("```python(.*?)```", re.DOTALL)


def get_openai_client():
    load_dotenv()
    return OpenAI()


@dataclass
class Question:
    text: str


@dataclass
class Prompt:
    prompt: str = None
    full_completion: str = None
    extracted_completion: str = None
    eig: float = None
    map_prob: float = None
    occ_tiles: np.ndarray = None


class CodeQuestion:
    def __init__(
        self,
        question: str,
        fn_text: str,
        translation_prompt,
        full_completion,
    ):
        self.question = question
        self.fn_str = fn_text
        self.translation_prompt = translation_prompt
        self.full_completion = full_completion

        self.__evaluate_fn_text()

    def __evaluate_fn_text(self):
        local_namespace = {}

        def _restricted_input(*args):
            raise RuntimeError("input() function is not allowed in generated code")

        try:
            exec(self.fn_str, {"np": np, "input": _restricted_input}, local_namespace)
        except Exception as e:
            raise RuntimeError(
                f"Error evaluating the provided code: {e}\n{traceback.format_exc()}"
            )

        if "answer" not in local_namespace:
            raise RuntimeError(
                "The answer function is not defined in the provided code."
            )

        self.fn = local_namespace["answer"]

    def __call__(self, true_board: np.ndarray, partial_board: np.ndarray) -> bool:
        try:
            return self.fn(true_board, partial_board)
        except:
            return None


class NullCodeQuestion(CodeQuestion):
    def __init__(self, translation_prompt):
        self.question = None
        self.fn = lambda true_board, partial_board: None
        self.fn_str = None
        self.translation_prompt = translation_prompt
        self.full_completion = None


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
        decision_counter: Counter = None,
        index_counter: Counter = None,
        round_id: str = None,
    ):
        self.round_id = round_id
        self.use_cot = use_cot
        self.rng = np.random.default_rng(seed)
        self.model_string = model_string
        self.decision_counter = decision_counter
        self.index_counter = index_counter
        self.stage_list = []
        self.prompt_list = []


class EIGCalculator:
    def __init__(self, seed, spotter):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.spotter = spotter

    def calculate_eig(self, question, state, pregenerated_question=None, samples=100):
        if pregenerated_question is None:
            code_question = self.spotter.translate(
                question=question, occ_tiles=state.board, history=None
            )
        else:
            code_question = pregenerated_question

        sampler = FastSampler(
            board=state,
            ship_lengths=Board.SHIP_LENGTHS,
            ship_labels=Board.SHIP_LABELS,
            seed=self.rng,
        )

        results = {"yes": 0, "no": 0}
        curr_time = time()
        while sum(results.values()) < samples:
            if time() - curr_time > 15:
                return float("nan")
            board = None
            while not board:
                board = sampler.populate_board()
            result = code_question(true_board=board.board, partial_board=state.board)
            try:
                result = parse_answer_to_str(result)
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
