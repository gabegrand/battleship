import json
import logging
import os
import re
import time
import traceback
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from battleship.board import Board
from battleship.fast_sampler import FastSampler
from battleship.utils import parse_answer_to_str

# Set up logging
logger = logging.getLogger(__name__)

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
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    return client


@dataclass
class ActionData:
    stage_index: int = None  # Corresponds to the stage index of the game
    timestamp: float = None  # When the action was taken

    action: str = None  # "decision", "question", "move", "answer"
    question: "Question" = None  # For question actions
    answer: "Answer" = None  # For question/answer actions
    decision: str = None  # For decision actions
    move: Tuple[int, int] = None  # For move actions

    eig: float = None  # For EIG calculations
    map_prob: float = None  # For MAP calculations
    board_state: np.ndarray = None  # Board state at time of action

    prompt: str = None  # The prompt text
    completion: dict = None  # Full completion object as JSON

    eig_questions: list["ActionData"] = None
    plan: dict = None  # Optional compact plan summary for logging

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        """Convert action data to dictionary format for JSON serialization."""
        return {
            "stage_index": int(self.stage_index)
            if self.stage_index
            else None,  # Convert numpy.int64 to Python int
            "action": self.action,
            "prompt": self.prompt,
            "completion": self.completion,
            "question": self.question.to_dict() if self.question else None,
            "answer": self.answer.to_dict() if self.answer else None,
            "decision": self.decision,
            "move": (
                tuple(int(x) for x in self.move) if self.move else None
            ),  # Convert numpy.int64 to Python int
            "timestamp": (
                float(self.timestamp) if self.timestamp else None
            ),  # Convert numpy.float64 to Python float
            "eig": (
                float(self.eig) if self.eig is not None else None
            ),  # Convert numpy.float64 to Python float
            "map_prob": (
                float(self.map_prob) if self.map_prob is not None else None
            ),  # Convert numpy.float64 to Python float
            "board_state": (
                self.board_state.astype(int).tolist()
                if self.board_state is not None
                else None
            ),  # Convert numpy array to Python list
            "eig_questions": (self.eig_questions),  # List of ActionData objects
            "plan": self.plan,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionData":
        """Create an ActionData instance from a dictionary."""
        return cls(
            stage_index=data["stage_index"],
            action=data["action"],
            prompt=data.get("prompt"),
            completion=data.get("completion"),
            question=(
                Question.from_dict(data["question"]) if data.get("question") else None
            ),
            answer=Answer.from_dict(data["answer"]) if data.get("answer") else None,
            decision=data.get("decision"),
            move=tuple(data["move"]) if data.get("move") else None,
            timestamp=data.get("timestamp"),
            eig=data.get("eig"),
            map_prob=data.get("map_prob"),
            board_state=np.array(data["board_state"])
            if data.get("board_state")
            else None,
            plan=data.get("plan"),
        )


@dataclass
class Question:
    text: str

    def to_dict(self) -> dict:
        return {"text": self.text}

    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        return cls(text=data["text"])


@dataclass
class Answer:
    text: str
    code_question: "CodeQuestion" = None
    _value: Optional[bool] = None

    def __post_init__(self):
        """Parse the answer text once upon creation."""
        if self._value is None:
            self._value = self.parse(self.text)

    @property
    def value(self) -> Optional[bool]:
        """Get the cached parsed boolean value of the answer text."""
        return self._value

    @staticmethod
    def parse(answer: Union[str, bool, None]) -> Optional[bool]:
        """Parse answer text into boolean or None."""
        if isinstance(answer, (bool, np.bool_)):
            return bool(answer)

        if pd.isnull(answer) or answer is None:
            return None

        if not isinstance(answer, str):
            warnings.warn(f"Answer should be a string, got {type(answer)}: {answer}")
            return None

        answer = answer.lower()
        if answer == "true":
            return True
        elif answer == "false":
            return False
        elif answer == "yes":
            return True
        elif answer == "no":
            return False
        elif answer == "(captain timed out)":
            return None
        elif answer == "(answer timed out)":
            return None
        elif answer == "(no question asked)":
            return None
        elif answer == "none":
            return None
        else:
            warnings.warn(f"Unknown answer will be parsed as `null`: {answer}")
            return None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "value": self.value,
            "code_question": self.code_question.to_dict()
            if self.code_question
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Answer":
        # If value is already in the dict, use it to avoid re-parsing
        answer = cls(
            text=data["text"],
            code_question=CodeQuestion.from_dict(data["code_question"])
            if data.get("code_question")
            else None,
            _value=data.get("value"),  # Use cached value if available
        )
        return answer


class CodeQuestion:
    def __init__(
        self,
        question: Question,
        fn_text: str,
        translation_prompt: str,
        completion: dict,
    ):
        if not isinstance(question, Question):
            raise ValueError(
                f"question must be an instance of Question, got {type(question)}"
            )

        self.question = question
        self.fn_str = fn_text
        self.translation_prompt = translation_prompt
        self.completion = completion
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
            fn_return_text = parse_answer_to_str(self.fn(true_board, partial_board))
        except:
            return None

        return Answer(text=fn_return_text, code_question=self)

    def to_dict(self) -> dict:
        return {
            "question": self.question.to_dict(),
            "fn_str": self.fn_str,
            "translation_prompt": str(self.translation_prompt),
            "completion": self.completion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodeQuestion":
        return cls(
            question=data["question"],
            fn_text=data["fn_str"],
            translation_prompt=data["translation_prompt"],
            completion=data["completion"],
        )


class NullCodeQuestion(CodeQuestion):
    def __init__(self, question, translation_prompt):
        self.question = question
        self.fn = lambda true_board, partial_board: None
        self.fn_str = None
        self.translation_prompt = translation_prompt
        self.completion = None


# ---------------------
# Abstract Agent Classes
# ---------------------


class Agent(ABC):
    def __init__(
        self,
        seed: int = None,
        llm: str = None,
        use_cot: bool = False,
        json_path: str = None,
    ):
        self.use_cot = use_cot
        self.rng = np.random.default_rng(seed)
        self.llm = llm
        self.json_path = json_path
        self.stage_index = 0

    def save_action_data(self, action_data: ActionData):
        """Save action data to JSON file."""
        if not self.json_path:
            return

        # Load existing data
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        # Add stage index to action data
        action_data.stage_index = self.stage_index

        # Add new action
        data.append(action_data.to_dict())
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)


def binary_entropy(p: float) -> float:
    """
    Calculate the binary channel entropy given a probability p.
    Returns NaN if p is not in [0, 1].
    """
    if p < 0 or p > 1:
        return float("nan")
    elif p == 0 or p == 1:
        return 0.0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


class EIGCalculator:
    def __init__(
        self,
        seed: int = None,
        timeout: int = 15,
        samples: int = 1000,
        epsilon: float = 0.1,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.timeout = timeout
        self.samples = samples
        self.epsilon = epsilon

    def __call__(
        self,
        code_question: CodeQuestion,
        state: Board,
        ship_tracker: List[Tuple[int, Optional[str]]] = None,
        constraints: list = [],
        weighted_boards: list = None,
    ) -> float:
        weighted_board_answers = self._evaluate_code_question(
            code_question, state, ship_tracker, constraints, weighted_boards
        )

        # Collect weighted results for EIG calculation
        weighted_results = {True: 0.0, False: 0.0}

        for board, weight, answer in weighted_board_answers:
            if answer is None or answer.value is None:
                logger.warning(f"CodeQuestion returned None - skipping EIG calculation")
                return float("nan")
            elif answer.value is True:
                weighted_results[True] += weight
            elif answer.value is False:
                weighted_results[False] += weight
            else:
                logger.warning(
                    f"CodeQuestion returned unknown value `{answer.value}` - skipping EIG calculation"
                )
                return float("nan")

        # Check if we have valid results
        if any(v == 0 for v in weighted_results.values()):
            return 0

        # Calculate EIG using weighted probabilities
        total_weight = sum(weighted_results.values())
        p_true = weighted_results[True] / total_weight

        return binary_entropy(
            self.epsilon + ((1 - 2 * self.epsilon) * p_true)
        ) - binary_entropy(self.epsilon)

    def _evaluate_code_question(
        self,
        code_question: CodeQuestion,
        state: Board,
        ship_tracker: List[Tuple[int, Optional[str]]] = None,
        constraints: list = [],
        weighted_boards: list = None,
    ) -> List[Tuple[Board, float, bool]]:
        sampler = FastSampler(
            board=state,
            ship_tracker=ship_tracker,
            seed=self.rng,
        )

        # Use shared weighted sampling method for both conditional and unconditional cases
        # When no constraints, get_weighted_samples returns uniform weights (1.0 for each board)
        if not weighted_boards:
            weighted_boards = sampler.get_weighted_samples(
                n_samples=self.samples, constraints=constraints, epsilon=self.epsilon
            )

        weighted_board_answers = []
        for board, weight in weighted_boards:
            # Evaluate main question on weighted board
            answer: Answer = code_question(
                true_board=board.board, partial_board=state.board
            )
            weighted_board_answers.append((board, weight, answer))

        return weighted_board_answers


def config_move_regex(size):
    """Generate a regex pattern for move validation based on board size."""
    max_letter = chr(ord("A") + size - 1)
    max_number = str(size)
    return f"[A-{max_letter}]{{1}}[1-{max_number}]{{1}}"
