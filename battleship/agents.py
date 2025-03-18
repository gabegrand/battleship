import json
import os
import re
import traceback
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import random
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from openai import OpenAI

from battleship.board import Board
from battleship.board import coords_to_tile
from battleship.board import tile_to_coords
from battleship.fast_sampler import FastSampler
from battleship.game import Decision
from battleship.prompting import MovePrompt
from battleship.prompting import QuestionPrompt
from battleship.prompting import SpotterPrompt

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

MOVE_PATTERN = re.compile("^[A-H]{1}[1-8]{1}$")
MOVE_COT_PATTERN = re.compile(r"\s*<answer>\s*([A-H][1-8])\s*</answer>\s*")
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


class Agent(ABC):
    # Class variable for global counter
    action_counter = 0

    @classmethod
    def increment_counter(cls):
        cls.action_counter += 1
        return cls.action_counter

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
        return f"{Agent.action_counter}_{decision_type}"

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


class Captain(Agent):
    def __init__(
        self,
        seed: int = None,
        cache_mode: CacheMode = CacheMode.NO_CACHE,
        model_string=None,
        temperature=None,
    ):
        super().__init__(seed=seed, model_string=model_string, cache_mode=cache_mode)
        self.temperature = temperature

    def decision(
        self,
        state: Board,
        history: List[Dict],
        questions_remaining: int,
        moves_remaining: int,
    ):
        cached_result = self.read_cache("DECISION")

        # If we have a cache hit in READ_WRITE mode
        if cached_result == "GENERATE_NEW":
            # Generate new response anyway
            result = self._get_decision(
                state, history, questions_remaining, moves_remaining
            )
            # Cache the new result
            self.write_cache(
                "DECISION",
                prompt={
                    "type": "decision",
                    "state": str(state.board.tobytes()),
                    "questions_remaining": questions_remaining,
                    "moves_remaining": moves_remaining,
                },
                code=None,
                result=str(result.value),
            )

            # Increment counter after decision
            Agent.increment_counter()
            return result

        # Normal cache hit
        elif cached_result is not None:
            # Increment counter after decision
            Agent.increment_counter()
            return Decision(cached_result)

        # Cache miss
        result = self._get_decision(
            state, history, questions_remaining, moves_remaining
        )

        # Cache the result
        self.write_cache(
            "DECISION",
            prompt={
                "type": "decision",
                "state": str(state.board.tobytes()),
                "questions_remaining": questions_remaining,
                "moves_remaining": moves_remaining,
            },
            code=None,
            result=str(result.value),
        )

        # Increment counter after decision
        Agent.increment_counter()
        return result

    def move(self, state: Board, history: List[Dict]):
        cached_result = self.read_cache("MOVE")

        # If we have a cache hit in READ_WRITE mode
        if cached_result == "GENERATE_NEW":
            # Generate new anyway
            result = self._get_move(state, history)
            # Cache the result
            # self.write_cache(
            #    "MOVE",
            #    prompt={"type": "move", "state": str(state.board.tobytes())},
            #    code=None,
            #    result=str(result),
            # )

            # Increment counter after move
            Agent.increment_counter()
            return result

        # Normal cache hit
        elif cached_result is not None:
            # Convert cached string representation back to tuple
            coords = tuple(map(int, cached_result.strip("()").split(", ")))
            # Increment counter after move
            Agent.increment_counter()
            return coords

        # Cache miss
        result = self._get_move(state, history)

        # Increment counter after move
        Agent.increment_counter()
        return result

    def question(self, state: Board, history: List[Dict]):
        cached_result = self.read_cache("QUESTION")

        # If we have a cache hit in READ_WRITE mode
        if cached_result == "GENERATE_NEW":
            # Generate new anyway
            result = self._get_question(state, history)
            # Cache the result
            # self.write_cache(
            #    "QUESTION",
            #    prompt={"type": "question", "state": str(state.board.tobytes())},
            #    code=None,
            #    result=result.text,
            # )

            # Increment counter after question
            Agent.increment_counter()
            return result

        # Normal cache hit
        elif cached_result is not None:
            # Increment counter after question
            Agent.increment_counter()
            return Question(text=cached_result)

        # Cache miss
        result = self._get_question(state, history)

        # Increment counter after question
        Agent.increment_counter()
        return result

    def _get_decision(self, state, history, questions_remaining, moves_remaining):
        raise NotImplementedError

    def _get_move(self, state, history):
        raise NotImplementedError

    def _get_question(self, state, history):
        raise NotImplementedError


class Spotter(Agent):
    def __init__(
        self,
        board_id,
        board_experiment,
        cache_mode=CacheMode.NO_CACHE,
        model_string="gpt-4o",
        temperature=None,
        use_cot=False,
    ):
        self.board_id = board_id
        self.board_experiment = board_experiment
        self.temperature = temperature

        # Use proper Agent initialization to handle model string and cache path
        super().__init__(
            seed=None, model_string=model_string, cache_mode=cache_mode, use_cot=use_cot
        )

    @abstractmethod
    def _get_model_answer(
        self, question: Question, history: List[dict], occ_tiles
    ) -> Answer:
        raise NotImplementedError

    def answer(
        self, question: Question, history: List[dict] = None, occ_tiles=None
    ) -> Answer:
        cached_result = self.read_cache("ANSWER")

        # If we have a cache hit in READ_WRITE mode
        if cached_result == "GENERATE_NEW":
            # Generate new response anyway
            result = self._get_model_answer(question, history, occ_tiles)

            # Increment counter after answer
            Agent.increment_counter()
            return result

        # Normal cache hit
        elif cached_result is not None:
            # Increment counter after answer
            Agent.increment_counter()
            return Answer(text=cached_result)

        # Cache miss
        result = self._get_model_answer(question, history, occ_tiles)

        # Increment counter after answer
        Agent.increment_counter()
        return result


class DirectSpotterModel(Spotter):
    def _get_model_answer(
        self, question: Question, history: List[dict], occ_tiles=None
    ) -> Answer:
        prompt = SpotterPrompt(
            target_trial_id=self.board_id,
            target_trial_experiment=self.board_experiment,
            target_occ_tiles=occ_tiles,
            board_format="grid",
            question=question,
            use_code=False,
            include_final_prefix=False,
            history=history,
            use_cot=self.use_cot,
        )

        completion = client.chat.completions.create(
            model=self.model_string,
            messages=prompt.to_chat_format(),
            temperature=self.temperature,
        )

        if not self.use_cot:
            response = completion.choices[0].message.content
            self.write_cache(
                question.get_cache_key(self.board_id),
                prompt=prompt.to_chat_format(),
                code=None,
                result=response,
            )
        else:
            response = None
            while response is None:
                completion = client.chat.completions.create(
                    model=self.model_string,
                    messages=prompt.to_chat_format(),
                    temperature=self.temperature,
                )
                response_match = BOOL_ANSWER_PATTERN.search(
                    completion.choices[0].message.content
                )
                if response_match:
                    response = response_match.group(
                        1
                    )  # This extracts just the "Yes" or "No"
                else:
                    response = None

            self.write_cache(
                question.get_cache_key(self.board_id),
                prompt=prompt.to_chat_format(),
                code=completion.choices[0].message.content,
                result=response,
            )

        answer = Answer(text=response)
        return answer


class CodeSpotterModel(Spotter):
    def translate(
        self, question: Question, history: List[dict], occ_tiles=None
    ) -> CodeQuestion:
        translation_prompt = SpotterPrompt(
            target_trial_id=self.board_id,
            target_trial_experiment=self.board_experiment,
            target_occ_tiles=occ_tiles,
            board_format="grid",
            question=question,
            use_code=True,
            include_final_prefix=False,
            history=history,
            use_cot=self.use_cot,
        )

        if not self.use_cot:
            completion = client.chat.completions.create(
                model=self.model_string,
                messages=translation_prompt.to_chat_format(),
                temperature=self.temperature,
            )
            code_generated = completion.choices[0].message.content
        else:
            code_generated = None
            while code_generated is None:
                completion = client.chat.completions.create(
                    model=self.model_string,
                    messages=translation_prompt.to_chat_format(),
                    temperature=self.temperature,
                )
                response_match = CODE_ANSWER_PATTERN.search(
                    completion.choices[0].message.content
                )
                if response_match:
                    code_generated = response_match.group(
                        1
                    )  # This extracts just the "Yes" or "No"
                else:
                    code_generated = None

        local_vars = {}
        tb = None

        if "python\n" in code_generated:
            code_generated = code_generated.split("python\n")[1]

        try:
            _ = exec(code_generated, {"np": np}, local_vars)
        except Exception as e:
            tb = "".join(traceback.format_tb(e.__traceback__))
            local_vars["answer"] = None

        return CodeQuestion(
            question=question,
            fn=local_vars["answer"],
            fn_string=code_generated,
            translation_prompt=translation_prompt,
            traceback=tb,
        )

    def _get_model_answer(
        self, question: Question, history: List[dict], occ_tiles=None
    ) -> Answer:
        code_question = self.translate(question, history, occ_tiles)

        board = Board.from_trial_id(
            trial_id=self.board_id, experiment=self.board_experiment
        ).to_symbolic_array()

        result = code_question(board)

        # Use our global counter caching system
        self.write_cache(
            "ANSWER",
            prompt=code_question.translation_prompt.to_chat_format(),
            code=code_question.fn_str,
            result=result,
            traceback=code_question.traceback,
        )

        return Answer(text=result, code_question=code_question)


class RandomCaptain(Captain):
    def decision(self, *args, **kwargs):
        return Decision.MOVE

    def move(self, state: Board, history: List[Dict]) -> Tuple[int, int]:
        hidden_tiles = np.argwhere(state.board == Board.hidden)
        if len(hidden_tiles) == 0:
            raise ValueError("No hidden tiles left.")
        coords = self.rng.choice(hidden_tiles)
        return tuple(coords)


class MAPCaptain(Captain):
    def __init__(self, seed: int = None, n_samples: int = 10000):
        super().__init__(seed)
        self.n_samples = n_samples

    def decision(self, *args, **kwargs):
        return Decision.MOVE

    def move(self, state: Board, history: List[Dict]) -> Tuple[int, int]:
        sampler = FastSampler(
            board=state,
            ship_lengths=Board.SHIP_LENGTHS,
            ship_labels=Board.SHIP_LABELS,
            seed=self.rng,
        )

        # Compute the raw posterior counts over board positions.
        posterior = sampler.compute_posterior(n_samples=self.n_samples, normalize=False)

        # For tiles that have already been revealed, force their probability to -infinity.
        posterior = posterior.astype(float)
        posterior[state.board != Board.hidden] = -np.inf

        # Select the tile with the maximum posterior probability (MAP estimate).
        flat_idx = int(np.argmax(posterior))

        # Map the flat index back to 2D coordinates.
        move_coords = np.unravel_index(flat_idx, state.board.shape)
        return tuple(move_coords)


class ProbabilisticCaptain(Captain):
    def __init__(
        self,
        seed: int = None,
        questions_remaining=15,
        q_prob: float = 0.5,
        use_cot=False,
        model_string: str = "openai/gpt-4o",
        temperature: float = None,
        cache_mode: CacheMode = CacheMode.NO_CACHE,
    ):
        super().__init__(
            seed=seed,
            model_string=model_string,
            temperature=temperature,
            cache_mode=cache_mode,
        )
        self.use_cot = use_cot
        self.q_prob = q_prob
        self.questions_remaining = questions_remaining

    def get_decision_cache_key(self, state_hash):
        """Generate a cache key for decisions"""
        return f"decision_{state_hash}"

    def get_move_cache_key(self, state_hash):
        """Generate a cache key for move operations"""
        return f"move_{state_hash}"

    def get_question_cache_key(self, state_hash):
        """Generate a cache key for question operations"""
        return f"question_{state_hash}"

    def _get_decision(self, state, history, questions_remaining, moves_remaining):
        if random() < self.q_prob and questions_remaining > 0:
            return Decision.QUESTION
        return Decision.MOVE

    def _get_move(self, state: Board, history: List[Dict]) -> Tuple[int, int]:
        visible_tiles = list(zip(*np.where(state.board != Board.hidden)))

        move_prompt = MovePrompt(
            target_occ_tiles=state,
            board_format="grid",
            history=history,
            use_cot=self.use_cot,
        )

        # Format the prompt for caching
        prompt_for_cache = move_prompt.to_chat_format()

        candidate_move = None
        while candidate_move is None or candidate_move in visible_tiles:
            completion = client.chat.completions.create(
                model=self.model_string,
                messages=move_prompt.to_chat_format(),
                temperature=self.temperature,
            )
            if self.use_cot:
                match = MOVE_COT_PATTERN.search(completion.choices[0].message.content)
                if match is not None:
                    candidate_move = tile_to_coords(match.group(1))
            else:
                candidate_move = MOVE_PATTERN.match(
                    completion.choices[0].message.content
                )
                if candidate_move is not None:
                    candidate_move = tile_to_coords(candidate_move.group())

        self.write_cache(
            "MOVE",
            prompt_for_cache,
            None,
            candidate_move,
            traceback=completion.choices[0].message.content,
        )

        print("prob move", coords_to_tile(candidate_move))
        return candidate_move

    def _get_question(self, state: Board, history: List[Dict]) -> Question:
        question_prompt = QuestionPrompt(
            target_occ_tiles=state,
            board_format="grid",
            history=history,
            use_cot=self.use_cot,
        )

        # Format the prompt for caching
        prompt_for_cache = question_prompt.to_chat_format()

        if not self.use_cot:
            completion = client.chat.completions.create(
                model=self.model_string,
                messages=question_prompt.to_chat_format(),
                temperature=self.temperature,
            )
            candidate_question = completion.choices[0].message.content
        else:
            candidate_question = None
            while candidate_question is None:
                completion = client.chat.completions.create(
                    model=self.model_string,
                    messages=question_prompt.to_chat_format(),
                    temperature=self.temperature,
                )
                candidate_question = ANSWER_MATCH_PATTERN.search(
                    completion.choices[0].message.content
                )

                if candidate_question:
                    candidate_question = candidate_question.group(
                        1
                    )  # This extracts just the "Yes" or "No"
                else:
                    candidate_question = None

        result = Question(text=candidate_question)

        print("prob question", candidate_question)
        self.write_cache(
            "QUESTION",
            str(prompt_for_cache),
            None,
            candidate_question,
            traceback=completion.choices[0].message.content,
        )

        return result
