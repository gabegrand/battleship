import hashlib
import json
import traceback
from abc import ABC
from abc import abstractmethod
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from board import *
from openai import OpenAI
from prompting import *

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
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


class BaseSpotterModel(ABC):
    def __init__(
        self,
        board_id,
        board_experiment,
        use_cache=True,
        model_string="gpt-4o",
        temperature=None,
    ):
        self.board_id = board_id
        self.board_experiment = board_experiment
        self.model_string = model_string
        self.temperature = temperature
        self.use_cache = use_cache
        self.cache_path = CACHE_DIR / self.__class__.__name__
        self.cache_path.mkdir(exist_ok=True)

    def read_cache(self, cache_key):
        cache_file = os.path.join(self.cache_path, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache_dict = json.load(f)
            return cache_dict["result"]
        return None

    def write_cache(self, cache_key, prompt, code, result, traceback):
        cache_file = os.path.join(self.cache_path, f"{cache_key}.json")
        cache_data = {
            "prompt": prompt,
            "code": code,
            "result": result,
            "traceback": traceback,
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=4)

    @abstractmethod
    def _get_model_answer(
        self, question: Question, history: List[dict], occ_tiles
    ) -> Answer:
        raise NotImplementedError

    def answer(
        self, question: Question, history: List[dict] = None, occ_tiles=None
    ) -> Answer:
        if self.use_cache:
            cache_key = question.get_cache_key(self.board_id)
            cached_result = self.read_cache(cache_key)
            if cached_result:
                return Answer(text=cached_result)

        result = self._get_model_answer(question, history, occ_tiles)
        return result


class DirectSpotterModel(BaseSpotterModel):
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
        )

        completion = client.chat.completions.create(
            model=self.model_string,
            messages=prompt.to_chat_format(),
            temperature=self.temperature,
        )

        response = completion.choices[0].message.content
        self.write_cache(
            question.get_cache_key(self.board_id),
            prompt=prompt.to_chat_format(),
            code=None,
            result=response,
        )

        answer = Answer(text=response)

        return answer


class CodeSpotterModel(BaseSpotterModel):
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
        )

        completion = client.chat.completions.create(
            model=self.model_string,
            messages=translation_prompt.to_chat_format(),
            temperature=self.temperature,
        )
        code_generated = completion.choices[0].message.content

        local_vars = {}
        tb = None

        try:
            _ = exec(code_generated, {"np": np}, local_vars)
        except Exception as e:
            tb = "".join(traceback.format_tb(e.__traceback__))
            # code_generated += f"\n# Error: {e}\n{tb}"
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
        self.write_cache(
            question.get_cache_key(self.board_id),
            prompt=code_question.translation_prompt.to_chat_format(),
            code=code_question.fn_str,
            result=result,
            traceback=code_question.traceback,
        )

        return Answer(text=result, code_question=code_question)
