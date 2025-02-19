from battleship.prompting import *
from battleship.board import *
from dataclasses import dataclass
from openai import OpenAI
import numpy as np
from abc import ABC, abstractmethod
from ast import literal_eval
import json
from pathlib import Path
import hashlib

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
client = OpenAI()
CODE_FAIL_STR = "code crashed"

@dataclass
class Question:
    text: str
    def get_cache_key(self, board_id):
        return f"{self.text.lower().replace(' ','_').replace('/','_')}_{board_id}"

class CodeQuestion:
    def __init__(self, question, fn, fn_string):
        self.question = question
        self.fn = fn
        self.fn_str = fn_string

    def __call__(self, board):
        try:
            result = self.fn(board)
            return result
        except:
            return CODE_FAIL_STR
        
@dataclass
class Answer:
    text: str
    code_question: CodeQuestion = None

class BaseSpotterModel(ABC):
    def __init__(self, board_id, board_experiment, history: List[dict], occ_tiles = None, use_cache = True):
        self.board_id = board_id
        self.board_experiment = board_experiment
        self.occ_tiles = occ_tiles
        self.history = history
        self.use_cache = use_cache
        self.cache_path = CACHE_DIR / self.__class__.__name__
        self.cache_path.mkdir(exist_ok=True)

    def read_cache(self, cache_key):
        cache_file = self.cache_path / f"{cache_key}.txt"
        if cache_file.exists():
            try:
                cache_dict = literal_eval(cache_file.read_text())
            except:
                print("Cache evaluation failed for:", cache_file.read_text())
            return cache_dict["result"]
        return None

    def write_cache(self, cache_key, prompt, code, result):
        cache_file = self.cache_path / f"{cache_key}.txt"
        cache_file.write_text(str({"prompt": prompt, "code": code, "result": result}))

    @abstractmethod 
    def _get_model_answer(self, question: Question) -> Answer:
        raise NotImplementedError

    def answer(self, question: Question) -> Answer:
        if self.use_cache:
            cache_key = question.get_cache_key(self.board_id)
            cached_result = self.read_cache(cache_key)
            if cached_result:
                return Answer(text=cached_result)
            
        result = self._get_model_answer(question)
        return result

class DirectSpotterModel(BaseSpotterModel):
    def _get_model_answer(self, question: Question) -> Answer:
        prompt = SpotterPrompt(target_trial_id = self.board_id, 
                    target_trial_experiment = self.board_experiment,
                    target_occ_tiles = self.occ_tiles, 
                    board_format = "grid", 
                    question=question,
                    use_code=False,
                    include_final_prefix=False,
                    history=self.history
                    )
        
        completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=prompt.to_chat_format(),
                    temperature=0
                )
        
        response = completion.choices[0].message.content
        self.write_cache(question.get_cache_key(self.board_id), prompt=prompt.to_chat_format(), code=None, result = response)

        answer = Answer(text=response)

        return answer

class CodeSpotterModel(BaseSpotterModel):
    def translate(self, question: Question) -> CodeQuestion:
        translation_prompt = SpotterPrompt(target_trial_id = self.board_id, 
                target_trial_experiment = self.board_experiment, 
                target_occ_tiles = self.occ_tiles,
                board_format = "grid", 
                question=question,
                use_code=True,
                include_final_prefix=False,
                history=self.history
                )
            
        completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=translation_prompt.to_chat_format(),
                    temperature=0
                )
        code_generated = completion.choices[0].message.content

        local_vars = {}

        try:
            _ = exec(code_generated, {'np': np}, local_vars)
        except:
            #print(code_generated)
            local_vars["answer"] = CODE_FAIL_STR

        return CodeQuestion(question=question, fn=local_vars["answer"], fn_string=code_generated), translation_prompt

    def _get_model_answer(self, question: Question) -> Answer:
        code_question, prompt = self.translate(question)

        board = Board.from_trial_id(trial_id=self.board_id, experiment=self.board_experiment).to_symbolic_array()

        result = code_question(board)
        self.write_cache(question.get_cache_key(self.board_id), prompt = prompt.to_chat_format(), code = code_question.fn_str, result = result)

        return Answer(text=result, code_question = code_question)
