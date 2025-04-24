import traceback
from abc import abstractmethod
from typing import List

import numpy as np

from battleship.battleship.agents import Agent
from battleship.battleship.agents import Answer
from battleship.battleship.agents import BOOL_ANSWER_PATTERN
from battleship.battleship.agents import CacheMode
from battleship.battleship.agents import client
from battleship.battleship.agents import CODE_ANSWER_PATTERN
from battleship.battleship.agents import CodeQuestion
from battleship.battleship.agents import Question
from battleship.board import Board
from battleship.prompting import SpotterPrompt


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
            Agent.increment_counter(self)
            return result

        # Normal cache hit
        elif cached_result is not None:
            # Increment counter after answer
            Agent.increment_counter(self)
            return Answer(text=cached_result)

        # Cache miss
        result = self._get_model_answer(question, history, occ_tiles)

        # Increment counter after answer
        Agent.increment_counter(self)
        return result


# ---------------------
# Spotter Classes
# ---------------------


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
                    code_generated = response_match.group(1)
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
