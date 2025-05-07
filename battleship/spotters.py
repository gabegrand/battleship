import traceback
from abc import abstractmethod
from typing import List
from typing import Tuple

import numpy as np

from battleship.agents import Agent
from battleship.agents import Answer
from battleship.agents import BOOL_ANSWER_PATTERN
from battleship.agents import CacheData
from battleship.agents import client
from battleship.agents import CODE_ANSWER_PATTERN
from battleship.agents import CodeQuestion
from battleship.agents import Prompt
from battleship.agents import Question
from battleship.board import Board
from battleship.prompting import SpotterPrompt


class Spotter(Agent):
    def __init__(
        self,
        board_id,
        board_experiment,
        use_cache=True,
        model_string="gpt-4o",
        temperature=None,
        use_cot=False,
        decision_counter=None,
        index_counter=None,
        round_id=None,
        spotter_benchmark=None,
    ):
        self.board_id = board_id
        self.board_experiment = board_experiment
        self.temperature = temperature
        self.spotter_benchmark = spotter_benchmark

        # Use proper Agent initialization to handle model string and cache path
        super().__init__(
            seed=None,
            model_string=model_string,
            use_cache=use_cache,
            use_cot=use_cot,
            decision_counter=decision_counter,
            index_counter=index_counter,
            round_id=round_id,
        )

    @abstractmethod
    def _get_model_answer(
        self, question: Question, history: List[dict], occ_tiles
    ) -> Answer:
        raise NotImplementedError

    def answer(
        self, question: Question, history: List[dict] = None, occ_tiles=None
    ) -> Answer:
        self.index_counter.increment_counter()
        result, answer_cache = self._get_model_answer(question, history, occ_tiles)

        if self.spotter_benchmark is None:
            if self.use_cache:
                self.write_cache(
                    message_type="ANSWER",
                    cache_data=answer_cache,
                )

            return result
        else:
            return result, answer_cache


# ---------------------
# Spotter Classes
# ---------------------


class DirectSpotterModel(Spotter):
    def _get_model_answer(
        self, question: Question, history: List[dict], occ_tiles=None
    ) -> Tuple[Answer, CacheData]:
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

        prompt = Prompt(
            prompt=prompt.to_chat_format(),
            full_completion=completion.choices[0].message.content,
            extracted_completion=response,
            occ_tiles=occ_tiles,
        )

        answer = Answer(text=response)
        return answer, CacheData(
            message_text=response, occ_tiles=occ_tiles, prompts=[prompt]
        )


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
                if completion.choices:
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

        def blocked_input(*args):
            raise RuntimeError("input() function is not allowed in generated code")

        try:
            _ = exec(code_generated, {"np": np, "input": blocked_input}, local_vars)
        except Exception as e:
            tb = "".join(traceback.format_tb(e.__traceback__))
            local_vars["answer"] = None

        try:
            return CodeQuestion(
                question=question,
                fn=local_vars["answer"],
                fn_string=code_generated,
                translation_prompt=translation_prompt,
                full_completion=completion.choices[0].message.content,
                traceback=tb,
            )
        except:
            return CodeQuestion(
                question=question,
                fn=None,
                fn_string=code_generated,
                translation_prompt=translation_prompt,
                full_completion=completion.choices[0].message.content,
                traceback=tb,
            )

    def _get_model_answer(
        self, question: Question, history: List[dict], occ_tiles=None
    ) -> Tuple[Answer, CacheData]:
        code_question = self.translate(question, history, occ_tiles)

        board = Board.from_trial_id(
            trial_id=self.board_id, experiment=self.board_experiment
        ).to_symbolic_array()

        result = code_question(board)

        prompt = Prompt(
            prompt=code_question.translation_prompt.to_chat_format(),
            full_completion=code_question.full_completion,
            extracted_completion=result,
            occ_tiles=occ_tiles,
        )

        return Answer(text=result, code_question=code_question), CacheData(
            message_text=result, occ_tiles=occ_tiles, prompts=[prompt]
        )
