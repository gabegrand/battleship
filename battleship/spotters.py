import logging
import traceback
from abc import abstractmethod
from typing import List
from typing import Tuple

import numpy as np

from battleship.agents import Agent
from battleship.agents import Answer
from battleship.agents import BOOL_ANSWER_PATTERN
from battleship.agents import CacheData
from battleship.agents import CODE_ANSWER_PATTERN
from battleship.agents import CodeQuestion
from battleship.agents import get_openai_client
from battleship.agents import NullCodeQuestion
from battleship.agents import Prompt
from battleship.agents import Question
from battleship.board import Board
from battleship.prompting import SpotterPrompt
from battleship.utils import parse_answer_to_str


logger = logging.getLogger(__name__)


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
        stage_dir=None,
        prompts_dir=None,
    ):
        self.board_id = board_id
        self.board_experiment = board_experiment
        self.temperature = temperature
        self.spotter_benchmark = spotter_benchmark
        self.client = get_openai_client()

        # Use proper Agent initialization to handle model string and cache path
        super().__init__(
            seed=None,
            model_string=model_string,
            use_cache=use_cache,
            use_cot=use_cot,
            decision_counter=decision_counter,
            index_counter=index_counter,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )

    @abstractmethod
    def _get_model_answer(
        self, question: Question, occ_tiles: np.ndarray, history: List[dict] = None
    ) -> Answer:
        raise NotImplementedError

    def answer(
        self, question: Question, occ_tiles: np.ndarray, history: List[dict] = None
    ) -> Answer:
        self.index_counter.increment_counter()
        result, answer_cache = self._get_model_answer(
            question,
            occ_tiles=occ_tiles,
            history=history,
        )

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
        self,
        question: Question,
        occ_tiles: np.ndarray,
        history: List[dict] = None,
        n_attempts=10,
    ) -> Tuple[Answer, CacheData]:
        prompt = SpotterPrompt(
            target_trial_id=self.board_id,
            target_trial_experiment=self.board_experiment,
            target_occ_tiles=occ_tiles,
            board_format="grid",
            question=question,
            use_code=False,
            history=history,
            use_cot=self.use_cot,
        )
        logging.info(str(prompt))

        response = None
        for attempt in range(n_attempts):
            completion = self.client.chat.completions.create(
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

            if response is not None:
                break

        logging.info(response)

        if isinstance(response, str):
            response = response.lower()

        output_prompt = Prompt(
            prompt=prompt.to_chat_format(),
            full_completion=completion.choices[0].message.content,
            extracted_completion=response,
            occ_tiles=occ_tiles,
        )

        answer = Answer(text=response)
        return answer, CacheData(
            message_text=response, occ_tiles=occ_tiles, prompts=[output_prompt]
        )


class CodeSpotterModel(Spotter):
    def translate(
        self,
        question: Question,
        occ_tiles: np.ndarray,
        history: List[dict],
        n_attempts: int = 10,
    ) -> CodeQuestion:
        translation_prompt = SpotterPrompt(
            target_trial_id=self.board_id,
            target_trial_experiment=self.board_experiment,
            target_occ_tiles=occ_tiles,
            board_format="grid",
            question=question,
            use_code=True,
            history=history,
            use_cot=self.use_cot,
        )
        logging.info(str(translation_prompt))

        # Generate code using the translation prompt
        for attempt in range(n_attempts):
            completion = self.client.chat.completions.create(
                model=self.model_string,
                messages=translation_prompt.to_chat_format(),
                temperature=self.temperature,
            )

            content = completion.choices[0].message.content
            logging.info(content)

            # Extract the code block from the response
            fn_text: str = self.extract_code(content)
            if not fn_text:
                logging.warning(
                    f"CodeQuestion.translate(): Failed to extract code (attempt {attempt+1}/{n_attempts})."
                )
                continue

            # Evaluate the extracted code
            try:
                return CodeQuestion(
                    question=question,
                    fn_text=fn_text,
                    translation_prompt=translation_prompt,
                    full_completion=content,
                )
            except Exception as e:
                logging.error(
                    f"CodeQuestion.translate(): Error in evaluation (attempt {attempt+1}/{n_attempts}): {e}\n{traceback.format_exc()}"
                )

        return NullCodeQuestion(translation_prompt=translation_prompt)

    def extract_code(self, text: str) -> str:
        """
        Extracts the code from the text using regex.
        """
        match = CODE_ANSWER_PATTERN.search(text)
        if match:
            return match.group(1)
        else:
            return None

    def _get_model_answer(
        self,
        question: Question,
        occ_tiles: np.ndarray,
        history: List[dict],
    ) -> Tuple[Answer, CacheData]:
        code_question = self.translate(
            question,
            occ_tiles=occ_tiles,
            history=history,
        )

        true_board = Board.from_trial_id(
            trial_id=self.board_id, experiment=self.board_experiment
        ).to_numpy()

        partial_board = occ_tiles.copy()

        result = code_question(true_board, partial_board)

        result_text = parse_answer_to_str(result)

        output_prompt = Prompt(
            prompt=code_question.translation_prompt.to_chat_format(),
            full_completion=code_question.full_completion,
            extracted_completion=result,
            occ_tiles=occ_tiles,
        )

        return Answer(text=result_text, code_question=code_question), CacheData(
            message_text=result_text, occ_tiles=occ_tiles, prompts=[output_prompt]
        )
