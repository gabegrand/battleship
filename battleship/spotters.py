import logging
import time
import traceback
from abc import abstractmethod
from typing import List
from typing import Tuple

import numpy as np

from battleship.agents import ActionData
from battleship.agents import Agent
from battleship.agents import Answer
from battleship.agents import BOOL_ANSWER_PATTERN
from battleship.agents import CODE_ANSWER_PATTERN
from battleship.agents import CodeQuestion
from battleship.agents import get_openai_client
from battleship.agents import NullCodeQuestion
from battleship.agents import Question
from battleship.board import Board
from battleship.prompting import SpotterPrompt
from battleship.strategies import AnswerStrategy
from battleship.utils import parse_answer_to_str


logger = logging.getLogger(__name__)


class Spotter(Agent):
    def __init__(
        self,
        board_id,
        board_experiment,
        answer_strategy=None,
        model_string="gpt-4o",
        temperature=None,
        use_cot=False,
        json_path=None,
    ):
        self.board_id = board_id
        self.board_experiment = board_experiment
        self.temperature = temperature
        self.answer_strategy = answer_strategy
        self.client = get_openai_client()

        super().__init__(
            seed=None,
            model_string=model_string,
            use_cot=use_cot,
            json_path=json_path,
        )

    def answer(
        self, question: Question, occ_tiles: np.ndarray, history: List[dict] = None
    ) -> Answer:
        answer, action_data = self.answer_strategy(
            question=question,
            occ_tiles=occ_tiles,
            history=history,
        )

        # Save the action data
        self.save_action_data(action_data)

        return answer

    def translate(
        self,
        question: Question,
        occ_tiles: np.ndarray,
        history: List[dict],
    ) -> CodeQuestion:
        if hasattr(self.answer_strategy, "translate"):
            return self.answer_strategy.translate(question, occ_tiles, history)
        else:
            raise NotImplementedError(
                f"Spotter {self.__class__.__name__} does not have a translate method."
            )


# ---------------------
# Answer Strategy Classes
# ---------------------


class DirectAnswerStrategy(AnswerStrategy):
    def __init__(
        self,
        board_id,
        board_experiment,
        model_string="gpt-4o",
        temperature=None,
        use_cot=False,
    ):
        super().__init__()
        self.board_id = board_id
        self.board_experiment = board_experiment
        self.model_string = model_string
        self.temperature = temperature
        self.use_cot = use_cot
        self.client = get_openai_client()

    def __call__(
        self,
        question: Question,
        occ_tiles: np.ndarray,
        history: List[dict] = None,
        n_attempts=10,
    ) -> Tuple[Answer, ActionData]:
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
        completion = None
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

        answer = Answer(text=response)

        # Create ActionData object
        action_data = ActionData(
            action="answer",
            prompt=str(prompt),
            completion=completion.model_dump() if completion else None,
            question=question,
            answer=answer,
        )

        return answer, action_data


class CodeAnswerStrategy(AnswerStrategy):
    def __init__(
        self,
        board_id,
        board_experiment,
        model_string="gpt-4o",
        temperature=None,
        use_cot=False,
    ):
        super().__init__()
        self.board_id = board_id
        self.board_experiment = board_experiment
        self.model_string = model_string
        self.temperature = temperature
        self.use_cot = use_cot
        self.client = get_openai_client()

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
                    completion=completion.model_dump(),
                )
            except Exception as e:
                logging.error(
                    f"CodeQuestion.translate(): Error in evaluation (attempt {attempt+1}/{n_attempts}): {e}\n{traceback.format_exc()}"
                )

        return NullCodeQuestion(
            question=question, translation_prompt=translation_prompt
        )

    def extract_code(self, text: str) -> str:
        """
        Extracts the code from the text using regex.
        """
        match = CODE_ANSWER_PATTERN.search(text)
        if match:
            return match.group(1)
        else:
            return None

    def __call__(
        self,
        question: Question,
        occ_tiles: np.ndarray,
        history: List[dict],
    ) -> Tuple[Answer, ActionData]:
        code_question = self.translate(
            question,
            occ_tiles=occ_tiles,
            history=history,
        )

        true_board = Board.from_trial_id(
            trial_id=self.board_id, experiment=self.board_experiment
        ).to_numpy()

        partial_board = occ_tiles.copy()

        answer = code_question(true_board=true_board, partial_board=partial_board)

        # Create ActionData object
        action_data = ActionData(
            action="answer",
            prompt="Code translation and execution",  # Could be more detailed
            question=question,
            answer=answer,
        )

        return answer, action_data


# ---------------------
# Legacy Classes (for backward compatibility)
# ---------------------


class DirectSpotterModel(Spotter):
    """Legacy class - use Spotter with DirectAnswerStrategy instead"""

    def __init__(self, *args, **kwargs):
        # Extract strategy-specific parameters
        strategy_kwargs = {
            "board_id": args[0] if len(args) > 0 else kwargs.get("board_id"),
            "board_experiment": args[1]
            if len(args) > 1
            else kwargs.get("board_experiment"),
            "model_string": kwargs.get("model_string", "gpt-4o"),
            "temperature": kwargs.get("temperature"),
            "use_cot": kwargs.get("use_cot", False),
        }

        answer_strategy = DirectAnswerStrategy(**strategy_kwargs)
        super().__init__(answer_strategy=answer_strategy, *args, **kwargs)


class CodeSpotterModel(Spotter):
    """Legacy class - use Spotter with CodeAnswerStrategy instead"""

    def __init__(self, *args, **kwargs):
        # Extract strategy-specific parameters
        strategy_kwargs = {
            "board_id": args[0] if len(args) > 0 else kwargs.get("board_id"),
            "board_experiment": args[1]
            if len(args) > 1
            else kwargs.get("board_experiment"),
            "model_string": kwargs.get("model_string", "gpt-4o"),
            "temperature": kwargs.get("temperature"),
            "use_cot": kwargs.get("use_cot", False),
        }

        answer_strategy = CodeAnswerStrategy(**strategy_kwargs)
        super().__init__(answer_strategy=answer_strategy, *args, **kwargs)


def create_spotter(
    spotter_type,
    board_id,
    board_experiment,
    model_string="gpt-4o",
    temperature=None,
    use_cot=False,
    json_path=None,
):
    """
    Factory function to create Spotter instances with properly configured answer strategies.
    """

    if spotter_type == "DirectSpotterModel":
        answer_strategy = DirectAnswerStrategy(
            board_id=board_id,
            board_experiment=board_experiment,
            model_string=model_string,
            temperature=temperature,
            use_cot=use_cot,
        )
    elif spotter_type == "CodeSpotterModel":
        answer_strategy = CodeAnswerStrategy(
            board_id=board_id,
            board_experiment=board_experiment,
            model_string=model_string,
            temperature=temperature,
            use_cot=use_cot,
        )
    else:
        raise ValueError(f"Unknown spotter type: {spotter_type}")

    return Spotter(
        board_id=board_id,
        board_experiment=board_experiment,
        answer_strategy=answer_strategy,
        model_string=model_string,
        temperature=temperature,
        use_cot=use_cot,
        json_path=json_path,
    )
