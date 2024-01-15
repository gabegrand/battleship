import os

import numpy as np
import pandas as pd

from battleship.board import Board
from battleship.board import BoardFormat
from battleship.board import TRIAL_IDS

HUMAN_DATASET_PATH = os.path.join(
    f"{os.path.abspath(os.path.dirname(__file__))}",
    "prompts",
    "examples_full.csv",
)


class BasePrompt(object):
    """Base class for constructing prompts for the Battleship task.

    Sampling logic is shared between QuestionGenerationPrompt and TranslationPrompt.
    Each subclass implements its own .to_chat_format() method.

    Set random_seed to ensure determinism. If random_seed is None, a random seed will be generated.

    """

    def __init__(
        self,
        target_trial_id: int,
        board_format: BoardFormat = None,
        n_example_trials: int = 0,
        n_examples_per_trial: int = 1,
        include_system_prompt: bool = True,
        random_seed: int = None,
    ):
        self.target_trial_id = target_trial_id
        self.board_format = board_format
        self.n_example_trials = n_example_trials
        self.n_questions_per_trial = n_examples_per_trial
        self.include_system_prompt = include_system_prompt
        self.random_seed = random_seed

        # Sample example trial ids, excluding the target trial id
        if self.n_example_trials > 0:
            self.rng = np.random.default_rng(self.random_seed)
            self.example_trial_ids = self.rng.choice(
                [
                    trial_id
                    for trial_id in TRIAL_IDS
                    if trial_id != self.target_trial_id
                ],
                size=self.n_example_trials,
                replace=False,
            )

            # Load question dataset
            df = pd.read_csv(HUMAN_DATASET_PATH)

            # Sample questions from the question dataset
            self.examples = []
            for trial_id in self.example_trial_ids:
                df_trial_examples = df.sample(
                    n=self.n_questions_per_trial, random_state=self.rng
                )
                for example_id, example in df_trial_examples.iterrows():
                    self.examples.append(
                        {
                            "example_id": example_id,
                            "trial_id": trial_id,
                            "question": example["question"],
                            "code": example["code"],
                        }
                    )

    def __str__(self):
        return "\n".join([message["content"] for message in self.to_chat_format()])

    def __dict__(self):
        return {
            "target_trial_id": self.target_trial_id,
            "n_example_trials": self.n_example_trials,
            "n_questions_per_trial": self.n_questions_per_trial,
            "random_seed": self.random_seed,
            "example_trial_ids": self.example_trial_ids,
            "examples": self.examples,
            "include_system_prompt": self.include_system_prompt,
            "text": str(self),
        }

    @staticmethod
    def from_dict(data):
        return QuestionGenerationPrompt(
            target_trial_id=data["target_trial_id"],
            n_example_trials=data["n_example_trials"],
            n_questions_per_trial=data["n_questions_per_trial"],
            random_seed=data["random_seed"],
            include_system_prompt=data["include_system_prompt"],
        )

    def to_chat_format(self):
        """Returns a list of messages in OpenAI chat format.

        {
            'role': <user/assistant/system>,
            'content': <message content>,
        }

        """
        raise NotImplementedError()


"""
Question generation prompt for the Battleship task.
"""

# Optional system prompt for GPT agents
PROMPT_SYSTEM = (
    "You are a game-playing agent. "
    "Read the game instructions and examples carefully. "
    "Respond with a single question that can be answered with one word. "
    "Do not include any other explanation or prose.\n"
)

# Main game description
PROMPT_GAME = (
    "You are playing the board game Battleship. "
    "There are three ships on the board: Red, Blue, and Purple. "
    "Ships are oriented either horizontally or vertically and can be 2, 3, or 4 tiles in length. "
    "The board is a 6x6 grid, with numbered rows 1, 2, 3, 4, 5, 6 and lettered columns A, B, C, D, E, F. "
    "Coordinates are specified as a row, column pair. For example, 2-C is the tile in row 2, column C.\n"
)

# Task description
PROMPT_TASK_QUESTION_GENERATION = (
    "You will be given a partially-revealed game board. "
    "Your task is to ask a single question that will help you gain information about the position of the remaining hidden ships on the board. "
    "You can ask any question, but it must be answerable with a single word answer. "
)

# Board format descriptions
PROMPT_VARIANT_GRID = (
    "The board is represented as a grid with the following symbols:\n\n"
    "H: Hidden\n"
    "W: Water\n"
    "R: Red ship\n"
    "B: Blue ship\n"
    "P: Purple ship\n"
)
PROMPT_VARIANT_TEXTUAL = "The board is represented as a textual description.\n"
PROMPT_VARIANT_VISUAL = "The board is represented as an image, with light gray indicating hidden tiles, dark gray indicating water tiles, and red, blue and purple indicating ship tiles.\n"

# Example questions
PROMPT_EXAMPLES = (
    "Here are some examples of questions from other agents about different boards."
)

# Target board
PROMPT_TARGET_BOARD = "Now, it's your turn. Here is your board:\n"


class QuestionGenerationPrompt(BasePrompt):
    """Prompt for generating questions for the Battleship task."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.board_format is None:
            raise ValueError("Board format must be specified.")

    def to_chat_format(self):
        messages = []
        if self.include_system_prompt:
            messages.append({"role": "system", "content": PROMPT_SYSTEM})

        messages.append({"role": "user", "content": PROMPT_GAME})
        messages.append({"role": "user", "content": PROMPT_TASK_QUESTION_GENERATION})

        if self.board_format == BoardFormat.GRID:
            messages.append({"role": "user", "content": PROMPT_VARIANT_GRID})
        elif self.board_format == BoardFormat.TEXTUAL:
            messages.append({"role": "user", "content": PROMPT_VARIANT_TEXTUAL})
        elif self.board_format == BoardFormat.VISUAL:
            # TODO: Requires a slightly different message format. Also, we should format the images to be 512px x 512px image for low-res mode with GPT-4V.
            raise NotImplementedError("Visual board format not yet implemented.")
        else:
            raise ValueError(f"Unknown board format: {self.board_format}")

        if self.n_example_trials > 0:
            messages.append({"role": "user", "content": PROMPT_EXAMPLES})

            for trial_id in self.example_trial_ids:
                board = Board.from_trial_id(trial_id)
                messages.append(
                    {
                        "role": "user",
                        "name": "example_user",
                        "content": f"\nBoard:\n{board.to_format(self.board_format)}\n",
                    }
                )
                for example in filter(
                    lambda x: x["trial_id"] == trial_id, self.examples
                ):
                    messages.append(
                        {
                            "role": "assistant",
                            "name": "example_agent",
                            "content": "Question: " + example["question"],
                        }
                    )

        messages.append({"role": "user", "content": PROMPT_TARGET_BOARD})
        board = Board.from_trial_id(self.target_trial_id)
        messages.append(
            {"role": "user", "content": f"{board.to_format(self.board_format)}\n"}
        )

        # TODO: Awkward
        messages.append({"role": "assistant", "content": "Question:"})

        return messages


"""
Translation prompt for the Battleship task.
"""

PROMPT_TASK_TRANSLATION = "Your task is to translate each question into code.\n"


class TranslationPrompt(BasePrompt):
    """Translation prompt for the Battleship task."""

    def __init__(
        self,
        target_question: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.target_question = target_question

        if not self.n_example_trials > 0:
            raise ValueError("Translation prompt requires example trials.")

    def to_chat_format(self):
        messages = []

        messages.append({"role": "user", "content": PROMPT_GAME})
        messages.append({"role": "user", "content": PROMPT_TASK_TRANSLATION})

        for example in self.examples:
            messages.append(
                {
                    "role": "user",
                    "name": "example_user",
                    "content": f"Question: {example['question']}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "name": "example_agent",
                    "content": f"Code: {example['code']}",
                }
            )

        messages.append(
            {"role": "user", "content": f"Question: {self.target_question}"}
        )

        # TODO: Awkward
        messages.append({"role": "assistant", "content": f"Code:"})

        return messages
