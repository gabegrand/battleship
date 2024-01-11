import os

import numpy as np
import pandas as pd

from battleship.board import Board
from battleship.board import BoardFormat
from battleship.board import TRIAL_IDS

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
    "The board is a 6x6 grid, with numbered rows 1, 2, 3, ... and lettered columns A, B, C, ... "
    "Coordinates are specified as a row, column pair. For example, 2-C is the tile in row 2, column C.\n"
)

# Task description
PROMPT_TASK = (
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
PROMPT_VARIANT_LINGUISTIC = "The board is represented as a textual description.\n"
PROMPT_VARIANT_VISUAL = "The board is represented as an image, with light gray indicating hidden tiles, dark gray indicating water tiles, and red, blue and purple indicating ship tiles.\n"

# Example questions
PROMPT_EXAMPLES = (
    "Here are some examples of questions from other agents about different boards."
)

# Target board
PROMPT_TARGET_BOARD = "Now, it's your turn. Here is your board:\n"


class Prompt(object):
    """Class for constructing prompts for the Battleship task."""

    def __init__(
        self,
        target_trial_id: int,
        board_format: BoardFormat,
        n_example_trials: int = 0,
        n_questions_per_trial: int = 3,
        include_system_prompt: bool = True,
        random_seed: int = 123,
    ):
        self.target_trial_id = target_trial_id
        self.board_format = board_format
        self.n_example_trials = n_example_trials
        self.n_questions_per_trial = n_questions_per_trial
        self.include_system_prompt = include_system_prompt
        self.random_seed = random_seed

        # Sample example trial ids, excluding the target trial id
        if self.n_example_trials > 0:
            rng = np.random.default_rng(self.random_seed)
            self.example_trial_ids = rng.choice(
                [
                    trial_id
                    for trial_id in TRIAL_IDS
                    if trial_id != self.target_trial_id
                ],
                size=self.n_example_trials,
                replace=False,
            )

            # Load question dataset
            df = pd.read_csv(
                os.path.join(
                    f"{os.path.abspath(os.path.dirname(__file__))}",
                    "prompts",
                    "examples_full.csv",
                )
            )

            # Sample questions from the question dataset
            self.examples = []
            for trial_id in self.example_trial_ids:
                self.examples.append(
                    {
                        "trial_id": trial_id,
                        "questions": df.sample(
                            n=self.n_questions_per_trial, random_state=rng
                        )["question"].tolist(),
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
            "example_questions": self.examples,
            "include_system_prompt": self.include_system_prompt,
            "text": str(self),
        }

    @staticmethod
    def from_dict(data):
        return Prompt(
            target_trial_id=data["target_trial_id"],
            n_example_trials=data["n_example_trials"],
            n_questions_per_trial=data["n_questions_per_trial"],
            random_seed=data["random_seed"],
            include_system_prompt=data["include_system_prompt"],
        )

    def to_chat_format(self):
        messages = []
        if self.include_system_prompt:
            messages.append({"role": "system", "content": PROMPT_SYSTEM})

        messages.append({"role": "user", "content": PROMPT_GAME})
        messages.append({"role": "user", "content": PROMPT_TASK})

        if self.board_format == BoardFormat.GRID:
            messages.append({"role": "user", "content": PROMPT_VARIANT_GRID})
        elif self.board_format == BoardFormat.LINGUISTIC:
            messages.append({"role": "user", "content": PROMPT_VARIANT_LINGUISTIC})
        elif self.board_format == BoardFormat.VISUAL:
            # TODO: Requires a slightly different message format. Also, we should format the images to be 512px x 512px image for low-res mode with GPT-4V.
            raise NotImplementedError("Visual board format not yet implemented.")
        else:
            raise ValueError(f"Unknown board format: {self.board_format}")

        if self.n_example_trials > 0:
            messages.append({"role": "user", "content": PROMPT_EXAMPLES})

            for example in self.examples:
                board = Board.from_trial_id(example["trial_id"])
                messages.append(
                    {
                        "role": "user",
                        "name": "example_user",
                        "content": f"\nBoard:\n{board.to_format(self.board_format)}\n",
                    }
                )
                for question in example["questions"]:
                    messages.append(
                        {
                            "role": "assistant",
                            "name": "example_agent",
                            "content": "Question: " + question,
                        }
                    )

        messages.append({"role": "user", "content": PROMPT_TARGET_BOARD})
        board = Board.from_trial_id(self.target_trial_id)
        messages.append(
            {"role": "user", "content": f"{board.to_format(self.board_format)}\n"}
        )
        messages.append({"role": "assistant", "content": "Question:"})

        return messages
