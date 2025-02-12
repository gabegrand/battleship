import os

import numpy as np
import pandas as pd

from battleship.board import Board
from battleship.board import SYMBOL_MEANING_MAPPING
from battleship.board import BoardFormat
from battleship.board import TRIAL_IDS

HUMAN_DATASET_PATH = os.path.join(
    f"{os.path.abspath(os.path.dirname(__file__))}",
    "prompts",
    "human_question_dataset.csv",
)


class BasePrompt(object):
    """Base class for constructing prompts for the Battleship task.

    Sampling logic is shared between QuestionGenerationPrompt and TranslationPrompt.
    Each subclass implements its own .to_chat_format() method.

    Set random_seed to ensure determinism. If random_seed is None, a random seed will be generated.

    """

    PREFIX_QUESTION = "User:"
    PREFIX_CODE = "Query:"

    EXAMPLE_DELIMITER = "#" * 4

    def __init__(
        self,
        target_trial_id: int,
        target_trial_experiment: str,
        board_format: BoardFormat = None,
        n_example_trials: int = 0,
        n_examples_per_trial: int = 1,
        include_instructions: bool = True,
        include_board: bool = True,
        include_system_prompt: bool = True,
        include_final_prefix: bool = True,
        random_seed: int = None,
    ):
        self.target_trial_experiment = target_trial_experiment
        self.target_trial_id = target_trial_id
        self.board_format = board_format
        self.n_example_trials = n_example_trials
        self.n_questions_per_trial = n_examples_per_trial
        self.include_system_prompt = include_system_prompt
        self.include_board = include_board
        self.include_instructions = include_instructions
        self.include_final_prefix = include_final_prefix
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
            ).tolist()

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
        else:
            self.example_trial_ids = []
            self.examples = []

    def __str__(self):
        return "\n".join([str(message["content"]) for message in self.to_chat_format()])
        # return "\n".join([f"[{message['role']}]{message['content']}" for message in self.to_chat_format()])

    def to_dict(self):
        return {
            "target_trial_id": self.target_trial_id,
            "n_example_trials": self.n_example_trials,
            "n_questions_per_trial": self.n_questions_per_trial,
            "random_seed": self.random_seed
            if isinstance(self.random_seed, int)
            else None,
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

    @staticmethod
    def optional_space(prefix: str = None, text: str = None):
        output = ""
        if prefix:
            output += prefix
        if prefix and text:
            output += " "
        if text:
            output += text
        return output

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
    "There are four ships on the board: Green, Red, Purple, and Orange. "
    "Ships are oriented either horizontally or vertically and can be 2, 3, or 4 tiles in length. "
    "The board is an 8x8 grid, with numbered rows 1, 2, 3, 4, 5, 6, 7, 8 and lettered columns A, B, C, D, E, F, G, H. "
    "Coordinates are specified as a row, column pair. For example, 2-C is the tile in row 2, column C.\n"
)

# Task description
PROMPT_TASK_QUESTION_GENERATION = (
    "You will be given a partially-revealed game board. "
    "Your task is to ask a single question that will help you gain information about the position of the remaining hidden ships on the board. "
    "You can ask any question, but it must be answerable with a single word answer. "
)

#Board format descriptions
PROMPT_VARIANT_GRID = "The board is represented as a grid with the following symbols:" + "\n".join([item[0]+": "+item[1] for item in SYMBOL_MEANING_MAPPING.items() if item[0] != "H"]) + "\n"
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

    PREFIX_QUESTION = "Question:"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.board_format is None:
            raise ValueError("Board format must be specified.")

    def to_chat_format(self):
        messages = []

        if self.include_instructions:
            if self.include_system_prompt:
                messages.append({"role": "system", "content": PROMPT_SYSTEM})

            messages.append({"role": "user", "content": PROMPT_GAME})
            messages.append(
                {"role": "user", "content": PROMPT_TASK_QUESTION_GENERATION}
            )

            if self.include_board:
                if self.board_format == BoardFormat.GRID:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_GRID})
                elif self.board_format == BoardFormat.TEXTUAL:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_TEXTUAL})
                elif self.board_format == BoardFormat.VISUAL:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_VISUAL})
                else:
                    raise ValueError(f"Unknown board format: {self.board_format}")

        if self.n_example_trials > 0:
            if self.include_instructions:
                messages.append({"role": "user", "content": PROMPT_EXAMPLES})

            for trial_id in self.example_trial_ids:
                if self.include_board:
                    board_str = Board.from_trial_id(trial_id=trial_id, experiment=self.target_trial_experiment).to_format(
                        self.board_format
                    )
                    if self.board_format == BoardFormat.VISUAL:
                        messages.append(
                            {
                                "role": "user",
                                "name": "example_user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{board_str}",
                                            "detail": "low",
                                        },
                                    },
                                ],
                            },
                        )
                    else:
                        if not board_str.endswith("\n"):
                            board_str += "\n"
                        messages.append(
                            {
                                "role": "user",
                                "name": "example_user",
                                "content": f"\n{self.EXAMPLE_DELIMITER}\n\n{board_str}",
                            }
                        )
                for example in filter(
                    lambda x: x["trial_id"] == trial_id, self.examples
                ):
                    messages.append(
                        {
                            "role": "assistant",
                            "name": "example_agent",
                            "content": self.optional_space(
                                self.PREFIX_QUESTION, example["question"]
                            ),
                        }
                    )

        if self.include_board:
            if self.include_instructions:
                messages.append({"role": "user", "content": "\n" + PROMPT_TARGET_BOARD})

            board_str = Board.from_trial_id(self.target_trial_id, self.target_trial_experiment).to_format(
                self.board_format
            )
            if self.board_format == BoardFormat.VISUAL:
                messages.append(
                    {
                        "role": "user",
                        "name": "example_user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{board_str}",
                                    "detail": "low",
                                },
                            }
                        ],
                    },
                )
            else:
                if not board_str.endswith("\n"):
                    board_str += "\n"
                messages.append(
                    {
                        "role": "user",
                        "content": f"{self.EXAMPLE_DELIMITER}\n\n{board_str}",
                    }
                )

        # Set to false if using GPT-4; the model will generate this message itself
        if self.include_final_prefix:
            messages.append(
                {
                    "role": "assistant",
                    "content": self.optional_space(self.PREFIX_QUESTION),
                }
            )

        return messages


"""
Translation prompt for the Battleship task.
"""

PROMPT_TASK_TRANSLATION = (
    "Your task is to translate each of the user's questions into a query program.\n"
)


class TranslationPrompt(BasePrompt):
    """Translation prompt for the Battleship task."""

    def __init__(
        self,
        target_question: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.target_question = target_question

        if not self.n_example_trials > 0:
            raise ValueError("Translation prompt requires example trials.")

    def to_chat_format(self):
        messages = []

        if self.include_instructions:
            messages.append({"role": "user", "content": PROMPT_GAME})
            messages.append({"role": "user", "content": PROMPT_TASK_TRANSLATION})

        for example in self.examples:
            messages.append(
                {
                    "role": "user",
                    "name": "example_user",
                    "content": self.optional_space(
                        self.PREFIX_QUESTION, example["question"]
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "name": "example_agent",
                    "content": self.optional_space(self.PREFIX_CODE, example["code"]),
                }
            )

        if self.target_question:
            messages.append(
                {
                    "role": "user",
                    "content": self.optional_space(
                        self.PREFIX_QUESTION, self.target_question
                    ),
                }
            )
            # Set to false if using GPT-4; the model will generate this message itself
            if self.include_final_prefix:
                messages.append(
                    {
                        "role": "assistant",
                        "content": self.optional_space(self.PREFIX_CODE),
                    }
                )

        return messages

"""
Spotter prompt for the Collaborative Battleship task.
"""

PROMPT_SYSTEM_SPOTTER = (
    "You are a game-playing agent. "
    "Read the game instructions and examples carefully. "
    "Respond in one word or with code, as necessary. "
    "Do not include any other explanation or prose.\n"
)

PROMPT_TASK_BASE_SPOTTER = (
    "You will be given a fully-revealed game board in a numpy array-like style. "
    "You are in a team with a partner, the 'captain', who cannot see all of the board, but can ask you a question per turn about it. "
    "Your task is to answer the captain's question honestly and as accurately as possible." 
)

PROMPT_TASK_CODE_SPOTTER = (
    "Please generate a piece of numpy code that answers the question. Make sure your code returns 'Yes' or 'No'. This code will be executed in an environment with both numpy (namespaced as np) and a 'board' variable (a numpy representation of the board) so give some function answer(board) that could be used across any board. Don't add any tickmarks or other formatting to denote that this is code, and do not invoke 'python' in the CLI."
)

PROMPT_TASK_DIRECT_SPOTTER = (
    "You can only answer with 'Yes' or 'No'. Please only answer with a single word." 
)

# Example questions
PROMPT_EXAMPLES_SPOTTER = (
    "Here are the past turns in the game so far, which include what questions the user has asked, their answers, and what moves the user made."
)

QUESTION_PRESENTATION_PROMPT = "Here is the question the captain asked: "

class SpotterPrompt(BasePrompt):
    """Prompt for generating questions for the Battleship task."""

    PREFIX_QUESTION = "Question:"

    def __init__(
        self,
        question,
        use_code = False,
        history = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.question = question
        self.history = history
        self.use_code = use_code
        if self.board_format is None:
            raise ValueError("Board format must be specified.")
        
    def to_chat_format(self):
        messages = []

        if self.include_instructions:
            if self.include_system_prompt:
                messages.append({"role": "system", "content": PROMPT_SYSTEM_SPOTTER})

            messages.append({"role": "user", "content": PROMPT_GAME})
            messages.append(
                {"role": "user", "content": PROMPT_TASK_BASE_SPOTTER}
            )
            
            if self.use_code:
                messages.append(
                    {"role": "user", "content": PROMPT_TASK_CODE_SPOTTER}
                )
            else:
                messages.append(
                    {"role": "user", "content": PROMPT_TASK_DIRECT_SPOTTER}
                )

            if self.include_board:
                if self.board_format == BoardFormat.GRID:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_GRID})
                elif self.board_format == BoardFormat.TEXTUAL:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_TEXTUAL})
                elif self.board_format == BoardFormat.VISUAL:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_VISUAL})
                else:
                    raise ValueError(f"Unknown board format: {self.board_format}")

        if self.history is not None:
            if self.include_board:
                if self.include_instructions:
                    messages.append({"role": "user", "content": "\n" + PROMPT_TARGET_BOARD})

                board_str = Board.from_trial_id(self.target_trial_id, self.target_trial_experiment).to_format(
                    self.board_format
                )
                if self.board_format == BoardFormat.VISUAL:
                    messages.append(
                        {
                            "role": "user",
                            "name": "example_user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{board_str}",
                                        "detail": "low",
                                    },
                                }
                            ],
                        },
                    )
                else:
                    if not board_str.endswith("\n"):
                        board_str += "\n"
                    messages.append(
                        {
                            "role": "user",
                            "content": f"{self.EXAMPLE_DELIMITER}\n\n{board_str}",
                        }
                )
            if self.include_instructions:
                messages.append({"role": "user", "content": PROMPT_EXAMPLES_SPOTTER})

            for example in self.history:
                if example["decision"] == "question":
                    decision_str = f"""Question: {example["question"]}\n 
                                        Answer: {example["answer"]}"""
                else:
                    decision_str = f"""Move: {example["move"]}"""

                messages.append(
                    {
                        "role": "user",
                        "content": f""" {decision_str}\n
                                        {self.EXAMPLE_DELIMITER}\n\n""", 
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": QUESTION_PRESENTATION_PROMPT,
            }
        )

        messages.append(
            {
                "role": "user",
                "content": self.question.text,
            }
        )

        # Set to false if using GPT-4; the model will generate this message itself
        if self.include_final_prefix:
            messages.append(
                {
                    "role": "assistant",
                    "content": self.optional_space(self.PREFIX_QUESTION),
                }
            )

        return messages