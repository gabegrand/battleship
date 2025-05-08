import numpy as np

from battleship.board import Board
from battleship.board import BoardFormat
from battleship.board import coords_to_tile
from battleship.game import Decision

PROMPT_GAME = (
    "You are playing the board game Battleship. In this variant of the game, pairs of players collaborate as a team to find the location of ships on the board. "
    "Each player is assigned to one of two roles: the 'Captain' or the 'Spotter'. "
    "The Captain's role is to decide when and where to reveal tiles on the board. On each turn, the Captain can ask the Spotter a question about the board, or make a move by guessing a tile that they think contains a ship. "
    "The Spotter's role is to provide the Captain with information about the hidden tiles. The spotter has full visibility of the board, but can only answer the captain's question with 'Yes' or 'No'."
    "\n"
    "The board is an 8x8 grid, with lettered rows A, B, C, D, E, F, G, H and numbered columns 1, 2, 3, 4, 5, 6, 7, 8. "
    "Coordinates are specified as a row, column pair. For example, C2 is the tile in row C, column 2.\n"
    "There are four ships on the board: Green, Red, Purple, and Orange. "
    "Ships are oriented either horizontally or vertically and range from 2 to 5 tiles in length. "
)

# Board format descriptions
PROMPT_VARIANT_GRID = (
    "The board is represented as a grid with the following symbols:\n\n"
    "W: Water\n"
    "R: Red ship\n"
    "G: Green ship\n"
    "P: Purple ship\n"
    "O: Orange ship\n"
    "H: Hidden\n"
)
PROMPT_VARIANT_GRID_NUMERIC = (
    "The board is represented as a numpy array with the following symbols:\n"
    "-1: Hidden\n"
    "0: Water\n"
    "1: Green ship\n"
    "2: Red ship\n"
    "3: Purple ship\n"
    "4: Orange ship\n"
)
PROMPT_VARIANT_TEXTUAL = "The board is represented as a textual description.\n"
PROMPT_VARIANT_VISUAL = "The board is represented as an image, with light gray indicating hidden tiles, dark gray indicating water tiles, and red, blue and purple indicating ship tiles.\n"

PROMPT_TARGET_BOARD = "Here is the current board:\n"


class BasePrompt(object):
    """Base class for constructing prompts for the Battleship task.

    Each subclass implements its own .to_chat_format() method.
    """

    def __init__(
        self,
        target_trial_id: int = None,
        target_trial_experiment: str = None,
        board_format: BoardFormat = None,
        include_instructions: bool = True,
        include_board: bool = True,
        include_system_prompt: bool = True,
        include_final_prefix: bool = False,
    ):
        self.target_trial_experiment = target_trial_experiment
        self.target_trial_id = target_trial_id
        self.board_format = board_format
        self.include_system_prompt = include_system_prompt
        self.include_board = include_board
        self.include_instructions = include_instructions
        self.include_final_prefix = include_final_prefix

    def __str__(self):
        return "\n".join(
            [
                f"[{message['role']}]{message['content']}"
                for message in self.to_chat_format()
            ]
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
Spotter prompt for the Collaborative Battleship task.
"""

PROMPT_TASK_BASE_SPOTTER = (
    "You are playing as the Spotter. "
    "Your objective is to answer the Captain's questions as accurately as possible. "
)

PROMPT_TASK_DIRECT_SPOTTER = "Remember: You can only answer with 'Yes' or 'No'. Please only answer with a single word. Enclose your answer in <answer></answer> tags, e.g. <answer>Yes</answer> or <answer>No</answer>."

PROMPT_TASK_CODE_SPOTTER = (
    "Your task is to write a Python function that, when executed on the board, returns the answer to the question. "
    "Your function should return a Boolean value, which will be interpreted as 'Yes' or 'No'. "
    "\n\nYour function should be defined as follows: "
    "\n```python"
    "\ndef answer(board: np.ndarray) -> bool:"
    "\n    # Your code here"
    "\n    return ANSWER"
    "\n```"
    "\nYour code will be executed in an environment with `numpy` (namespaced as `np`) and a `board` variable (a numpy representation of the board). "
    "Make sure your code is valid Python and does not contain any syntax errors. "
    "You are responsible for implementing the `answer()` function, but do not invoke it or include any other code. "
)

PROMPT_COT = "Please think step-by-step about the task before returning your answer."

PROMPT_EXAMPLES_MOVE = "Here are the past turns in the game so far:\n"

PROMPT_TARGET_BOARD_SPOTTER = (
    "Here is the full board, which only you as Spotter have access to:\n"
)

QUESTION_PRESENTATION_PROMPT = "Here is the question the Captain asked:\n"


class SpotterPrompt(BasePrompt):
    """Prompt for answering questions for the Battleship task."""

    def __init__(
        self,
        question,
        use_code=False,
        target_occ_tiles=None,
        history=None,
        use_cot=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.question = question
        self.history = history
        self.use_code = use_code
        self.use_cot = use_cot
        self.target_occ_tiles = target_occ_tiles
        if self.board_format is None:
            raise ValueError("Board format must be specified.")

    def to_chat_format(self):
        messages = []

        system_prompt = (
            PROMPT_GAME + PROMPT_VARIANT_GRID_NUMERIC + PROMPT_TASK_BASE_SPOTTER
        )
        if self.use_code:
            system_prompt += PROMPT_TASK_CODE_SPOTTER
        else:
            system_prompt += PROMPT_TASK_DIRECT_SPOTTER

        if self.history:
            system_prompt += PROMPT_EXAMPLES_MOVE
            for example in self.history:
                if example["decision"] == Decision.QUESTION:
                    decision_str = f"Captain (question): {example['question']}\nSpotter (answer): {example['answer']}"
                else:
                    decision_str = (
                        f"Captain (move): {coords_to_tile(example['coords'])}"
                    )
                system_prompt += decision_str + "\n"

        board_str = str(
            Board.from_trial_id(
                self.target_trial_id, self.target_trial_experiment
            ).board
        )

        system_prompt += PROMPT_TARGET_BOARD_SPOTTER + board_str

        system_prompt += QUESTION_PRESENTATION_PROMPT

        if self.use_cot:
            system_prompt += PROMPT_COT

        messages.append({"role": "system", "content": system_prompt})

        messages.append(
            {
                "role": "user",
                "content": self.question.text,
            }
        )

        return messages


"""
Decision prompt for the Collaborative Battleship task.
"""

PROMPT_SYSTEM_DECISION = (
    "You are a game-playing agent. "
    "Read the game instructions and examples carefully. "
    "Respond in one word. "
    "Do not include any other explanation or prose.\n"
)

PROMPT_SYSTEM_DECISION_COT = (
    "You are a game-playing agent. "
    "Read the game instructions and examples carefully.\n"
)

PROMPT_TASK_BASE_DECISION = (
    "You will be given a partially-revealed game board. "
    "Your task is to choose whether you'd like to ask a question about the board to gain more information, or make a move by guessing a tile that you think contains a ship. Please answer in a single word: 'Question' or 'Move'.\n"
)

PROMPT_TASK_BASE_DECISION_COT = (
    "You will be given a partially-revealed game board. "
    "Your task is to choose whether you'd like to ask a question about the board to gain more information, or make a move by guessing a tile that you think contains a ship. Please answer by saying 'Question' or 'Move'."
    "Please think about the task step-by-step, and enclose your final answer in <answer></answer> tags, e.g. <answer>Question</answer> or <answer>Move</answer>.\n"
)

PROMPT_TARGET_BOARD_DECISION = (
    lambda x: f"You can ask {x} more questions over the course of the game. Here's your board: \n"
)


class DecisionPrompt(BasePrompt):
    """Prompt for generating decisions during a game of Battleship."""

    def __init__(
        self,
        target_occ_tiles=None,
        use_cot=False,
        history=None,
        q_remaining=None,
        sunk=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.history = history
        self.q_remaining = q_remaining
        self.target_occ_tiles = target_occ_tiles
        self.use_cot = use_cot
        self.sunk = sunk
        if self.board_format is None:
            raise ValueError("Board format must be specified.")

    def to_chat_format(self):
        messages = []

        if self.include_instructions:
            if self.include_system_prompt:
                if self.use_cot:
                    messages.append(
                        {"role": "system", "content": PROMPT_SYSTEM_DECISION_COT}
                    )
                else:
                    messages.append(
                        {"role": "system", "content": PROMPT_SYSTEM_DECISION}
                    )

            messages.append({"role": "user", "content": PROMPT_GAME})

            if self.use_cot:
                messages.append(
                    {"role": "user", "content": PROMPT_TASK_BASE_DECISION_COT}
                )
            else:
                messages.append({"role": "user", "content": PROMPT_TASK_BASE_DECISION})

            if self.include_board:
                if self.board_format == BoardFormat.GRID:
                    messages.append(
                        {"role": "user", "content": PROMPT_VARIANT_GRID_NUMERIC}
                    )
                elif self.board_format == BoardFormat.TEXTUAL:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_TEXTUAL})
                elif self.board_format == BoardFormat.VISUAL:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_VISUAL})
                else:
                    raise ValueError(f"Unknown board format: {self.board_format}")

        if self.history != []:
            if self.include_instructions:
                messages.append({"role": "user", "content": PROMPT_EXAMPLES_MOVE})

            for example in self.history:
                if example["decision"] == Decision.QUESTION:
                    decision_str = f"""Question: {example["question"].text}\n
                                        Answer: {example["answer"].text}"""
                else:
                    decision_str = f"""Move: {coords_to_tile(example["coords"])}"""

                messages.append(
                    {
                        "role": "user",
                        "content": f""" {decision_str}\n
                                        {self.EXAMPLE_DELIMITER}\n\n""",
                    }
                )

            if self.include_board:
                if self.include_instructions:
                    messages.append(
                        {
                            "role": "user",
                            "content": "\n"
                            + PROMPT_TARGET_BOARD_DECISION(self.q_remaining),
                        }
                    )

                if self.sunk is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Ship Status: {self.sunk}\n",
                        }
                    )

                board_str = self.target_occ_tiles.to_format(self.board_format)
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
        else:
            if self.include_board:
                if self.target_occ_tiles is not None:
                    captain_board_str = self.target_occ_tiles.to_format(
                        self.board_format
                    )
                    if not captain_board_str.endswith("\n"):
                        captain_board_str += "\n"
                    messages.append(
                        {
                            "role": "user",
                            "content": f"{self.EXAMPLE_DELIMITER}\n\n{captain_board_str}",
                        }
                    )

                if self.include_instructions:
                    messages.append(
                        {
                            "role": "user",
                            "content": "\n"
                            + PROMPT_TARGET_BOARD_DECISION(self.q_remaining),
                        }
                    )
                if self.sunk is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Ship Status: {self.sunk}\n",
                        }
                    )

        return messages


"""
Move prompt for the Collaborative Battleship task.
"""

PROMPT_SYSTEM_MOVE = (
    "You are a game-playing agent. "
    "Read the game instructions and examples carefully. "
    "Respond in one word. "
    "Do not include any other explanation or prose.\n"
)

PROMPT_SYSTEM_MOVE_COT = (
    "You are a game-playing agent. "
    "Read the game instructions and examples carefully.\n"
)

PROMPT_TASK_BASE_MOVE = (
    "You will be given a partially-revealed game board. "
    "Your task is to give the coordinates, of the hidden tile you think is most likely to contain a ship tile. "
)

PROMPT_TASK_BASE_MOVE_COT = (
    "You will be given a partially-revealed game board. "
    "Your task is to give the coordinates, of the hidden tile you think is most likely to contain a ship tile. "
    "Please think about the task step-by-step, and enclose your answer in <answer></answer> tags, e.g. <answer>A1</answer>."
)

PROMPT_TARGET_BOARD_MOVE = (
    lambda x: f"Remember that hidden tiles are marked by 'H', and that you have {x} moves left before the game ends. Here's your board: \n"
)


class MovePrompt(BasePrompt):
    """Prompt for generating moves during a game of Battleship."""

    def __init__(
        self,
        target_occ_tiles=None,
        use_cot=False,
        history=None,
        moves_remaining=None,
        sunk=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.history = history
        self.target_occ_tiles = target_occ_tiles
        self.use_cot = use_cot
        self.moves_remaining = moves_remaining
        self.sunk = sunk
        if self.board_format is None:
            raise ValueError("Board format must be specified.")

    def to_chat_format(self):
        messages = []

        if self.include_instructions:
            if self.include_system_prompt:
                if self.use_cot:
                    messages.append(
                        {"role": "system", "content": PROMPT_SYSTEM_MOVE_COT}
                    )
                else:
                    messages.append({"role": "system", "content": PROMPT_SYSTEM_MOVE})

            messages.append({"role": "user", "content": PROMPT_GAME})

            if self.use_cot:
                messages.append({"role": "user", "content": PROMPT_TASK_BASE_MOVE_COT})
            else:
                messages.append({"role": "user", "content": PROMPT_TASK_BASE_MOVE})

            if self.include_board:
                if self.board_format == BoardFormat.GRID:
                    messages.append(
                        {"role": "user", "content": PROMPT_VARIANT_GRID_NUMERIC}
                    )
                elif self.board_format == BoardFormat.TEXTUAL:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_TEXTUAL})
                elif self.board_format == BoardFormat.VISUAL:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_VISUAL})
                else:
                    raise ValueError(f"Unknown board format: {self.board_format}")

        if self.history != []:
            if self.include_instructions:
                messages.append({"role": "user", "content": PROMPT_EXAMPLES_MOVE})

            for example in self.history:
                if example["decision"] == Decision.QUESTION:
                    decision_str = f"""Question: {example["question"].text}\n
                                        Answer: {example["answer"].text}"""
                else:
                    decision_str = f"""Move: {coords_to_tile(example["coords"])}"""

                messages.append(
                    {
                        "role": "user",
                        "content": f""" {decision_str}\n
                                        {self.EXAMPLE_DELIMITER}\n\n""",
                    }
                )
            if self.include_board:
                if self.include_instructions:
                    messages.append(
                        {
                            "role": "user",
                            "content": "\n"
                            + PROMPT_TARGET_BOARD_MOVE(self.moves_remaining),
                        }
                    )

                if self.sunk is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Ship Status: {self.sunk}\n",
                        }
                    )

                board_str = self.target_occ_tiles.to_format(self.board_format)
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
        else:
            if self.include_board:
                if self.target_occ_tiles is not None:
                    captain_board_str = self.target_occ_tiles.to_format(
                        self.board_format
                    )
                    if not captain_board_str.endswith("\n"):
                        captain_board_str += "\n"
                    messages.append(
                        {
                            "role": "user",
                            "content": f"{self.EXAMPLE_DELIMITER}\n\n{captain_board_str}",
                        }
                    )

                if self.include_instructions:
                    messages.append(
                        {
                            "role": "user",
                            "content": "\n"
                            + PROMPT_TARGET_BOARD_MOVE(self.moves_remaining),
                        }
                    )
                if self.sunk is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Ship Status: {self.sunk}\n",
                        }
                    )

        return messages


"""
Question prompt for the Collaborative Battleship task.
"""

PROMPT_SYSTEM_QUESTION = (
    "You are a game-playing agent. "
    "Read the game instructions and examples carefully. "
    "Respond with a single question that can be answered with one word. "
    "Do not include any other explanation or prose.\n"
)

PROMPT_SYSTEM_QUESTION_COT = (
    "You are a game-playing agent. "
    "Read the game instructions and examples carefully.\n"
)

# Task description
PROMPT_TASK_BASE_QUESTION = (
    "You will be given a partially-revealed game board. "
    "Your task is to ask a single question that will help you gain the most information possible about the position of the remaining hidden ships on the board. "
    "You can ask any question, but it must be answerable with a Boolean answer (Yes/No). "
)

PROMPT_TASK_BASE_QUESTION_COT = (
    "You will be given a partially-revealed game board. "
    "Your task is to ask a single question that will help you gain the most information possible about the position of the remaining hidden ships on the board. "
    "You can ask any question, but it must be answerable with a Boolean answer (Yes/No). Please think about this step by step, and, when you've come up with an answer, make sure to enclose it in <answer></answer> tags, e.g. <answer>Is the sky blue?</answer>."
)

PROMPT_EXAMPLES_QUESTION = "Here are the past turns in the game so far, which include what questions have already been asked about the board, and what moves have already been made. Make sure your questions are not similar to each other."

PROMPT_TARGET_BOARD_QUESTION = (
    lambda x: f"Including the one you are about to ask, you have {x} questions remaining. Here's your board: \n"
)

PROMPT_TARGET_BOARD_QUESTION_SEQ = (
    lambda x, y: f"Including the one you are about to ask, you have {x} questions remaining. You've already considered asking the following questions, so make sure to ask something different: {y}. Here's your board: \n"
)


class QuestionPrompt(BasePrompt):
    """Prompt for generating questions for the Battleship task."""

    def __init__(
        self,
        target_occ_tiles=None,
        history=None,
        use_cot=False,
        q_remaining=None,
        sunk=None,
        sequential_questions="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.history = history
        self.target_occ_tiles = target_occ_tiles
        self.use_cot = use_cot
        self.q_remaining = q_remaining
        self.sunk = sunk
        self.sequential_questions = sequential_questions
        if self.board_format is None:
            raise ValueError("Board format must be specified.")

    def to_chat_format(self):
        messages = []

        if self.include_instructions:
            if self.include_system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": PROMPT_SYSTEM_QUESTION
                        if not self.use_cot
                        else PROMPT_SYSTEM_QUESTION_COT,
                    }
                )

            messages.append({"role": "user", "content": PROMPT_GAME})
            messages.append(
                {
                    "role": "user",
                    "content": PROMPT_TASK_BASE_QUESTION
                    if not self.use_cot
                    else PROMPT_TASK_BASE_QUESTION_COT,
                }
            )

            if self.include_board:
                if self.board_format == BoardFormat.GRID:
                    messages.append(
                        {"role": "user", "content": PROMPT_VARIANT_GRID_NUMERIC}
                    )
                elif self.board_format == BoardFormat.TEXTUAL:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_TEXTUAL})
                elif self.board_format == BoardFormat.VISUAL:
                    messages.append({"role": "user", "content": PROMPT_VARIANT_VISUAL})
                else:
                    raise ValueError(f"Unknown board format: {self.board_format}")

        if self.history is not None:
            if self.include_instructions:
                messages.append({"role": "user", "content": PROMPT_EXAMPLES_QUESTION})

            for example in self.history:
                if example["decision"] == Decision.QUESTION:
                    decision_str = f"""Question: {example["question"].text}\n
                                        Answer: {example["answer"].text}"""
                else:
                    decision_str = f"""Move: {coords_to_tile(example["coords"])}"""

                messages.append(
                    {
                        "role": "user",
                        "content": f""" {decision_str}\n
                                        {self.EXAMPLE_DELIMITER}\n\n""",
                    }
                )
            if self.include_board:
                if self.include_instructions:
                    if self.sequential_questions == "":
                        messages.append(
                            {
                                "role": "user",
                                "content": "\n"
                                + PROMPT_TARGET_BOARD_QUESTION(self.q_remaining),
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": "\n"
                                + PROMPT_TARGET_BOARD_QUESTION_SEQ(
                                    self.q_remaining, self.sequential_questions
                                ),
                            }
                        )

                if self.sunk is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Ship Status: {self.sunk}\n",
                        }
                    )

                board_str = self.target_occ_tiles.to_format(self.board_format)
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
        else:
            if self.include_board:
                if self.target_occ_tiles is not None:
                    if self.include_instructions:
                        if self.sequential_questions == "":
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "\n"
                                    + PROMPT_TARGET_BOARD_QUESTION(self.q_remaining),
                                }
                            )
                        else:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "\n"
                                    + PROMPT_TARGET_BOARD_QUESTION_SEQ(
                                        self.q_remaining, self.sequential_questions
                                    ),
                                }
                            )

                    if self.sunk is not None:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Ship Status: {self.sunk}\n",
                            }
                        )

                    board_str = self.target_occ_tiles.to_format(self.board_format)
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

        return messages
