from typing import Dict
from typing import List

import numpy as np

from battleship.board import Board
from battleship.board import BoardFormat
from battleship.board import coords_to_tile
from battleship.game import Decision

PROMPT_GAME = (
    "You are playing the board game Battleship. In this variant of the game, pairs of players collaborate as a team to find the location of ships on the board. "
    "Each player is assigned to one of two roles: the 'Captain' or the 'Spotter'. "
    "The Captain's role is to decide when and where to reveal tiles on the board. On each turn, the Captain can ask the Spotter a question about the board, or make a move by guessing a tile that they think contains a ship. "
    "The Spotter's role is to provide the Captain with information about the hidden tiles. The Spotter has full visibility of the board, but can only answer the Captain's questions with 'Yes' or 'No'."
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
    "1: Red ship\n"
    "2: Green ship\n"
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

    CAPTAIN = "Captain"
    SPOTTER = "Spotter"

    def __init__(
        self,
        target_trial_id: int = None,
        target_trial_experiment: str = None,
        board_format: BoardFormat = None,
        history: List[Dict] = None,
    ):
        self.target_trial_experiment = target_trial_experiment
        self.target_trial_id = target_trial_id
        self.board_format = board_format
        self.history = history

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

    def format_history(self) -> str:
        text = ""
        for example in self.history:
            if example["decision"] == Decision.QUESTION:
                # TODO: Implement dataloader for human data to make this cleaner
                question_text = (
                    example["question"].text
                    if type(example["question"]) != str
                    else example["question"]
                )
                answer_text = (
                    example["answer"].text
                    if type(example["answer"]) != str
                    else example["answer"]
                )

                text += f"{self.CAPTAIN} (question): {question_text}\n"
                text += f"{self.SPOTTER} (answer): {answer_text}\n"

            elif example["decision"] == Decision.MOVE:
                # TODO: Implement dataloader for human data to make this cleaner
                move_text = (
                    str(example["move"])
                    if example.get("move")
                    else str(coords_to_tile(example["coords"]))
                )

                text += f"{self.CAPTAIN} (move): {move_text}\n"
            else:
                raise ValueError(f"Unknown decision type: {example['decision']}")
        return text


"""
Spotter prompt for the Collaborative Battleship task.
"""

PROMPT_TASK_BASE_SPOTTER = (
    "You are playing as the Spotter. "
    "Your objective is to answer the Captain's questions as accurately as possible. "
)

PROMPT_TASK_DIRECT_SPOTTER = "Remember: You can only answer with 'Yes' or 'No'. Please only answer with a single word. Enclose your answer in <answer></answer> tags, e.g. <answer>Yes</answer> or <answer>No</answer>."

PROMPT_TASK_CODE_SPOTTER = (
    "Your task is to write a Python function that computes the answer to the question. "
    "\n\nThe function should accept two numpy arrays as arguments: `true_board` and `partial_board`. "
    "\nThe `true_board` is the full board, which is only visible to you as the Spotter. "
    "\nThe `partial_board` is the current board that is visible to the captain, which may contain hidden tiles. This represents the current state of the game that the Captain is asking about. "
    "Your function should return a Boolean value, which will be interpreted as 'Yes' or 'No'. "
    "\n\nYour function should be defined generically to work with *any* true and partial board, not just the ones you are given. This means that your function must perform some operations on `true_board`, `partial_board`, or both boards in order to compute the answer to the Captain's question. Avoid hardcoding the answer. "
    "In some situations, the correct answer may depend on the current state of the game. For instance, if the Captain asks, 'Are there any ships in Row A?', the answer depends on what ships have already been revealed. If there are any unrevealed ship tiles in Row A, then the answer is 'Yes'. However, if all ships in Row A have already been revealed, then the correct answer is 'No'. Comparing the `partial_board` with the `true_board` will allow you to determine which ship tiles remain unrevealed. Remember: Your goal is to help the Captain find the location of the ships on the board, so your function should be designed to provide the useful information in context. "
    "\n\nYour function should be defined as follows:"
    "\n```python"
    "\ndef answer(true_board: np.ndarray, partial_board: np.ndarray) -> bool:"
    "\n    # Your code here"
    "\n    return ANSWER"
    "\n```"
    "\n\nYour code will be executed in an environment with `numpy` (namespaced as `np`) and a `board` variable (a numpy representation of the board). "
    "Make sure your code is valid Python and does not contain any syntax errors. "
    "You are responsible for implementing the `answer()` function, but do not invoke it or include any other code. "
)

PROMPT_DIRECT = (
    "Return your answer directly. Do not include any extra reasoning or prose."
)
PROMPT_COT = "Please think step-by-step about the task before returning your answer."

PROMPT_EXAMPLES_MOVE = "Here are the past turns in the game so far:\n"

PROMPT_TARGET_BOARD_CAPTAIN = (
    "Here is the partial board, which is the view that is visible to the Captain:\n"
)

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

        # Basic Spotter instructions
        system_prompt = (
            PROMPT_GAME + PROMPT_VARIANT_GRID_NUMERIC + PROMPT_TASK_BASE_SPOTTER
        )
        # Code vs. direct answering
        if self.use_code:
            system_prompt += PROMPT_TASK_CODE_SPOTTER
        else:
            system_prompt += PROMPT_TASK_DIRECT_SPOTTER

        # Game history
        if self.history is not None:
            system_prompt += "\n\n" + PROMPT_EXAMPLES_MOVE
            system_prompt += self.format_history()

        # Captain board (partial)
        if self.target_occ_tiles is not None:
            board_str = str(Board.from_occ_tiles(self.target_occ_tiles).to_numpy())
            system_prompt += "\n\n" + PROMPT_TARGET_BOARD_CAPTAIN + board_str

        # Spotter board (true)
        board_str = str(
            Board.from_trial_id(
                self.target_trial_id, self.target_trial_experiment
            ).to_numpy()
        )
        system_prompt += "\n\n" + PROMPT_TARGET_BOARD_SPOTTER + board_str

        # Chain-of-thought prompt (optional)
        if self.use_cot:
            system_prompt += "\n\n" + PROMPT_COT
        else:
            system_prompt += "\n\n" + PROMPT_DIRECT

        system_prompt += "\n\n" + QUESTION_PRESENTATION_PROMPT

        messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": f"{self.CAPTAIN} (question): {self.question.text}\n",
            }
        )

        return messages


# ====================
# Decision Prompt Constants
# ====================

PROMPT_SYSTEM_CAPTAIN = (
    "You are playing as the Captain. "
    "Your objective is to find all the ships on the board as efficiently as possible. "
)

PROMPT_TASK_DECISION = (
    "You will be given a partially-revealed game board. "
    "Your task is to choose whether you'd like to ask a question about the board to gain more information, or make a move by guessing a tile that you think contains a ship. "
    "Please answer in a single word: 'Question' or 'Move', and enclose your final answer in <answer></answer> tags, e.g. <answer>Question</answer> or <answer>Move</answer>."
)

PROMPT_QUESTIONS_REMAINING = (
    "You can ask {q_remaining} more questions over the course of the game."
)

PROMPT_SHIP_STATUS = "Ship Status: {sunk}"

PROMPT_CURRENT_BOARD = "Here's your board:"


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

        # System prompt
        system_prompt = (
            PROMPT_GAME + PROMPT_VARIANT_GRID_NUMERIC + PROMPT_SYSTEM_CAPTAIN
        )

        system_prompt += PROMPT_TASK_DECISION

        # Game history
        if self.history is not None:
            system_prompt += "\n\n" + PROMPT_EXAMPLES_MOVE
            system_prompt += self.format_history()

        # Current game state
        system_prompt += "\n\n" + PROMPT_QUESTIONS_REMAINING.format(
            q_remaining=self.q_remaining
        )

        if self.sunk is not None:
            system_prompt += "\n" + PROMPT_SHIP_STATUS.format(sunk=self.sunk)

        # Board state
        board_str = (
            "1, 2, 3, 4, 5, 6, 7, 8\n" + str(self.target_occ_tiles.to_numpy())
            if self.target_occ_tiles
            else ""
        )
        system_prompt += "\n\n" + PROMPT_CURRENT_BOARD + board_str

        # Add CoT instruction if needed
        if self.use_cot:
            system_prompt += "\n" + PROMPT_COT
        else:
            system_prompt += "\n" + PROMPT_DIRECT

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"{self.CAPTAIN} (decision): "})

        # print(messages)
        return messages


# ====================
# Move Prompt Constants
# ====================

PROMPT_TASK_MOVE = (
    "You will be given a partially-revealed game board. "
    "Your task is to give the coordinates of the hidden tile you think is most likely to contain a ship tile. "
    "Hidden tiles are marked by '-1'. "
    "Respond with only the coordinates (e.g., A1, B2, etc.), and enclose your answer in <answer></answer> tags, e.g. <answer>A1</answer>."
)

PROMPT_MOVES_REMAINING = "You have {moves_remaining} moves left before the game ends."


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

        # System prompt
        system_prompt = (
            PROMPT_GAME + PROMPT_VARIANT_GRID_NUMERIC + PROMPT_SYSTEM_CAPTAIN
        )

        system_prompt += PROMPT_TASK_MOVE

        # Game history
        if self.history:
            system_prompt += "\n\n" + PROMPT_EXAMPLES_MOVE
            system_prompt += self.format_history()

        # Current game state
        system_prompt += "\n\n" + PROMPT_MOVES_REMAINING.format(
            moves_remaining=self.moves_remaining
        )

        if self.sunk is not None:
            system_prompt += "\n" + PROMPT_SHIP_STATUS.format(sunk=self.sunk)

        # Board state
        board_str = (
            "1, 2, 3, 4, 5, 6, 7, 8\n" + str(self.target_occ_tiles.to_numpy())
            if self.target_occ_tiles
            else ""
        )
        system_prompt += "\n\n" + PROMPT_CURRENT_BOARD + board_str

        # Add CoT instruction if needed
        if self.use_cot:
            system_prompt += "\n" + PROMPT_COT
        else:
            system_prompt += "\n" + PROMPT_DIRECT

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"{self.CAPTAIN} (move): "})

        # print(messages)
        return messages


# ====================
# Question Prompt Constants
# ====================

PROMPT_TASK_QUESTION = (
    "You will be given a partially-revealed game board. "
    "Your task is to ask a single question that will help you gain the most information possible about the position of the remaining hidden ships on the board. "
    "You can ask any question, but it must be answerable with a Boolean answer (Yes/No). "
    "Make sure to enclose your question in <answer></answer> tags, e.g. <answer>Is the sky blue?</answer>."
)

# PROMPT_TASK_QUESTION_COT = (
#     "You will be given a partially-revealed game board. "
#     "Your task is to ask a single question that will help you gain the most information possible about the position of the remaining hidden ships on the board. "
#     "You can ask any question, but it must be answerable with a Boolean answer (Yes/No). "
#     "Please think about this step by step, and when you've come up with an answer, "
# )

PROMPT_EXAMPLES_QUESTION = (
    "Here are the past turns in the game so far, which include what questions have already been asked about the board, and what moves have already been made. "
    "Make sure your questions are not similar to each other."
)

PROMPT_QUESTIONS_REMAINING_QUESTION = "Including the one you are about to ask, you have {q_remaining} questions remaining."

PROMPT_SEQUENTIAL_QUESTIONS = "You've already considered asking the following questions, so make sure to ask something different: {sequential_questions}."


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

        # System prompt
        system_prompt = (
            PROMPT_GAME + PROMPT_VARIANT_GRID_NUMERIC + PROMPT_SYSTEM_CAPTAIN
        )

        system_prompt += PROMPT_TASK_QUESTION

        # Game history
        if self.history:
            system_prompt += "\n\n" + PROMPT_EXAMPLES_QUESTION
            system_prompt += self.format_history()

        # Current game state
        system_prompt += "\n\n" + PROMPT_QUESTIONS_REMAINING_QUESTION.format(
            q_remaining=self.q_remaining
        )

        if self.sequential_questions:
            system_prompt += "\n" + PROMPT_SEQUENTIAL_QUESTIONS.format(
                sequential_questions=self.sequential_questions
            )

        if self.sunk is not None:
            system_prompt += "\n" + PROMPT_SHIP_STATUS.format(sunk=self.sunk)

        # Board state
        board_str = (
            "1, 2, 3, 4, 5, 6, 7, 8\n" + str(self.target_occ_tiles.to_numpy())
            if self.target_occ_tiles
            else ""
        )
        system_prompt += "\n\n" + PROMPT_CURRENT_BOARD + board_str

        # Add CoT instruction if needed
        if self.use_cot:
            system_prompt += "\n" + PROMPT_COT
        else:
            system_prompt += "\n" + PROMPT_DIRECT

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"{self.CAPTAIN} (question): "})

        # print(messages)
        return messages
