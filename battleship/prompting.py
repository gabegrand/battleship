from typing import Dict
from typing import List

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

    def get_prompt_prefix(self) -> List[dict]:
        messages = []

        system_prompt = PROMPT_GAME + PROMPT_VARIANT_GRID_NUMERIC

        history_messages = []
        if self.history is not None:
            formatted_history = self.format_history()
            if formatted_history != []:
                history_messages.append({"role": "system", "content": PROMPT_EXAMPLES})
                history_messages.extend(formatted_history)

        messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)

        return messages

    def format_history(self) -> List[dict]:
        messages = []
        for example in self.history:
            if example["decision"] == Decision.QUESTION:
                # TODO: Implement dataloader for human data to make this cleaner
                question_text = (
                    example["question"]["text"]
                    if type(example["question"]) != str
                    else example["question"]
                )
                answer_text = (
                    example["answer"]["text"]
                    if type(example["answer"]) != str
                    else example["answer"]
                )

                question = f"{self.CAPTAIN} (question): {question_text}\n"
                answer = f"{self.SPOTTER} (answer): {answer_text}\n"

                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})

            elif example["decision"] == Decision.MOVE:
                # TODO: Implement dataloader for human data to make this cleaner

                if example.get("move"):
                    move_text = str(example["move"])
                else:
                    if example.get("coords"):
                        move_text = str(coords_to_tile(example["coords"]))
                    else:
                        move_text = None

                move = f"{self.CAPTAIN} (move): {move_text}\n"

                messages.append({"role": "user", "content": move})
            else:
                raise ValueError(f"Unknown decision type: {example['decision']}")
        return messages


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
    "Return your answer directly. Do not include any extra reasoning or explanation."
)
PROMPT_COT = "Please think step-by-step about the task before returning your answer."

PROMPT_EXAMPLES = "Here are the past turns in the game so far:\n"

PROMPT_BOARD_CURRENT = "Here is the current board:\n"

PROMPT_BOARD_CAPTAIN = (
    "Here is the partial board, which is the view that is visible to the Captain:\n"
)

PROMPT_BOARD_SPOTTER = (
    "Here is the full board, which only you as Spotter have access to:\n"
)

QUESTION_PRESENTATION_PROMPT = "Here is the question the Captain asked:\n"


class SpotterPrompt(BasePrompt):
    """Prompt for answering questions for the Battleship task."""

    def __init__(
        self,
        question,
        use_code=False,
        board: Board = None,
        history=None,
        use_cot=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.question = question
        self.history = history
        self.use_code = use_code
        self.use_cot = use_cot
        self.board = board
        if self.board_format is None:
            raise ValueError("Board format must be specified.")

    def to_chat_format(self):
        messages = []

        messages_prefix = self.get_prompt_prefix()

        # Captain board (partial)
        board_message = ""
        if self.board is not None:
            board_str = str(self.board.to_numpy())
            board_message += "\n\n" + PROMPT_BOARD_CAPTAIN + board_str

        # Spotter board (true)
        board_str = str(
            Board.from_trial_id(
                self.target_trial_id, self.target_trial_experiment
            ).to_numpy()
        )
        board_message += "\n\n" + PROMPT_BOARD_SPOTTER + board_str

        # Task description
        postfix = PROMPT_TASK_BASE_SPOTTER

        # Code vs. direct answering
        if self.use_code:
            postfix += PROMPT_TASK_CODE_SPOTTER
        else:
            postfix += PROMPT_TASK_DIRECT_SPOTTER

        # Add CoT instruction if needed
        if self.use_cot:
            postfix += "\n" + PROMPT_COT
        else:
            postfix += "\n" + PROMPT_DIRECT

        postfix += "\n" + QUESTION_PRESENTATION_PROMPT

        messages.extend(messages_prefix)
        messages.append({"role": "system", "content": board_message})
        messages.append({"role": "system", "content": postfix})
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

PROMPT_TASK_MOVE = (
    "You will be given a partially-revealed game board. "
    "Your task is to give the coordinates of the hidden tile you think is most likely to contain a ship tile. "
    "Hidden tiles are marked by '-1'. "
    "Respond with only the coordinates (e.g., A1, B2, etc.), and enclose your answer in <answer></answer> tags, e.g. <answer>A1</answer>."
)

PROMPT_TASK_QUESTION = (
    "You will be given a partially-revealed game board. "
    "Your task is to ask {num_questions} question(s) that will help you gain the most information possible about the position of the remaining hidden ships on the board. "
    "You can ask any question(s), but they must be answerable with a Boolean answer (Yes/No). "
    "Make sure to enclose your question(s) in <answer></answer> tags with separate <answer></answer> tags for each question, e.g. <answer>Is the ship in A1 vertical?</answer>, <answer>Is the ship in B2 horizontal?</answer>."
)

PROMPT_QUESTIONS_AND_MOVES_REMAINING = "You can ask {q_remaining} more questions over the course of the game, and can fire {moves_remaining} more times."

PROMPT_QUESTIONS_AND_MOVES_REMAINING_BATCHED = "You can ask {q_remaining} more batch(es) of questions over the course of the game, and can fire {moves_remaining} more times."

PROMPT_SHIP_STATUS = "Ship Status: {sunk}"


class CaptainPrompt(BasePrompt):
    """Prompt for generating decisions during a game of Battleship."""

    def __init__(
        self,
        board=None,
        use_cot=False,
        history=None,
        questions_remaining=None,
        moves_remaining=None,
        sunk=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.history = history
        self.questions_remaining = questions_remaining
        self.moves_remaining = moves_remaining
        self.board = board
        self.use_cot = use_cot
        self.sunk = sunk

        self.task_prompt = None

        if self.board_format is None:
            raise ValueError("Board format must be specified.")

    def to_chat_format(self):
        messages = []

        messages_prefix = self.get_prompt_prefix()

        # Board state
        board_str = str(self.board.to_numpy()) if self.board else ""
        board_message = "\n\n" + PROMPT_BOARD_CURRENT + board_str

        # Task description
        if hasattr(self, 'n_questions'):
            self.task_prompt = self.task_prompt.format(num_questions=self.n_questions)

        postfix = PROMPT_SYSTEM_CAPTAIN + "\n\n" + self.task_prompt

        # Qs and moves remaining, ship tracker
        if hasattr(self, 'n_questions'):
            postfix += "\n\n" + PROMPT_QUESTIONS_AND_MOVES_REMAINING_BATCHED.format(
                q_remaining=self.questions_remaining, moves_remaining=self.moves_remaining
            )
        else:
            postfix += "\n\n" + PROMPT_QUESTIONS_AND_MOVES_REMAINING.format(
                q_remaining=self.questions_remaining, moves_remaining=self.moves_remaining
            )

        postfix += "\n" + PROMPT_SHIP_STATUS.format(sunk=self.sunk)

        # Add CoT instruction if needed
        if self.use_cot:
            postfix += "\n" + PROMPT_COT
        else:
            postfix += "\n" + PROMPT_DIRECT

        messages.extend(messages_prefix)
        messages.append({"role": "system", "content": board_message})
        messages.append({"role": "system", "content": postfix})

        return messages


class DecisionPrompt(CaptainPrompt):
    """Prompt for generating decisions during a game of Battleship."""

    def __init__(
        self,
        board=None,
        use_cot=False,
        history=None,
        questions_remaining=None,
        moves_remaining=None,
        sunk=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.history = history
        self.questions_remaining = questions_remaining
        self.moves_remaining = moves_remaining
        self.board = board
        self.use_cot = use_cot
        self.sunk = sunk

        self.task_prompt = PROMPT_TASK_DECISION

        if self.board_format is None:
            raise ValueError("Board format must be specified.")


class MovePrompt(CaptainPrompt):
    """Prompt for generating decisions during a game of Battleship."""

    def __init__(
        self,
        board=None,
        use_cot=False,
        history=None,
        questions_remaining=None,
        moves_remaining=None,
        sunk=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.history = history
        self.questions_remaining = questions_remaining
        self.moves_remaining = moves_remaining
        self.board = board
        self.use_cot = use_cot
        self.sunk = sunk

        self.task_prompt = PROMPT_TASK_MOVE

        if self.board_format is None:
            raise ValueError("Board format must be specified.")


class QuestionPrompt(CaptainPrompt):
    """Prompt for generating decisions during a game of Battleship."""

    def __init__(
        self,
        board=None,
        use_cot=False,
        history=None,
        questions_remaining=None,
        moves_remaining=None,
        sunk=None,
        n_questions=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.history = history
        self.questions_remaining = questions_remaining
        self.moves_remaining = moves_remaining
        self.board = board
        self.use_cot = use_cot
        self.sunk = sunk
        self.n_questions = n_questions

        self.task_prompt = PROMPT_TASK_QUESTION

        if self.board_format is None:
            raise ValueError("Board format must be specified.")
