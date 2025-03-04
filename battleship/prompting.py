import numpy as np
from board import Board
from board import BoardFormat

PROMPT_GAME = (
    "You are playing the board game Battleship. "
    "There are four ships on the board: Red, Green, Purple, and Orange. "
    "Ships are oriented either horizontally or vertically and can be 2, 3, 4, or 5 tiles in length. "
    "The board is an 8x8 grid, with numbered columns 1, 2, 3, 4, 5, 6, 7, 8 and lettered rows A, B, C, D, E, F, G, H. "
    "Coordinates are specified as a row, column pair. For example, C2 is the tile in row C, column 2.\n"
)

# Board format descriptions
PROMPT_VARIANT_GRID = (
    "The board is represented as a grid with the following symbols:\n\n"
    "W: Water\n"
    "R: Red ship\n"
    "G: Green ship\n"
    "P: Purple ship\n"
    "O: Orange ship\n"
)
PROMPT_VARIANT_TEXTUAL = "The board is represented as a textual description.\n"
PROMPT_VARIANT_VISUAL = "The board is represented as an image, with light gray indicating hidden tiles, dark gray indicating water tiles, and red, blue and purple indicating ship tiles.\n"

PROMPT_TARGET_BOARD = "Now, it's your turn. Here is your board:\n"


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
        include_instructions: bool = True,
        include_board: bool = True,
        include_system_prompt: bool = True,
        include_final_prefix: bool = True,
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

PROMPT_TASK_CODE_SPOTTER = "Please generate a piece of numpy code that answers the question. Make sure your code returns 'Yes' or 'No'. This code will be executed in an environment with both numpy (namespaced as np) and a 'board' variable (a numpy representation of the board) so give some function answer(board) that could be used across any board. Don't add any tickmarks or other formatting to denote that this is code, and do not invoke 'python' in the CLI."

PROMPT_TASK_DIRECT_SPOTTER = (
    "You can only answer with 'Yes' or 'No'. Please only answer with a single word."
)

# Example questions
PROMPT_EXAMPLES_SPOTTER = "Here are the past turns in the game so far, which include what questions the user has asked, their answers, and what moves the user made."

PROMPT_PARTIAL_BOARD = "Here's what the captain could see when they asked this question: in this representation, 'H' stands for 'hidden', tiles the player has not fired at yet, which could be water or ship tiles."

PROMPT_TARGET_BOARD_SPOTTER = "Now it's your turn. Here's the fully-revealed board, that the captain did not have access to when asking their question: \n"

QUESTION_PRESENTATION_PROMPT = "Here is the question the captain asked: "


class SpotterPrompt(BasePrompt):
    """Prompt for generating questions for the Battleship task."""

    PREFIX_QUESTION = "Question:"

    def __init__(
        self,
        question,
        use_code=False,
        target_occ_tiles=None,
        history=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.question = question
        self.history = history
        self.use_code = use_code
        self.target_occ_tiles = target_occ_tiles
        if self.board_format is None:
            raise ValueError("Board format must be specified.")

    def to_chat_format(self):
        messages = []

        if self.include_instructions:
            if self.include_system_prompt:
                messages.append({"role": "system", "content": PROMPT_SYSTEM_SPOTTER})

            messages.append({"role": "user", "content": PROMPT_GAME})
            messages.append({"role": "user", "content": PROMPT_TASK_BASE_SPOTTER})

            if self.use_code:
                messages.append({"role": "user", "content": PROMPT_TASK_CODE_SPOTTER})
            else:
                messages.append({"role": "user", "content": PROMPT_TASK_DIRECT_SPOTTER})

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
                    messages.append(
                        {"role": "user", "content": "\n" + PROMPT_TARGET_BOARD}
                    )

                board_str = Board.from_trial_id(
                    self.target_trial_id, self.target_trial_experiment
                ).to_format(self.board_format)
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
        else:
            if self.include_board:
                if self.target_occ_tiles is not None:
                    if self.include_instructions:
                        messages.append(
                            {"role": "user", "content": "\n" + PROMPT_PARTIAL_BOARD}
                        )

                    captain_board_str = Board(
                        np.array(eval(self.target_occ_tiles))
                    ).to_format(self.board_format)
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
                        {"role": "user", "content": "\n" + PROMPT_TARGET_BOARD}
                    )

                board_str = Board.from_trial_id(
                    self.target_trial_id, self.target_trial_experiment
                ).to_format(self.board_format)
                if not board_str.endswith("\n"):
                    board_str += "\n"
                messages.append(
                    {
                        "role": "user",
                        "content": f"{self.EXAMPLE_DELIMITER}\n\n{board_str}",
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

        return messages
