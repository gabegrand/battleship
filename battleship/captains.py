from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from random import random
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from battleship.agents import Agent
from battleship.agents import ANSWER_MATCH_PATTERN
from battleship.agents import DECISION_PATTERN
from battleship.agents import EIGCalculator
from battleship.agents import get_openai_client
from battleship.agents import MOVE_PATTERN
from battleship.agents import Prompt
from battleship.agents import Question
from battleship.board import Board
from battleship.board import coords_to_tile
from battleship.board import tile_to_coords
from battleship.fast_sampler import FastSampler
from battleship.game import Decision
from battleship.prompting import DecisionPrompt
from battleship.prompting import MovePrompt
from battleship.prompting import QuestionPrompt


# Strategy interfaces
class DecisionStrategy(ABC):
    @abstractmethod
    def make_decision(
        self, state, history, questions_remaining, moves_remaining, sunk
    ) -> Decision:
        pass


class MoveStrategy(ABC):
    @abstractmethod
    def make_move(
        self, state, history, sunk, questions_remaining, moves_remaining, constraints
    ) -> Tuple[Tuple[int, int], Dict]:
        pass


class QuestionStrategy(ABC):
    @abstractmethod
    def ask_question(self, state, history, sunk) -> Tuple[Question, Dict]:
        pass


class Captain(Agent):
    def __init__(
        self,
        decision_strategy=None,
        move_strategy=None,
        question_strategy=None,
        seed: int = None,
        model_string=None,
        temperature=None,
        round_id=None,
        stage_dir=None,
        prompts_dir=None,
    ):
        super().__init__(
            seed=seed,
            model_string=model_string,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )
        self.temperature = temperature
        self.sampling_constraints = []

        # Optional strategies for modular approach
        self.round_id = round_id
        self.decision_strategy = decision_strategy
        self.move_strategy = move_strategy
        self.question_strategy = question_strategy

    def decision(
        self,
        state: Board,
        history: List[Dict],
        questions_remaining: int,
        moves_remaining: int,
        sunk: str,
    ) -> Decision:
        self.decision_counter.increment_counter()
        self.questions_remaining = questions_remaining
        self.moves_remaining = moves_remaining
        return self.decision_strategy.make_decision(
            state,
            history,
            questions_remaining,
            moves_remaining,
            sunk,
        )

    def move(self, state: Board, history: List[Dict], sunk: str, constraints: List):
        return self.move_strategy.make_move(
            state,
            history,
            sunk,
            self.questions_remaining,
            self.moves_remaining,
            constraints,
        )

    def question(self, state: Board, history: List[Dict], sunk: str):
        return self.question_strategy.ask_question(
            state, history, sunk, self.questions_remaining, self.moves_remaining
        )


# Example decision strategies
class AlwaysMoveDecisionStrategy(DecisionStrategy):
    def make_decision(self, state, history, questions_remaining, moves_remaining, sunk):
        return Decision.MOVE


class ProbabilisticDecisionStrategy(DecisionStrategy):
    def __init__(self, q_prob=0.5):
        self.q_prob = q_prob

    def make_decision(self, state, history, questions_remaining, moves_remaining, sunk):
        if random() < self.q_prob and questions_remaining > 0:
            return Decision.QUESTION
        return Decision.MOVE


class LLMDecisionStrategy(DecisionStrategy):
    def __init__(self, model_string, temperature=None, use_cot=False):
        self.model_string = model_string
        self.temperature = temperature
        self.use_cot = use_cot
        self.client = get_openai_client()

    def make_decision(
        self, state, history, questions_remaining, moves_remaining, sunk, n_attempts=3
    ):
        if questions_remaining > 0:
            decision_prompt = DecisionPrompt(
                target_occ_tiles=state,
                board_format="grid",
                history=history,
                use_cot=self.use_cot,
                questions_remaining=questions_remaining,
                moves_remaining=moves_remaining,
                sunk=sunk,
            )

            candidate_decision = None
            for _ in range(n_attempts):
                completion = self.client.chat.completions.create(
                    model=self.model_string,
                    messages=decision_prompt.to_chat_format(),
                    temperature=self.temperature,
                )
                match = DECISION_PATTERN.search(completion.choices[0].message.content)

                if match is not None:
                    candidate_decision = match.group(1)
                    break

            return Decision.MOVE if candidate_decision == "Move" else Decision.QUESTION
        return Decision.MOVE


# Example move strategies
class RandomMoveStrategy(MoveStrategy):
    def __init__(self, rng):
        self.rng = rng

    def make_move(
        self, state, history, sunk, questions_remaining, moves_remaining, constraints
    ):
        hidden_tiles = np.argwhere(state.board == Board.hidden)
        if len(hidden_tiles) == 0:
            raise ValueError("No hidden tiles left.")
        coords = self.rng.choice(hidden_tiles)
        return tuple(coords)


class MAPMoveStrategy(MoveStrategy):
    def __init__(self, rng, board_id, n_samples=1000):
        self.rng = rng
        self.board_id = board_id
        self.n_samples = n_samples

    def make_move(
        self, state, history, sunk, questions_remaining, moves_remaining, constraints
    ):
        sampler = FastSampler(
            board=state,
            ship_lengths=Board.SHIP_LENGTHS,
            ship_labels=Board.SHIP_LABELS,
            seed=self.rng,
        )

        if constraints != []:
            true_board = Board.from_trial_id(trial_id=self.board_id).to_numpy()

            posterior = sampler.constrained_posterior(
                ground_truth=true_board,
                n_samples=self.n_samples,
                normalize=False,
                constraints=constraints,
            )
        else:
            # Compute the raw posterior counts over board positions
            posterior = sampler.compute_posterior(
                n_samples=self.n_samples,
                normalize=False,
            )

        # For tiles that have already been revealed, force their probability to -infinity
        posterior = posterior.astype(float)
        posterior[state.board != Board.hidden] = -np.inf

        # Select the tile with the maximum posterior probability (MAP estimate)
        flat_idx = int(np.argmax(posterior))

        # Map the flat index back to 2D coordinates
        move_coords = np.unravel_index(flat_idx, state.board.shape)
        return tuple(move_coords)


class LLMMoveStrategy(MoveStrategy):
    def __init__(
        self,
        model_string,
        temperature=None,
        use_cot=False,
        moves_remaining=None,
        n_attempts=3,
    ):
        self.model_string = model_string
        self.temperature = temperature
        self.use_cot = use_cot
        self.moves_remaining = moves_remaining
        self.n_attempts = n_attempts
        self.client = get_openai_client()

    def make_move(
        self, state, history, sunk, questions_remaining, moves_remaining, constraints
    ):
        visible_tiles = list(zip(*np.where(state.board != Board.hidden)))

        move_prompt = MovePrompt(
            target_occ_tiles=state,
            board_format="grid",
            history=history,
            use_cot=self.use_cot,
            questions_remaining=questions_remaining,
            moves_remaining=moves_remaining,
            sunk=sunk,
        )

        for _ in range(self.n_attempts):
            completion = self.client.chat.completions.create(
                model=self.model_string,
                messages=move_prompt.to_chat_format(),
                temperature=self.temperature,
            )

            candidate_move = MOVE_PATTERN(state.size).search(
                completion.choices[0].message.content
            )
            if candidate_move is not None:
                candidate_move = tile_to_coords(candidate_move.group(1))
                if candidate_move not in visible_tiles:
                    return candidate_move

        return None


# Example question strategies
class EIGQuestionStrategy(QuestionStrategy):
    def __init__(
        self,
        model_string,
        spotter,
        rng,
        samples=100,
        k=3,
        use_cot=False,
        questions_remaining=None,
        n_attempts=3,
    ):
        self.model_string = model_string
        self.spotter = spotter
        self.rng = rng
        self.samples = samples
        self.k = k
        self.use_cot = use_cot
        self.questions_remaining = questions_remaining
        self.eig_calculator = EIGCalculator(seed=self.rng, spotter=self.spotter)
        self.n_attempts = n_attempts
        self.client = get_openai_client()

    def ask_question(self, state, history, sunk, questions_remaining, moves_remaining):
        best_question = None
        best_eig = -1

        for _ in range(self.k):
            question_prompt = QuestionPrompt(
                target_occ_tiles=state,
                board_format="grid",
                history=history,
                use_cot=self.use_cot,
                questions_remaining=questions_remaining,
                moves_remaining=moves_remaining,
                sunk=sunk,
            )

            candidate_question_text = None
            for _ in range(self.n_attempts):
                completion = self.client.chat.completions.create(
                    model=self.model_string,
                    messages=question_prompt.to_chat_format(),
                    temperature=None,
                )
                match = ANSWER_MATCH_PATTERN.search(
                    completion.choices[0].message.content
                )
                if match:
                    candidate_question_text = match.group(1)
                    break

            if candidate_question_text is None:
                continue

            candidate_question = Question(text=candidate_question_text)

            # Calculate EIG for this question
            eig = self.eig_calculator.calculate_eig(
                candidate_question, state, samples=self.samples
            )

            # Update best question if this one has higher EIG
            if eig > best_eig:
                best_eig = eig
                best_question = candidate_question

        return best_question


class LLMQuestionStrategy(QuestionStrategy):
    def __init__(
        self,
        model_string,
        temperature=None,
        use_cot=False,
        spotter=None,
        rng=None,
        questions_remaining=None,
        n_attempts=3,
    ):
        self.model_string = model_string
        self.temperature = temperature
        self.use_cot = use_cot
        self.questions_remaining = questions_remaining
        self.n_attempts = n_attempts
        self.spotter = spotter
        self.rng = rng
        self.eig_calculator = EIGCalculator(seed=self.rng, spotter=self.spotter)
        self.client = get_openai_client()

    def ask_question(self, state, history, sunk, questions_remaining, moves_remaining):
        question_prompt = QuestionPrompt(
            target_occ_tiles=state,
            board_format="grid",
            history=history,
            use_cot=self.use_cot,
            questions_remaining=questions_remaining,
            moves_remaining=moves_remaining,
            sunk=sunk,
        )

        for _ in range(self.n_attempts):
            completion = self.client.chat.completions.create(
                model=self.model_string,
                messages=question_prompt.to_chat_format(),
                temperature=self.temperature,
            )

            candidate_question = ANSWER_MATCH_PATTERN.search(
                completion.choices[0].message.content
            )

            if candidate_question:
                candidate_question = candidate_question.group(1)
                break

        # eig = self.eig_calculator.calculate_eig(
        #     Question(text=candidate_question), state, samples=1000
        # )

        return Question(text=candidate_question)


def create_captain(
    captain_type,
    seed,
    model,
    board_id,
    map_samples=None,
    prob_q_prob=None,
    eig_samples=None,
    eig_k=None,
    round_id=None,
    stage_dir=None,
    prompts_dir=None,
):
    """
    Factory function to create Captain instances with properly configured strategies.
    """
    from battleship.spotters import CodeSpotterModel

    # Initialize spotter for EIG captains
    def _get_spotter():
        return CodeSpotterModel(
            board_id=board_id,
            board_experiment="collaborative",
            model_string=model,
            temperature=None,
            use_cot=True,
        )

    if captain_type == "RandomCaptain":
        return Captain(
            decision_strategy=AlwaysMoveDecisionStrategy(),
            move_strategy=RandomMoveStrategy(rng=np.random.default_rng(seed)),
            question_strategy=None,
            seed=seed,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )

    elif captain_type == "MAPCaptain":
        return Captain(
            decision_strategy=AlwaysMoveDecisionStrategy(),
            move_strategy=MAPMoveStrategy(
                rng=np.random.default_rng(seed),
                board_id=board_id,
                n_samples=map_samples,
            ),
            question_strategy=None,
            seed=seed,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )

    elif captain_type == "ProbabilisticCaptain":
        captain = Captain(
            decision_strategy=ProbabilisticDecisionStrategy(q_prob=prob_q_prob),
            move_strategy=LLMMoveStrategy(model_string=model, use_cot=False),
            question_strategy=LLMQuestionStrategy(model_string=model, use_cot=False),
            seed=seed,
            model_string=model,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )
        return captain

    elif captain_type == "ProbabilisticCaptain_cot":
        captain = Captain(
            decision_strategy=ProbabilisticDecisionStrategy(q_prob=prob_q_prob),
            move_strategy=LLMMoveStrategy(
                model_string=model, use_cot=True, client=None
            ),
            question_strategy=LLMQuestionStrategy(
                model_string=model, use_cot=True, client=None
            ),
            seed=seed,
            model_string=model,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )
        return captain

    elif captain_type == "LLMDecisionCaptain":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=False),
            move_strategy=LLMMoveStrategy(
                model_string=model,
                use_cot=False,
            ),
            question_strategy=LLMQuestionStrategy(
                model_string=model,
                use_cot=False,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
            ),
            seed=seed,
            model_string=model,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )
        return captain

    elif captain_type == "LLMDecisionCaptain_cot":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=True),
            move_strategy=LLMMoveStrategy(model_string=model, use_cot=True),
            question_strategy=LLMQuestionStrategy(
                model_string=model,
                use_cot=True,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
            ),
            seed=seed,
            model_string=model,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )
        return captain

    elif captain_type == "EIGCaptain":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=False),
            move_strategy=LLMMoveStrategy(model_string=model, use_cot=False),
            question_strategy=EIGQuestionStrategy(
                model_string=model,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=False,
            ),
            seed=seed,
            model_string=model,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )
        return captain

    elif captain_type == "EIGCaptain_cot":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                model_string=model, use_cot=True, client=None
            ),
            move_strategy=LLMMoveStrategy(
                model_string=model, use_cot=True, client=None
            ),
            question_strategy=EIGQuestionStrategy(
                model_string=model,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=True,
            ),
            seed=seed,
            model_string=model,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )
        return captain

    elif captain_type == "MAPEIGCaptain":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=False),
            move_strategy=MAPMoveStrategy(
                rng=np.random.default_rng(seed),
                board_id=board_id,
                n_samples=eig_samples,
            ),
            question_strategy=EIGQuestionStrategy(
                model_string=model,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=False,
            ),
            seed=seed,
            model_string=model,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )
        return captain

    elif captain_type == "MAPEIGCaptain_cot":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(model_string=model, use_cot=True),
            move_strategy=MAPMoveStrategy(
                rng=np.random.default_rng(seed),
                board_id=board_id,
                n_samples=eig_samples,
            ),
            question_strategy=EIGQuestionStrategy(
                model_string=model,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=True,
            ),
            seed=seed,
            model_string=model,
            round_id=round_id,
            stage_dir=stage_dir,
            prompts_dir=prompts_dir,
        )
        return captain

    else:
        raise ValueError(f"Unknown captain type: {captain_type}")
