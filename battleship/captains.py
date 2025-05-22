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
from battleship.agents import CacheData
from battleship.agents import DECISION_PATTERN
from battleship.agents import EIGCalculator
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
    ) -> Tuple[Decision, Dict]:
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
        use_cache: bool = True,
        model_string=None,
        temperature=None,
        round_id=None,
    ):
        super().__init__(
            seed=seed, model_string=model_string, use_cache=use_cache, round_id=round_id
        )
        self.temperature = temperature
        self.sampling_constraints = []

        # Optional strategies for modular approach
        self.round_id = round_id
        self.decision_strategy = decision_strategy
        self.move_strategy = move_strategy
        self.question_strategy = question_strategy

        # Set client for strategies that need it
        self._set_client_for_strategies()

    def _set_client_for_strategies(self):
        """Set the OpenAI client for strategies that need it."""
        if (
            hasattr(self.decision_strategy, "client")
            and self.decision_strategy.client is None
        ):
            self.decision_strategy.client = self.client
        if hasattr(self.move_strategy, "client") and self.move_strategy.client is None:
            self.move_strategy.client = self.client
        if (
            hasattr(self.question_strategy, "client")
            and self.question_strategy.client is None
        ):
            self.question_strategy.client = self.client

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
        result, decision_cache = self.decision_strategy.make_decision(
            state,
            history,
            questions_remaining,
            moves_remaining,
            sunk,
        )

        if self.use_cache:
            self.write_cache(
                message_type="DECISION",
                cache_data=decision_cache,
            )

        return result

    def move(self, state: Board, history: List[Dict], sunk: str, constraints: List):
        result, move_cache = self.move_strategy.make_move(
            state,
            history,
            sunk,
            self.questions_remaining,
            self.moves_remaining,
            constraints,
        )

        if self.use_cache:
            self.write_cache(
                message_type="MOVE",
                cache_data=move_cache,
            )

        return result

    def question(self, state: Board, history: List[Dict], sunk: str):
        result, question_cache = self.question_strategy.ask_question(
            state, history, sunk, self.questions_remaining, self.moves_remaining
        )

        if self.use_cache:
            self.write_cache(
                message_type="QUESTION",
                cache_data=question_cache,
            )

        return result


# Example decision strategies
class AlwaysMoveDecisionStrategy(DecisionStrategy):
    def make_decision(self, state, history, questions_remaining, moves_remaining, sunk):
        return Decision.MOVE, CacheData(
            message_text=Decision.MOVE, occ_tiles=state.board
        )


class ProbabilisticDecisionStrategy(DecisionStrategy):
    def __init__(self, q_prob=0.5):
        self.q_prob = q_prob

    def make_decision(self, state, history, questions_remaining, moves_remaining, sunk):
        decision = None
        if random() < self.q_prob and questions_remaining > 0:
            decision = Decision.QUESTION
        else:
            decision = Decision.MOVE
        return decision, CacheData(message_text=decision, occ_tiles=state.board)


class LLMDecisionStrategy(DecisionStrategy):
    def __init__(self, model_string, temperature=None, use_cot=False, client=None):
        self.model_string = model_string
        self.temperature = temperature
        self.use_cot = use_cot
        self.client = client

    def make_decision(
        self, state, history, questions_remaining, moves_remaining, sunk, n_attempts=3
    ):
        decision = None
        prompts = []

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

                prompts.append(
                    Prompt(
                        prompt=decision_prompt.to_chat_format(),
                        full_completion=completion.choices[0].message.content,
                        extracted_completion=candidate_decision,
                        occ_tiles=state.board,
                    )
                )

                if candidate_decision:
                    break

            decision = (
                Decision.MOVE if candidate_decision == "Move" else Decision.QUESTION
            )
        else:
            decision = Decision.MOVE

        return decision, CacheData(
            message_text=decision, occ_tiles=state.board, prompts=prompts
        )


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
        return tuple(coords), CacheData(
            message_text=coords_to_tile(coords), occ_tiles=state.board
        )


class MAPMoveStrategy(MoveStrategy):
    def __init__(self, rng, n_samples=10000):
        self.rng = rng
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
            # Compute the raw posterior counts over board positions
            posterior = sampler.constrained_posterior(
                n_samples=self.n_samples, normalize=False, constraints=constraints
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
        return tuple(move_coords), CacheData(
            message_text=coords_to_tile(move_coords),
            map_prob=posterior[move_coords[0]][move_coords[1]],
            occ_tiles=state.board,
        )


class LLMMoveStrategy(MoveStrategy):
    def __init__(
        self,
        model_string,
        temperature=None,
        use_cot=False,
        moves_remaining=None,
        n_attempts=3,
        client=None,
    ):
        self.model_string = model_string
        self.temperature = temperature
        self.use_cot = use_cot
        self.moves_remaining = moves_remaining
        self.n_attempts = n_attempts
        self.client = client

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

        candidate_move = None
        prompts = []
        for _ in range(self.n_attempts):
            # while candidate_move is None or candidate_move in visible_tiles:
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

            prompts.append(
                Prompt(
                    prompt=move_prompt.to_chat_format(),
                    full_completion=completion.choices[0].message.content,
                    extracted_completion=coords_to_tile(candidate_move)
                    if candidate_move
                    else None,
                    occ_tiles=state.board,
                )
            )

            if candidate_move is not None and candidate_move not in visible_tiles:
                return candidate_move, CacheData(
                    message_text=coords_to_tile(candidate_move),
                    occ_tiles=state.board,
                    prompts=prompts,
                )

        return None, CacheData(
            message_text=None,
            occ_tiles=state.board,
            prompts=prompts,
        )


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
        client=None,
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
        self.client = client

    def ask_question(self, state, history, sunk, questions_remaining, moves_remaining):
        best_question = None
        best_eig = -1

        prompts = []
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

                if candidate_question_text is not None:
                    break
            candidate_question = Question(text=candidate_question_text)

            # Calculate EIG for this question
            eig = self.eig_calculator.calculate_eig(
                candidate_question, state, samples=self.samples
            )

            # Update best question if this one has higher EIG
            if eig > best_eig:
                best_eig = eig
                best_question = candidate_question

            prompts.append(
                Prompt(
                    prompt=question_prompt.to_chat_format(),
                    full_completion=completion.choices[0].message.content,
                    extracted_completion=candidate_question,
                    occ_tiles=state.board,
                    eig=eig,
                )
            )

        return best_question, CacheData(
            message_text=best_question.text,
            eig=best_eig,
            occ_tiles=state.board,
            prompts=prompts,
        )


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
        client=None,
    ):
        self.model_string = model_string
        self.temperature = temperature
        self.use_cot = use_cot
        self.questions_remaining = questions_remaining
        self.n_attempts = n_attempts
        self.spotter = spotter
        self.rng = rng
        self.eig_calculator = EIGCalculator(seed=self.rng, spotter=self.spotter)
        self.client = client

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

        prompts = []
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
            else:
                candidate_question = None

            prompts.append(
                Prompt(
                    prompt=question_prompt.to_chat_format(),
                    full_completion=completion.choices[0].message.content,
                    extracted_completion=candidate_question,
                    occ_tiles=state.board,
                )
            )

            if candidate_question is not None:
                break

        eig = self.eig_calculator.calculate_eig(
            Question(text=candidate_question), state, samples=1000
        )

        return Question(text=candidate_question), CacheData(
            message_text=candidate_question,
            occ_tiles=state.board,
            prompts=prompts,
            eig=eig,
        )
