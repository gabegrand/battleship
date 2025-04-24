from abc import ABC
from abc import abstractmethod
from random import random
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from battleship.battleship.agents import Agent
from battleship.battleship.agents import ANSWER_MATCH_PATTERN
from battleship.battleship.agents import CacheMode
from battleship.battleship.agents import client
from battleship.battleship.agents import DECISION_PATTERN
from battleship.battleship.agents import EIGCalculator
from battleship.battleship.agents import MOVE_COT_PATTERN
from battleship.battleship.agents import MOVE_PATTERN
from battleship.battleship.agents import Question
from battleship.board import Board
from battleship.board import tile_to_coords
from battleship.fast_sampler import FastSampler
from battleship.game import Decision
from battleship.prompting import DecisionPrompt
from battleship.prompting import MovePrompt
from battleship.prompting import QuestionPrompt


# Strategy interfaces
class DecisionStrategy(ABC):
    @abstractmethod
    def make_decision(self, state, history, questions_remaining, moves_remaining, sunk):
        pass


class MoveStrategy(ABC):
    @abstractmethod
    def make_move(self, state, history, sunk) -> Tuple[int, int]:
        pass


class QuestionStrategy(ABC):
    @abstractmethod
    def ask_question(self, state, history, sunk) -> Question:
        pass


class Captain(Agent):
    def __init__(
        self,
        decision_strategy=None,
        move_strategy=None,
        question_strategy=None,
        seed: int = None,
        cache_mode: CacheMode = CacheMode.WRITE_ONLY,
        model_string=None,
        temperature=None,
    ):
        super().__init__(seed=seed, model_string=model_string, cache_mode=cache_mode)
        self.temperature = temperature
        self.sampling_constraints = []

        # Optional strategies for modular approach
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
        cached_result = self.read_cache("DECISION")

        # If we have a cache hit in READ_WRITE mode
        if cached_result == "GENERATE_NEW":
            # Generate new response anyway
            result = self.decision_strategy.make_decision(
                state, history, questions_remaining, moves_remaining, sunk
            )

            # Increment counter after decision
            Agent.increment_counter(self)
            return result

        # Normal cache hit
        elif cached_result is not None:
            # Increment counter after decision
            Agent.increment_counter(self)
            return Decision(cached_result)

        # Cache miss
        result = self.decision_strategy.make_decision(
            state, history, questions_remaining, moves_remaining, sunk
        )

        # Increment counter after decision
        Agent.increment_counter(self)
        return result

    def move(self, state: Board, history: List[Dict], sunk: str):
        cached_result = self.read_cache("MOVE")

        # If we have a cache hit in READ_WRITE mode
        if cached_result == "GENERATE_NEW":
            # Generate new anyway
            result = self.move_strategy.make_move(state, history, sunk)

            # Increment counter after move
            Agent.increment_counter(self)
            return result

        # Normal cache hit
        elif cached_result is not None:
            # Convert cached string representation back to tuple
            coords = tuple(map(int, cached_result.strip("()").split(", ")))
            # Increment counter after move
            Agent.increment_counter(self)
            return coords

        # Cache miss
        result = self.move_strategy.make_move(state, history, sunk)

        # Increment counter after move
        Agent.increment_counter(self)
        return result

    def question(self, state: Board, history: List[Dict], sunk: str):
        cached_result = self.read_cache("QUESTION")

        # If we have a cache hit in READ_WRITE mode
        if cached_result == "GENERATE_NEW":
            # Generate new anyway
            result = self.question_strategy.ask_question(state, history, sunk)

            # Increment counter after question
            Agent.increment_counter(self)
            return result

        # Normal cache hit
        elif cached_result is not None:
            # Increment counter after question
            Agent.increment_counter(self)
            return Question(text=cached_result)

        # Cache miss
        result = self.question_strategy.ask_question(state, history, sunk)

        # Increment counter after question
        Agent.increment_counter(self)
        return result


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

    def make_decision(self, state, history, questions_remaining, moves_remaining, sunk):
        if questions_remaining > 0:
            decision_prompt = DecisionPrompt(
                target_occ_tiles=state,
                board_format="grid",
                history=history,
                use_cot=self.use_cot,
                q_remaining=questions_remaining,
                sunk=sunk,
            )

            candidate_decision = None
            while candidate_decision is None:
                completion = client.chat.completions.create(
                    model=self.model_string,
                    messages=decision_prompt.to_chat_format(),
                    temperature=self.temperature,
                )
                if self.use_cot:
                    match = ANSWER_MATCH_PATTERN.search(
                        completion.choices[0].message.content
                    )
                    if match is not None:
                        candidate_decision = match.group(1)
                else:
                    candidate_decision = DECISION_PATTERN.match(
                        completion.choices[0].message.content
                    ).group(0)

            return Decision.MOVE if candidate_decision == "Move" else Decision.QUESTION
        return Decision.MOVE


# Example move strategies
class RandomMoveStrategy(MoveStrategy):
    def __init__(self, rng):
        self.rng = rng

    def make_move(self, state, history, sunk):
        hidden_tiles = np.argwhere(state.board == Board.hidden)
        if len(hidden_tiles) == 0:
            raise ValueError("No hidden tiles left.")
        coords = self.rng.choice(hidden_tiles)
        return tuple(coords)


class MAPMoveStrategy(MoveStrategy):
    def __init__(self, rng, n_samples=10000, constraints=None):
        self.rng = rng
        self.n_samples = n_samples
        self.constraints = constraints or []

    def make_move(self, state, history, sunk):
        sampler = FastSampler(
            board=state,
            ship_lengths=Board.SHIP_LENGTHS,
            ship_labels=Board.SHIP_LABELS,
            seed=self.rng,
        )

        # Compute the raw posterior counts over board positions
        posterior = sampler.compute_posterior(
            n_samples=self.n_samples, normalize=False, constraints=self.constraints
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
        self, model_string, temperature=None, use_cot=False, moves_remaining=None
    ):
        self.model_string = model_string
        self.temperature = temperature
        self.use_cot = use_cot
        self.moves_remaining = moves_remaining

    def make_move(self, state, history, sunk):
        visible_tiles = list(zip(*np.where(state.board != Board.hidden)))

        move_prompt = MovePrompt(
            target_occ_tiles=state,
            board_format="grid",
            history=history,
            use_cot=self.use_cot,
            moves_remaining=self.moves_remaining,
            sunk=sunk,
        )

        candidate_move = None
        while candidate_move is None or candidate_move in visible_tiles:
            completion = client.chat.completions.create(
                model=self.model_string,
                messages=move_prompt.to_chat_format(),
                temperature=self.temperature,
            )
            if self.use_cot:
                match = MOVE_COT_PATTERN(state.size).search(
                    completion.choices[0].message.content
                )
                if match is not None:
                    candidate_move = tile_to_coords(match.group(1))
            else:
                candidate_move = MOVE_PATTERN(state.size).match(
                    completion.choices[0].message.content
                )
                if candidate_move is not None:
                    candidate_move = tile_to_coords(candidate_move.group())

        return candidate_move


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
    ):
        self.model_string = model_string
        self.spotter = spotter
        self.rng = rng
        self.samples = samples
        self.k = k
        self.use_cot = use_cot
        self.questions_remaining = questions_remaining
        self.eig_calculator = EIGCalculator(seed=self.rng, spotter=self.spotter)

    def ask_question(self, state, history, sunk):
        best_question = None
        best_eig = -1

        for _ in range(self.k):
            question_prompt = QuestionPrompt(
                target_occ_tiles=state,
                board_format="grid",
                history=history,
                use_cot=self.use_cot,
                q_remaining=self.questions_remaining,
                sunk=sunk,
            )

            # Generate a candidate question
            if not self.use_cot:
                completion = client.chat.completions.create(
                    model=self.model_string,
                    messages=question_prompt.to_chat_format(),
                    temperature=None,
                )
                candidate_question_text = completion.choices[0].message.content
            else:
                candidate_question_text = None
                while candidate_question_text is None:
                    completion = client.chat.completions.create(
                        model=self.model_string,
                        messages=question_prompt.to_chat_format(),
                        temperature=None,
                    )
                    match = ANSWER_MATCH_PATTERN.search(
                        completion.choices[0].message.content
                    )
                    if match:
                        candidate_question_text = match.group(1)

            # Create question object
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
        self, model_string, temperature=None, use_cot=False, questions_remaining=None
    ):
        self.model_string = model_string
        self.temperature = temperature
        self.use_cot = use_cot
        self.questions_remaining = questions_remaining

    def ask_question(self, state, history, sunk):
        question_prompt = QuestionPrompt(
            target_occ_tiles=state,
            board_format="grid",
            history=history,
            use_cot=self.use_cot,
            q_remaining=self.questions_remaining,
            sunk=sunk,
        )

        if not self.use_cot:
            completion = client.chat.completions.create(
                model=self.model_string,
                messages=question_prompt.to_chat_format(),
                temperature=self.temperature,
            )
            candidate_question = completion.choices[0].message.content
        else:
            candidate_question = None
            while candidate_question is None:
                completion = client.chat.completions.create(
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

        return Question(text=candidate_question)
