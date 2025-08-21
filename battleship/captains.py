from random import random
from typing import Dict
from typing import List

import numpy as np

from battleship.agents import ActionData
from battleship.agents import Agent
from battleship.agents import ANSWER_MATCH_PATTERN
from battleship.agents import DECISION_PATTERN
from battleship.agents import EIGCalculator
from battleship.agents import get_openai_client
from battleship.agents import MOVE_PATTERN
from battleship.agents import Question
from battleship.board import Board
from battleship.board import tile_to_coords
from battleship.fast_sampler import FastSampler
from battleship.game import Decision
from battleship.prompting import DecisionPrompt
from battleship.prompting import MovePrompt
from battleship.prompting import QuestionPrompt
from battleship.strategies import DecisionStrategy
from battleship.strategies import MoveStrategy
from battleship.strategies import QuestionStrategy


class Captain(Agent):
    def __init__(
        self,
        decision_strategy=None,
        move_strategy=None,
        question_strategy=None,
        seed: int = None,
        llm=None,
        temperature=None,
        json_path=None,
    ):
        super().__init__(
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
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
        decision, action_data = self.decision_strategy(
            state,
            history,
            questions_remaining,
            moves_remaining,
            sunk,
        )

        # Save the action data
        self.save_action_data(action_data)

        return decision

    def move(
        self,
        state: Board,
        history: List[Dict],
        sunk: str,
        questions_remaining: int,
        moves_remaining: int,
        constraints: List,
    ):
        move, action_data = self.move_strategy(
            state,
            history,
            sunk,
            questions_remaining,
            moves_remaining,
            constraints,
        )

        # Save the action data
        self.save_action_data(action_data)

        return move

    def question(
        self,
        state: Board,
        history: List[Dict],
        sunk: str,
        questions_remaining: int,
        moves_remaining: int,
        constraints: List,
        true_board: Board = None,
    ):
        question, action_data = self.question_strategy(
            state, history, sunk, questions_remaining, moves_remaining, constraints, true_board
        )

        # Save the action data
        self.save_action_data(action_data)

        return question


# Example decision strategies
class AlwaysMoveDecisionStrategy(DecisionStrategy):
    def __call__(self, state, history, questions_remaining, moves_remaining, sunk):
        action_data = ActionData(
            action="decision",
            decision=Decision.MOVE,
            board_state=state.to_numpy(),
        )
        return Decision.MOVE, action_data


class ProbabilisticDecisionStrategy(DecisionStrategy):
    def __init__(self, q_prob=0.5):
        super().__init__()
        self.q_prob = q_prob

    def __call__(self, state, history, questions_remaining, moves_remaining, sunk):
        if random() < self.q_prob and questions_remaining > 0:
            decision = Decision.QUESTION
        else:
            decision = Decision.MOVE

        action_data = ActionData(
            action="decision",
            decision=decision,
            board_state=state.to_numpy(),
        )
        return decision, action_data


class LLMDecisionStrategy(DecisionStrategy):
    def __init__(
        self,
        llm,
        temperature=None,
        use_cot=False,
    ):
        super().__init__()
        self.llm = llm
        self.temperature = temperature
        self.use_cot = use_cot
        self.client = get_openai_client()

    def __call__(
        self, state, history, questions_remaining, moves_remaining, sunk, n_attempts=3
    ):
        if questions_remaining > 0:
            decision_prompt = DecisionPrompt(
                board=state,
                board_format="grid",
                history=history,
                use_cot=self.use_cot,
                questions_remaining=questions_remaining,
                moves_remaining=moves_remaining,
                sunk=sunk,
            )

            candidate_decision = None
            completion = None
            for _ in range(n_attempts):
                completion = self.client.chat.completions.create(
                    model=self.llm,
                    messages=decision_prompt.to_chat_format(),
                    temperature=self.temperature,
                )
                match = DECISION_PATTERN.search(completion.choices[0].message.content)

                if match is not None:
                    candidate_decision = match.group(1)
                    break

            decision = (
                Decision.MOVE if candidate_decision == "Move" else Decision.QUESTION
            )

        else:
            completion = None
            decision_prompt = None
            decision = Decision.MOVE

        # Create an ActionData object to store the interaction
        action_data = ActionData(
            action="decision",
            prompt=str(decision_prompt) if decision_prompt else None,
            completion=completion.model_dump() if completion else None,
            decision=decision,
            board_state=state.to_numpy(),
        )
        return decision, action_data


# Example move strategies
class RandomMoveStrategy(MoveStrategy):
    def __init__(self, rng):
        super().__init__()
        self.rng = rng

    def __call__(
        self, state, history, sunk, questions_remaining, moves_remaining, constraints
    ):
        hidden_tiles = np.argwhere(state.board == Board.hidden)
        if len(hidden_tiles) == 0:
            raise ValueError("No hidden tiles left.")
        coords = tuple(self.rng.choice(hidden_tiles))

        action_data = ActionData(
            action="move",
            move=coords,
            board_state=state.to_numpy(),
        )
        return coords, action_data


class MAPMoveStrategy(MoveStrategy):
    def __init__(self, rng, board_id, n_samples=1000):
        super().__init__()
        self.rng = rng
        self.board_id = board_id
        self.n_samples = n_samples

    def __call__(
        self, state, history, sunk, questions_remaining, moves_remaining, constraints
    ):
        sampler = FastSampler(
            board=state,
            ship_lengths=Board.SHIP_LENGTHS,
            ship_labels=Board.SHIP_LABELS,
            seed=self.rng,
        )

        # Compute the posterior counts over board positions (handles constraints internally)
        posterior = sampler.compute_posterior(
            n_samples=self.n_samples,
            normalize=False,
            constraints=constraints,
        )

        # For tiles that have already been revealed, force their probability to -infinity
        posterior = posterior.astype(float)
        posterior[state.board != Board.hidden] = -np.inf

        # Select the tile with the maximum posterior probability (MAP estimate)
        flat_idx = int(np.argmax(posterior))

        # Map the flat index back to 2D coordinates
        move_coords = np.unravel_index(flat_idx, state.board.shape)
        move = tuple(move_coords)

        action_data = ActionData(
            action="move",
            move=move,
            map_prob=float(posterior.max()),
            board_state=state.to_numpy(),
        )
        return move, action_data


class LLMMoveStrategy(MoveStrategy):
    def __init__(
        self,
        llm,
        temperature=None,
        use_cot=False,
        n_attempts=3,
        rng=np.random.default_rng()
    ):
        super().__init__()
        self.llm = llm
        self.temperature = temperature
        self.use_cot = use_cot
        self.n_attempts = n_attempts
        self.client = get_openai_client()
        self.rng = rng

    def __call__(
        self, state, history, sunk, questions_remaining, moves_remaining, constraints
    ):
        visible_tiles = list(zip(*np.where(state.board != Board.hidden)))

        move_prompt = MovePrompt(
            board=state,
            board_format="grid",
            history=history,
            use_cot=self.use_cot,
            questions_remaining=questions_remaining,
            moves_remaining=moves_remaining,
            sunk=sunk,
        )

        sampler = FastSampler(
            board=state,
            ship_lengths=Board.SHIP_LENGTHS,
            ship_labels=Board.SHIP_LABELS,
            seed=self.rng,
        )

        posterior = sampler.compute_posterior(
                n_samples=1000,
                normalize=False,
            )

        completion = None
        for _ in range(self.n_attempts):
            completion = self.client.chat.completions.create(
                model=self.llm,
                messages=move_prompt.to_chat_format(),
                temperature=self.temperature,
            )

            candidate_move = MOVE_PATTERN(state.size).search(
                completion.choices[0].message.content
            )
            if candidate_move is not None:
                candidate_move = tile_to_coords(candidate_move.group(1))
                if candidate_move not in visible_tiles:
                    # Create an ActionData object to store the interaction
                    action_data = ActionData(
                        action="move",
                        prompt=str(move_prompt),
                        completion=completion.model_dump(),
                        move=candidate_move,
                        map_prob=float(posterior[candidate_move]),
                        board_state=state.to_numpy(),
                    )

                    return candidate_move, action_data

        # If no valid move found, return None with ActionData
        action_data = ActionData(
            action="move",
            prompt=str(move_prompt),
            completion=completion.model_dump() if completion else None,
            move=None,
            board_state=state.to_numpy(),
        )
        return None, action_data


# Example question strategies
class EIGQuestionStrategy(QuestionStrategy):
    def __init__(
        self,
        llm,
        spotter,
        rng,
        samples=100,
        k=3,
        use_cot=False,
        n_attempts=3,
    ):
        super().__init__()
        self.llm = llm
        self.spotter = spotter
        self.rng = rng
        self.samples = samples
        self.k = k
        self.use_cot = use_cot
        self.eig_calculator = EIGCalculator(seed=self.rng, samples=self.samples)
        self.n_attempts = n_attempts
        self.client = get_openai_client()

    def __call__(self, state, history, sunk, questions_remaining, moves_remaining, constraints, true_board=None):
        best_question = None
        best_eig = -1
        best_action_data = None

        # Generate shared weighted boards once for all candidate questions
        from battleship.fast_sampler import FastSampler
        sampler = FastSampler(
            board=state,
            ship_lengths=Board.SHIP_LENGTHS,
            ship_labels=Board.SHIP_LABELS,
            seed=self.rng,
        )
        shared_weighted_boards = sampler.get_weighted_samples(
            n_boards=self.samples,
            constraints=[],  # No constraints for basic EIG
            epsilon=self.eig_calculator.epsilon
        )

        candidate_question_list = []
        for _ in range(self.k):
            question_prompt = QuestionPrompt(
                board=state,
                board_format="grid",
                history=history,
                use_cot=self.use_cot,
                questions_remaining=questions_remaining,
                moves_remaining=moves_remaining,
                sunk=sunk,
            )

            candidate_question_text = None
            completion = None
            for _ in range(self.n_attempts):
                completion = self.client.chat.completions.create(
                    model=self.llm,
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

            # First translate the question
            code_question = self.spotter.translate(
                question=candidate_question,
                board=state,
                history=history,
            )

            # Then calculate EIG using shared weighted boards
            eig = self.eig_calculator(code_question, state, [], shared_weighted_boards)

            # Create an ActionData object to store the interaction
            action_data = ActionData(
                action="question",
                prompt=str(question_prompt),
                completion=completion.model_dump(),
                question=code_question,
                eig=eig,
                board_state=state.to_numpy(),
                eig_questions=None
            )

            candidate_question_list.append(action_data.to_dict())

            # Update best question if this one has higher EIG
            if eig > best_eig:
                best_eig = eig
                best_question = candidate_question
                best_action_data = action_data

        best_action_data.eig_questions = candidate_question_list
        return best_question, best_action_data


class ConditionalEIGQuestionStrategy(QuestionStrategy):
    def __init__(
        self,
        llm,
        spotter,
        rng,
        samples=100,
        k=3,
        use_cot=False,
        n_attempts=3,
        epsilon=0.1,
    ):
        super().__init__()
        self.llm = llm
        self.spotter = spotter
        self.rng = rng
        self.samples = samples
        self.k = k
        self.use_cot = use_cot
        self.eig_calculator = EIGCalculator(seed=self.rng, samples=self.samples, epsilon=epsilon)
        self.n_attempts = n_attempts
        self.client = get_openai_client()

    def __call__(self, state, history, sunk, questions_remaining, moves_remaining, constraints, true_board=None):
        best_question = None
        best_eig = -1
        best_action_data = None

        # Generate shared weighted boards once for all candidate questions
        from battleship.fast_sampler import FastSampler
        sampler = FastSampler(
            board=state,
            ship_lengths=Board.SHIP_LENGTHS,
            ship_labels=Board.SHIP_LABELS,
            seed=self.rng,
        )
        shared_weighted_boards = sampler.get_weighted_samples(
            n_boards=self.samples,
            constraints=constraints,  # Use passed constraints for conditional EIG
            epsilon=self.eig_calculator.epsilon
        )

        candidate_question_list = []
        for _ in range(self.k):
            question_prompt = QuestionPrompt(
                board=state,
                board_format="grid",
                history=history,
                use_cot=self.use_cot,
                questions_remaining=questions_remaining,
                moves_remaining=moves_remaining,
                sunk=sunk,
            )

            candidate_question_text = None
            completion = None
            for _ in range(self.n_attempts):
                completion = self.client.chat.completions.create(
                    model=self.llm,
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

            # First translate the question
            code_question = self.spotter.translate(
                question=candidate_question,
                board=state,
                history=history,
            )

            # Then calculate conditional EIG using shared weighted boards
            eig = self.eig_calculator(code_question, state, constraints, shared_weighted_boards)

            # Create an ActionData object to store the interaction
            action_data = ActionData(
                action="question",
                prompt=str(question_prompt),
                completion=completion.model_dump(),
                question=code_question,
                eig=eig,
                board_state=state.to_numpy(),
                eig_questions=None
            )

            candidate_question_list.append(action_data.to_dict())

            # Update best question if this one has higher EIG
            if eig > best_eig:
                best_eig = eig
                best_question = candidate_question
                best_action_data = action_data

        best_action_data.eig_questions = candidate_question_list
        return best_question, best_action_data


class LLMQuestionStrategy(QuestionStrategy):
    def __init__(
        self,
        llm,
        temperature=None,
        use_cot=False,
        spotter=None,
        rng=None,
        n_attempts=3,
    ):
        super().__init__()
        self.llm = llm
        self.temperature = temperature
        self.use_cot = use_cot
        self.n_attempts = n_attempts
        self.spotter = spotter
        self.rng = rng
        self.eig_calculator = EIGCalculator(seed=self.rng)
        self.client = get_openai_client()

    def __call__(self, state, history, sunk, questions_remaining, moves_remaining, constraints, true_board=None):
        question_prompt = QuestionPrompt(
            board=state,
            board_format="grid",
            history=history,
            use_cot=self.use_cot,
            questions_remaining=questions_remaining,
            moves_remaining=moves_remaining,
            sunk=sunk,
        )

        completion = None
        for _ in range(self.n_attempts):
            completion = self.client.chat.completions.create(
                model=self.llm,
                messages=question_prompt.to_chat_format(),
                temperature=self.temperature,
            )

            candidate_question = ANSWER_MATCH_PATTERN.search(
                completion.choices[0].message.content
            )

            if candidate_question:
                candidate_question = candidate_question.group(1)
                question = Question(text=candidate_question)

                code_question = self.spotter.translate(
                    question=question,
                    board=state,
                    history=history,
                )

                eig = self.eig_calculator(code_question, state)

                action_data = ActionData(
                action="question",
                prompt=str(question_prompt),
                completion=completion.model_dump(),
                question=code_question,
                eig=eig,
                board_state=state.to_numpy(),
                )

                return question, action_data

        # If no valid question found, return None with ActionData
        action_data = ActionData(
            action="question",
            prompt=str(question_prompt),
            completion=completion.model_dump() if completion else None,
            question=None,
            board_state=state.to_numpy(),
        )
        return None, action_data


def create_captain(
    captain_type,
    seed,
    llm,
    board_id,
    map_samples=None,
    prob_q_prob=None,
    eig_samples=None,
    eig_k=None,
    json_path=None,
    completions_dir=None,
):
    """
    Factory function to create Captain instances with properly configured strategies.
    """
    from battleship.spotters import create_spotter

    # Initialize spotter for EIG captains
    def _get_spotter():
        return create_spotter(
            spotter_type="CodeSpotterModel",
            board_id=board_id,
            board_experiment="collaborative",
            llm=llm,
            temperature=None,
            use_cot=True,
        )

    if captain_type == "RandomCaptain":
        return Captain(
            decision_strategy=AlwaysMoveDecisionStrategy(),
            move_strategy=RandomMoveStrategy(rng=np.random.default_rng(seed)),
            question_strategy=None,
            seed=seed,
            json_path=json_path,
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
            json_path=json_path,
        )

    elif captain_type == "ProbabilisticCaptain":
        captain = Captain(
            decision_strategy=ProbabilisticDecisionStrategy(q_prob=prob_q_prob),
            move_strategy=LLMMoveStrategy(
                llm=llm,
                use_cot=False,
                rng=np.random.default_rng(seed),
            ),
            question_strategy=LLMQuestionStrategy(
                llm=llm,
                use_cot=False,
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "ProbabilisticCaptain_cot":
        captain = Captain(
            decision_strategy=ProbabilisticDecisionStrategy(q_prob=prob_q_prob),
            move_strategy=LLMMoveStrategy(
                llm=llm,
                use_cot=True,
                rng=np.random.default_rng(seed),
            ),
            question_strategy=LLMQuestionStrategy(
                llm=llm,
                use_cot=True,
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "LLMDecisionCaptain":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                llm=llm,
                use_cot=False,
            ),
            move_strategy=LLMMoveStrategy(
                llm=llm,
                use_cot=False,
                rng=np.random.default_rng(seed),
            ),
            question_strategy=LLMQuestionStrategy(
                llm=llm,
                use_cot=False,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "LLMDecisionCaptain_cot":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                llm=llm,
                use_cot=True,
            ),
            move_strategy=LLMMoveStrategy(
                llm=llm,
                use_cot=True,
                rng=np.random.default_rng(seed),
            ),
            question_strategy=LLMQuestionStrategy(
                llm=llm,
                use_cot=True,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "EIGCaptain":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                llm=llm,
                use_cot=False,
            ),
            move_strategy=LLMMoveStrategy(
                llm=llm,
                use_cot=False,
                rng=np.random.default_rng(seed),
            ),
            question_strategy=EIGQuestionStrategy(
                llm=llm,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=False,
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "EIGCaptain_cot":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                llm=llm,
                use_cot=True,
            ),
            move_strategy=LLMMoveStrategy(
                llm=llm,
                use_cot=True,
                rng=np.random.default_rng(seed),
            ),
            question_strategy=EIGQuestionStrategy(
                llm=llm,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=True,
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "MAPEIGCaptain":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                llm=llm,
                use_cot=False,
            ),
            move_strategy=MAPMoveStrategy(
                rng=np.random.default_rng(seed),
                board_id=board_id,
                n_samples=eig_samples,
            ),
            question_strategy=EIGQuestionStrategy(
                llm=llm,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=False,
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "MAPEIGCaptain_cot":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                llm=llm,
                use_cot=True,
            ),
            move_strategy=MAPMoveStrategy(
                rng=np.random.default_rng(seed),
                board_id=board_id,
                n_samples=eig_samples,
            ),
            question_strategy=EIGQuestionStrategy(
                llm=llm,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=True,
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "ConditionalEIGCaptain":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                llm=llm,
                use_cot=False,
            ),
            move_strategy=LLMMoveStrategy(
                llm=llm,
                use_cot=False,
                rng=np.random.default_rng(seed),
            ),
            question_strategy=ConditionalEIGQuestionStrategy(
                llm=llm,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=False,
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "ConditionalEIGCaptain_cot":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                llm=llm,
                use_cot=True,
            ),
            move_strategy=LLMMoveStrategy(
                llm=llm,
                use_cot=True,
                rng=np.random.default_rng(seed),
            ),
            question_strategy=ConditionalEIGQuestionStrategy(
                llm=llm,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=True,
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "MAPConditionalEIGCaptain":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                llm=llm,
                use_cot=False,
            ),
            move_strategy=MAPMoveStrategy(
                rng=np.random.default_rng(seed),
                board_id=board_id,
                n_samples=eig_samples,
            ),
            question_strategy=ConditionalEIGQuestionStrategy(
                llm=llm,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=False,
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    elif captain_type == "MAPConditionalEIGCaptain_cot":
        captain = Captain(
            decision_strategy=LLMDecisionStrategy(
                llm=llm,
                use_cot=True,
            ),
            move_strategy=MAPMoveStrategy(
                rng=np.random.default_rng(seed),
                board_id=board_id,
                n_samples=eig_samples,
            ),
            question_strategy=ConditionalEIGQuestionStrategy(
                llm=llm,
                spotter=_get_spotter(),
                rng=np.random.default_rng(seed),
                samples=eig_samples,
                k=eig_k,
                use_cot=True,
            ),
            seed=seed,
            llm=llm,
            json_path=json_path,
        )
        return captain

    else:
        raise ValueError(f"Unknown captain type: {captain_type}")
