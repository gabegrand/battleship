from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from battleship.agents import ActionData
from battleship.agents import ANSWER_MATCH_PATTERN
from battleship.agents import EIGCalculator
from battleship.agents import get_openai_client
from battleship.agents import Question
from battleship.board import Board
from battleship.fast_sampler import FastSampler
from battleship.game import Decision
from battleship.prompting import QuestionPrompt
from battleship.strategies import DecisionStrategy
from battleship.strategies import MoveStrategy
from battleship.strategies import QuestionStrategy


@dataclass
class Plan:
    """Per-stage plan computed at decision time and consumed by question/move."""

    best_move: Optional[Tuple[int, int]] = None
    p_hit_before: float = 0.0

    best_question: Optional[Question] = None
    best_code_question: Any = None
    best_eig: Optional[float] = None

    eig_candidates: List[Dict[str, Any]] = field(default_factory=list)
    p_hit_after: Optional[float] = None

    true_partition: Dict[str, Any] = field(default_factory=dict)
    false_partition: Dict[str, Any] = field(default_factory=dict)


class StrategyPlanner:
    """Computes a per-stage plan used by Decision/Question/Move adapters."""

    def __init__(
        self,
        *,
        llm,
        spotter,
        rng,
        samples: int = 1000,
        k: int = 10,
        use_cot: bool = False,
        temperature: Optional[float] = None,
        n_attempts: int = 3,
    ) -> None:
        self.llm = llm
        self.spotter = spotter
        self.rng = rng
        self.samples = samples
        self.k = k
        self.use_cot = use_cot
        self.temperature = temperature
        self.n_attempts = n_attempts

        self.client = get_openai_client()
        self.eig_calculator = EIGCalculator(seed=self.rng, samples=self.samples)

        self._plan: Optional[Plan] = None
        self._constraints_ref: Optional[List] = None

    def set_constraints_ref(self, ref: List) -> None:
        self._constraints_ref = ref

    @staticmethod
    def _map_from_weighted_boards(
        weighted_boards: List[Tuple[Board, float]], state: Board
    ):
        if not weighted_boards:
            return None, 0.0, 0.0

        boards, weights = zip(*weighted_boards)
        total_weight = float(sum(weights))
        posterior = sum(
            w * (b.board > 0).astype(float) for b, w in zip(boards, weights)
        )
        posterior = posterior.astype(float)
        posterior[state.board != Board.hidden] = 0.0

        if total_weight > 0:
            posterior /= total_weight

        flat_idx = int(np.argmax(posterior))
        move_coords = np.unravel_index(flat_idx, posterior.shape)
        move = tuple(move_coords)
        map_prob = float(posterior[move])
        return move, map_prob, total_weight

    def _generate_candidate_questions(
        self,
        *,
        state: Board,
        history: List[Dict],
        questions_remaining: int,
        moves_remaining: int,
        ship_tracker: List[Tuple[int, Optional[str]]],
        shared_weighted_boards: List[Tuple[Board, float]],
        constraints: List,
    ) -> Tuple[
        Optional[Question], Optional[Any], Optional[float], List[Dict[str, Any]]
    ]:
        best_q = None
        best_code_q = None
        best_eig = -np.inf
        candidates_log: List[Dict[str, Any]] = []

        for _ in range(self.k):
            question_prompt = QuestionPrompt(
                board=state,
                board_format="grid",
                history=history,
                use_cot=self.use_cot,
                questions_remaining=questions_remaining,
                moves_remaining=moves_remaining,
                ship_tracker=ship_tracker,
            )

            candidate_text = None
            completion = None
            for _ in range(self.n_attempts):
                completion = self.client.chat.completions.create(
                    model=self.llm,
                    messages=question_prompt.to_chat_format(),
                    temperature=self.temperature,
                )
                match = ANSWER_MATCH_PATTERN.search(
                    completion.choices[0].message.content
                )
                if match:
                    candidate_text = match.group(1)
                    break

            if not candidate_text:
                continue

            q = Question(text=candidate_text)
            code_q = self.spotter.translate(
                question=q,
                board=state,
                history=history,
            )

            eig = self.eig_calculator(
                code_question=code_q,
                state=state,
                ship_tracker=ship_tracker,
                constraints=constraints,
                weighted_boards=shared_weighted_boards,
            )

            action_data = ActionData(
                action="question",
                prompt=str(question_prompt),
                completion=completion.model_dump() if completion else None,
                question=code_q,
                eig=float(eig) if eig is not None else None,
                board_state=state.to_numpy(),
                eig_questions=None,
            )
            candidates_log.append(action_data.to_dict())

            if eig is not None and eig > best_eig:
                best_eig = float(eig)
                best_q = q
                best_code_q = code_q

        if best_eig == -np.inf:
            best_eig = None

        return best_q, best_code_q, best_eig, candidates_log

    def _expected_post_question_hit_prob(
        self,
        *,
        state: Board,
        code_question: Any,
        shared_weighted_boards: List[Tuple[Board, float]],
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        true_wb: List[Tuple[Board, float]] = []
        false_wb: List[Tuple[Board, float]] = []

        for b, w in shared_weighted_boards:
            ans = code_question(true_board=b.board, partial_board=state.board)
            val = getattr(ans, "value", None)
            if val is True:
                true_wb.append((b, w))
            elif val is False:
                false_wb.append((b, w))

        t_move, t_prob, t_weight = self._map_from_weighted_boards(true_wb, state)
        f_move, f_prob, f_weight = self._map_from_weighted_boards(false_wb, state)

        total = t_weight + f_weight
        if total <= 0:
            return 0.0, {}, {}

        p_after = 0.0
        if t_weight > 0:
            p_after += (t_weight / total) * (t_prob or 0.0)
        if f_weight > 0:
            p_after += (f_weight / total) * (f_prob or 0.0)

        true_info = {"move": t_move, "prob": t_prob, "weight": t_weight}
        false_info = {"move": f_move, "prob": f_prob, "weight": f_weight}

        return float(p_after), true_info, false_info

    def plan_turn(
        self,
        *,
        state: Board,
        history: List[Dict],
        questions_remaining: int,
        moves_remaining: int,
        ship_tracker: List[Tuple[int, Optional[str]]],
        constraints: Optional[List] = None,
    ) -> Plan:
        if constraints is None:
            constraints = self._constraints_ref or []

        sampler = FastSampler(board=state, ship_tracker=ship_tracker, seed=self.rng)
        shared_weighted_boards = sampler.get_weighted_samples(
            n_samples=self.samples,
            constraints=constraints,
            epsilon=self.eig_calculator.epsilon,
        )

        weighted_boards_simple = [(b, w) for (b, w) in shared_weighted_boards]
        best_move, p_before, _ = self._map_from_weighted_boards(
            weighted_boards_simple, state
        )

        plan = Plan(best_move=best_move, p_hit_before=float(p_before))

        if questions_remaining <= 0:
            self._plan = plan
            return plan

        (
            best_q,
            best_code_q,
            best_eig,
            candidates_log,
        ) = self._generate_candidate_questions(
            state=state,
            history=history,
            questions_remaining=questions_remaining,
            moves_remaining=moves_remaining,
            ship_tracker=ship_tracker,
            shared_weighted_boards=weighted_boards_simple,
            constraints=constraints,
        )

        plan.best_question = best_q
        plan.best_code_question = best_code_q
        plan.best_eig = best_eig
        plan.eig_candidates = candidates_log

        if best_code_q is not None:
            p_after, true_info, false_info = self._expected_post_question_hit_prob(
                state=state,
                code_question=best_code_q,
                shared_weighted_boards=weighted_boards_simple,
            )
            plan.p_hit_after = float(p_after)
            plan.true_partition = true_info
            plan.false_partition = false_info

        self._plan = plan
        return plan

    def get_plan(self) -> Optional[Plan]:
        return self._plan

    def invalidate(self) -> None:
        self._plan = None

    def plan_summary(self) -> Optional[Dict[str, Any]]:
        """Return a compact, JSON-serializable snapshot of the current plan."""
        p = self._plan
        if p is None:
            return None
        return {
            "p_hit_before": float(p.p_hit_before)
            if p.p_hit_before is not None
            else None,
            "p_hit_after": float(p.p_hit_after) if p.p_hit_after is not None else None,
            "best_move": tuple(int(x) for x in p.best_move) if p.best_move else None,
            "best_eig": float(p.best_eig) if p.best_eig is not None else None,
            "best_question": p.best_question.to_dict() if p.best_question else None,
            "true_partition": {
                "move": p.true_partition.get("move"),
                "prob": p.true_partition.get("prob"),
                "weight": p.true_partition.get("weight"),
            }
            if p.true_partition
            else None,
            "false_partition": {
                "move": p.false_partition.get("move"),
                "prob": p.false_partition.get("prob"),
                "weight": p.false_partition.get("weight"),
            }
            if p.false_partition
            else None,
        }

    def expected_post_question_hit_prob(
        self,
        *,
        state: Board,
        code_question: Any,
        constraints: Optional[List] = None,
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        if constraints is None:
            constraints = self._constraints_ref or []

        sampler = FastSampler(board=state, ship_tracker=None, seed=self.rng)
        shared_weighted_boards = sampler.get_weighted_samples(
            n_samples=self.samples,
            constraints=constraints,
            epsilon=self.eig_calculator.epsilon,
        )

        return self._expected_post_question_hit_prob(
            state=state,
            code_question=code_question,
            shared_weighted_boards=shared_weighted_boards,
        )


class PlannedDecision(DecisionStrategy):
    def __init__(self, planner: StrategyPlanner, gamma: float = 0.95) -> None:
        """Decision adapter with a discount factor.

        gamma discounts the value of asking a question (future move benefit).
        Rule: ask iff gamma * p_hit_after > p_hit_before.
        """
        super().__init__()
        self.planner = planner
        self.gamma = gamma

    def __call__(
        self,
        state,
        history,
        questions_remaining,
        moves_remaining,
        ship_tracker,
    ):
        plan = self.planner.plan_turn(
            state=state,
            history=history,
            questions_remaining=questions_remaining,
            moves_remaining=moves_remaining,
            ship_tracker=ship_tracker,
            constraints=None,
        )

        if questions_remaining <= 0 or plan.p_hit_after is None:
            decision = Decision.MOVE
        else:
            decision = (
                Decision.QUESTION
                if (self.gamma * plan.p_hit_after > plan.p_hit_before)
                else Decision.MOVE
            )

        action_data = ActionData(
            action="decision",
            decision=decision,
            board_state=state.to_numpy(),
            plan=self.planner.plan_summary(),
        )
        return decision, action_data


class PlannedQuestion(QuestionStrategy):
    def __init__(self, planner: StrategyPlanner) -> None:
        super().__init__()
        self.planner = planner

    def __call__(
        self,
        state,
        history,
        ship_tracker,
        questions_remaining,
        moves_remaining,
        constraints,
    ):
        plan = self.planner.get_plan()
        if plan is None or plan.best_question is None:
            action_data = ActionData(
                action="question",
                question=None,
                board_state=state.to_numpy(),
            )
            return None, action_data

        action_data = ActionData(
            action="question",
            question=plan.best_code_question,
            eig=plan.best_eig,
            board_state=state.to_numpy(),
            eig_questions=plan.eig_candidates,
            plan=self.planner.plan_summary(),
        )

        self.planner.invalidate()
        return plan.best_question, action_data


class PlannedMove(MoveStrategy):
    def __init__(self, planner: StrategyPlanner):
        super().__init__()
        self.planner = planner

    def __call__(
        self,
        state,
        history,
        ship_tracker,
        questions_remaining,
        moves_remaining,
        constraints,
    ):
        plan = self.planner.get_plan()
        if plan is None or plan.best_move is None:
            action_data = ActionData(
                action="move",
                move=None,
                board_state=state.to_numpy(),
            )
            return None, action_data

        action_data = ActionData(
            action="move",
            move=plan.best_move,
            map_prob=plan.p_hit_before,
            board_state=state.to_numpy(),
            plan=self.planner.plan_summary(),
        )
        self.planner.invalidate()
        return plan.best_move, action_data
