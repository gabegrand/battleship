import asyncio

import numpy as np

from battleship.board import Board
from battleship.prompting import TranslationPrompt
from battleship.scoring import compute_score
from hfppl.distributions import LMContext
from hfppl.llms import CachedCausalLM
from hfppl.modeling import Model


class QuestionGenerationModel(Model):
    """A model for generating questions about a battleship board via Monte Carlo
    Tree Search (MCTS).

    This model uses the LLM to generate questions. During generation, each
    partial question is completed with `n_rollouts` completions. Each completion
    is then translated to code and scored via EIG. The mean score of all
    completions is used as the score for the partial question.

    :param lm: The LLM to use for generation.
    :param board: The board to generate questions about.
    :param question_prompt: The base prompt to use for generating questions.
    :param translation_prompt: The base prompt to use for translating questions to code.
    :param n_rollouts: The number of rollouts to use for scoring.
    :param max_tokens: The maximum number of tokens to generate.
    """

    def __init__(
        self,
        lm: CachedCausalLM,
        board: Board,
        question_prompt: str,
        translation_prompt: str,
        n_rollouts: int = 3,
        use_score_max: bool = True,
        max_tokens: int = 32,
        verbose: bool = True,
    ):
        super().__init__()
        self.lm = lm
        self.context = LMContext(lm, question_prompt)
        self.result = None
        self.rollouts = []

        self.board = board

        self.question_prompt = question_prompt
        self.translation_prompt = translation_prompt

        self.n_rollouts = n_rollouts
        self.use_score_max = use_score_max
        self.max_tokens = max_tokens

        self.verbose = verbose

    def get_final_results(self):
        if not self.finished:
            raise RuntimeError("Inference is not finished.")
        return [self.result] + self.rollouts

    async def step(self):
        token = await self.sample(self.context.next_token())

        # Run rollouts
        results = await asyncio.gather(
            *[self.rollout(str(self.context)) for _ in range(self.n_rollouts)]
        )
        self.rollouts.extend(results)

        score_mean = np.mean([result["score"] for result in results])
        score_max = np.max([result["score"] for result in results])

        score_metric = score_max if self.use_score_max else score_mean
        self.twist(score_metric)

        if self.verbose:
            print(f"Partial question: {str(self.context)}")
            print(f"|- EIG mean: {score_mean:.4f}")
            print(f"|- EIG max: {score_max:.4f}")
            print(f"|- Particle weight: {self.weight:.4f}")
            for result in results:
                print(f"  |- Completion: {result['completion']}")
                print(f"    |- Translation: {result['translation']}")
                print(f"    |- Score: {result['score']:.4f}")
            print()

        if self.is_final_token(token) or self.is_final_context(self.context):
            translation = await self._translate_question(str(self.context))
            score = compute_score(program=translation, board=self.board)
            self.score(score)
            self.result = {
                "prefix": str(self.context),
                "completion": str(self.context),
                "translation": translation,
                "score": score,
                "type": "final",
            }
            self.finish()

    async def rollout(self, question: str):
        completion = await self._complete_question(question)
        translation = await self._translate_question(completion)
        score = compute_score(program=translation, board=self.board)
        return {
            "prefix": question,
            "completion": completion,
            "translation": translation,
            "score": score,
            "type": "rollout",
        }

    async def _complete_question(self, question: str, temp: float = 0.7):
        # Complete the question
        ctx = LMContext(self.lm, self.question_prompt + question, temp=temp)
        for _ in range(self.max_tokens):
            token = await self.sample(ctx.next_token())
            if self.is_final_token(token):
                break
            if self.is_final_context(ctx):
                break
        completion = question + str(ctx).split("\n")[0]
        completion = completion.strip()
        return completion

    async def _translate_question(self, completion: str, temp: float = 0.1):
        completion = completion.strip()

        # Translate the question to code
        translation_prompt_with_question = (
            self.translation_prompt
            + "\n"
            + TranslationPrompt.optional_space(
                TranslationPrompt.PREFIX_QUESTION, completion + "\n"
            )
            + TranslationPrompt.optional_space(TranslationPrompt.PREFIX_CODE)
            # + TranslationPrompt.optional_space(TranslationPrompt.PREFIX_CODE, "(")
        )
        ctx = LMContext(
            self.lm,
            translation_prompt_with_question,
            temp=temp,
        )
        for _ in range(self.max_tokens):
            token = await self.sample(ctx.next_token())
            if self.is_final_token(token, include_punctuation=False):
                break
            if self.is_final_context(ctx):
                break
        translation = str(ctx).split("\n")[0]
        translation = translation.strip()
        # translation = "(" + translation
        return translation

    def is_final_token(
        self, token, include_punctuation: bool = True, include_newline: bool = True
    ):
        if token.token_id == self.lm.tokenizer.eos_token_id:
            return True
        if include_punctuation and str(token) in [".", "!", "?"]:
            return True
        if include_newline and str(token) in ["\n"]:
            return True
        return False

    def is_final_context(self, context):
        if "\n" in str(context):
            return True

    def immutable_properties(self):
        return set(
            [
                "board",
                "question_prompt",
                "translation_prompt",
                "n_rollouts",
                "max_tokens",
            ]
        )


class SingleStepQuestionGenerationModel(QuestionGenerationModel):
    async def step(self):
        while True:
            token = await self.sample(self.context.next_token())
            if self.verbose:
                print(self.context)
            if self.is_final_token(token) or self.is_final_context(self.context):
                break

        translation = await self._translate_question(str(self.context))
        score = compute_score(program=translation, board=self.board)
        self.score(score)
        self.result = {
            "prefix": str(self.context),
            "completion": str(self.context),
            "translation": translation,
            "score": score,
            "type": "final",
        }
        if self.verbose:
            print(self.result)
        self.finish()
