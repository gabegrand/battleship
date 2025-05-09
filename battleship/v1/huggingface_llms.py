"""Wrapper classes for HuggingFace models."""
import logging
import os
from typing import List
from typing import Union

import pandas as pd
import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList

from battleship.v1.prompting import BasePrompt


class HuggingFaceModel(object):
    def __init__(
        self,
        model_name: str = None,
        model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None,
        max_new_tokens: int = 128,
        stop_words: list = ["\n\n"],
        use_bettertransformer: bool = True,
        load_in_8bit: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.stop_words = stop_words

        # Load model from HuggingFace Hub
        if model_name:
            tokenizer, model = self.load_tokenizer_and_model(
                model_name=model_name,
                use_bettertransformer=use_bettertransformer,
                load_in_8bit=load_in_8bit,
            )
        else:
            assert model is not None
            assert tokenizer is not None
        self.model, self.tokenizer = model, tokenizer

    @staticmethod
    def load_tokenizer_and_model(
        model_name: str,
        use_bettertransformer: bool = True,
        load_in_8bit: bool = True,
    ):
        if not torch.cuda.is_available():
            logging.warning(
                "Warning: CUDA is not available. Model will be loaded on CPU."
            )
        hf_auth_token = os.environ.get("HF_AUTH_TOKEN")
        if not hf_auth_token:
            logging.warning(
                "Warning: HF_AUTH_TOKEN environment variable is not set. "
                "This may cause issues when loading models from the HuggingFace Hub."
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_auth_token,
            device_map="auto",
            load_in_8bit=load_in_8bit,
        )
        if use_bettertransformer:
            model = BetterTransformer.transform(model, keep_original_model=False)
        return tokenizer, model

    def __call__(self, prompts: List[str]) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(
            device=self.model.device
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [
                    EndOfFunctionCriteria(
                        inputs["input_ids"].shape[1], self.stop_words, self.tokenizer
                    )
                ]
            ),
        )

        # Return only the completion
        completions = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        # Remove everything after the first newline and strip whitespace
        completions = [completion.split("\n")[0].strip() for completion in completions]
        return completions


class EndOfFunctionCriteria(StoppingCriteria):
    """Adapted from: github.com/benlipkin/linc

    Custom `StoppingCriteria` which checks if all generated functions in the batch are completed.
    """

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)
