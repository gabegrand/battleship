import os

import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class Translator(object):
    EXAMPLES_PATH = "prompts/examples.csv"

    QUESTION = "question"
    CODE = "code"

    def __init__(
        self,
        model_name: str = "Salesforce/codegen-350M-multi",
        max_new_tokens: int = 128,
        stop_tokens: list = ["\n"],
    ):
        hf_auth_token = os.environ.get("HF_AUTH_TOKEN")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_auth_token
        )

        stop_token_ids = [self.tokenizer(t)["input_ids"] for t in stop_tokens]
        for token, token_ids in zip(stop_tokens, stop_token_ids):
            if len(token_ids) != 1:
                raise ValueError(
                    f"Stop token {token} has length {len(token_ids)}, but should have length 1."
                )
        self.stop_token_ids = [t[0] for t in stop_token_ids] + [
            self.tokenizer.eos_token_id
        ]

        self.df_examples = self.load_examples(
            os.path.join(os.path.dirname(__file__), self.EXAMPLES_PATH)
        )

    def __call__(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.stop_token_ids,
        )

        # Return only the completion
        completion = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        completion = completion.strip().replace(
            "\n", ""
        )  # Remove leading space and newline
        return completion

    def question_to_code(self, question: str) -> str:
        return self._translate(
            text=question,
            user_input_col=Translator.QUESTION,
            response_col=Translator.CODE,
        )

    def code_to_question(self, code: str) -> str:
        return self._translate(
            text=code, user_input_col=Translator.CODE, response_col=Translator.QUESTION
        )

    def _translate(self, text: str, user_input_col: str, response_col: str) -> str:
        prompt = "\n\n".join(
            self.df_examples.apply(
                lambda row: self.format_example(
                    user_input=row[user_input_col], response=row[response_col]
                ),
                axis=1,
            )
        )
        prompt += "\n\n"
        prompt += self.format_example(user_input=text)
        return self(prompt)

    def load_examples(self, path: str):
        return pd.read_csv(path)

    @staticmethod
    def format_example(user_input: str, response: str = None):
        return f"User: {user_input}\n" f"Assistant:{' ' + response if response else ''}"
