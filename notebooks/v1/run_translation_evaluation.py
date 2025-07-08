import os

import pandas as pd
from tqdm import tqdm

from battleship.translator import Translator
from eig.battleship import Parser
from eig.battleship.program import ProgramSyntaxError

# Load HF_AUTH_TOKEN from .hf_auth_token
with open(os.path.join(os.path.dirname(__file__), "../", ".hf_auth_token"), "r") as f:
    os.environ["HF_AUTH_TOKEN"] = f.read().strip()

MODEL_NAMES = [
    # "bigcode/starcoder",
    "codellama/CodeLlama-7b-hf",
    # "WizardLM/WizardCoder-15B-V1.0",
]


def main():
    HF_AUTH_TOKEN = os.environ["HF_AUTH_TOKEN"]

    df_test = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "../battleship/prompts/test_examples.csv"
        )
    )

    for model_name in MODEL_NAMES:
        print(model_name)
        translator = Translator(model_name=model_name)

        df_test[f"question-{model_name}"] = [
            translator.code_to_question(text) for text in tqdm(df_test["code"])
        ]
        df_test[f"code-{model_name}"] = [
            translator.question_to_code(text) for text in tqdm(df_test["question"])
        ]
        df_test[f"parseable-{model_name}"] = [
            check_parse(p) for p in df_test[f"code-{model_name}"]
        ]
        df_test[f"exact_match-{model_name}"] = [
            p == a for p, a in zip(df_test[f"code-{model_name}"], df_test["code"])
        ]

        df_test.to_csv("test_examples_with_completions.csv", index=False)


def check_parse(program):
    try:
        Parser.parse(program)
        return True
    except ProgramSyntaxError:
        return False


if __name__ == "__main__":
    main()
