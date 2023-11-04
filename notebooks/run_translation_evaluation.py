import os

import pandas as pd
from eig import compute_eig_fast
from eig.battleship import BattleshipHypothesis
from eig.battleship import Executor
from eig.battleship import Parser
from eig.battleship import Ship
from eig.battleship.program import ProgramSyntaxError
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from battleship.translator import Translator

# Load HF_AUTH_TOKEN from .hf_auth_token
with open(os.path.join(os.path.dirname(__file__), "../", ".hf_auth_token"), "r") as f:
    os.environ["HF_AUTH_TOKEN"] = f.read().strip()

MODEL_NAMES = [
    "bigcode/starcoder",
    "codellama/CodeLlama-7b-hf",
    "WizardLM/WizardCoder-15B-V1.0",
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_AUTH_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_AUTH_TOKEN,
            device_map="auto",
            load_in_8bit=True,
        )
        translator = Translator(model=model, tokenizer=tokenizer)

        df_test[f"question-{model_name}"] = [
            translator.code_to_question(text) for text in tqdm(df_test["code"])
        ]
        df_test[f"code-{model_name}"] = [
            translator.question_to_code(text) for text in tqdm(df_test["question"])
        ]
        df_test[f"parseable-{model_name}"] = [
            check_parse(p) for p in df_test["completion"]
        ]
        df_test[f"exact_match-{model_name}"] = [
            p == a for p, a in zip(df_test["completion"], df_test["code"])
        ]

        df_test.to_csv(
            "../battleship/prompts/test_examples_with_completions.csv", index=False
        )


def check_parse(program):
    try:
        Parser.parse(program)
        return True
    except ProgramSyntaxError:
        return False


if __name__ == "__main__":
    main()
