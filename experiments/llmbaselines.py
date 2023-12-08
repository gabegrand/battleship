import io
import os
from base64 import b64encode
from enum import StrEnum

import matplotlib.pyplot as plt
import numpy as np
import standard_prompts
from eig import *
from eig.battleship.program import ProgramSyntaxError
from openai import OpenAI

from battleship.board import *
from battleship.scoring import *
from battleship.translator import *


class OpenAIModels(StrEnum):
    TEXT = "gpt-4"
    VISION = "gpt-4-vision-preview"


client = OpenAI()

prompt_base_translator = standard_prompts.constructPrompt(
    standard_prompts.translator_constant, ["examples.csv", "additional_examples.csv"]
)


def get_completion(model_used, system_prompt, inputBoard, n_completions):
    if model_used != OpenAIModels.VISION:
        completion = client.chat.completions.create(
            model=model_used,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(inputBoard)},
            ],
            n=n_completions,
        )
    else:
        completion = client.chat.completions.create(
            model=model_used,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{str(inputBoard)}"
                            },
                        }
                    ],
                },
            ],
            n=n_completions,
        )
    return completion


def run_trial(context_file, mode, n_completion):
    with open(f"responses/responses_{context_file}_{mode}.txt", "w+") as saved:
        contextBoard = Board.from_trial_id(context_file)

        questions = []
        lines = []

        if mode == "ascii":
            inputBoard = Board(contextBoard)  # ASCII representation
            prompt = standard_prompts.ascii_prompt
        if mode == "serial":
            inputBoard = Board(
                contextBoard
            ).to_textual_representation()  # Serialized representation
            prompt = standard_prompts.serial_prompt
        if mode == "vision":  # Image representation
            inputBoard = Board(contextBoard).to_figure()
            IObytes = io.BytesIO()
            inputBoard.savefig(
                IObytes, format="jpg"
            )  # Saves figure to buffer as to avoid unnecessary I/O on disk
            IObytes.seek(0)
            inputBoard = b64encode(IObytes.read()).decode("utf-8")
            prompt = standard_prompts.vision_prompt
        saved.write(f"{prompt}\n{inputBoard}\n")

        if mode != "vision":
            completion = get_completion(
                OpenAIModels.TEXT, prompt, inputBoard, n_completion
            )
        else:
            completion = get_completion(
                OpenAIModels.VISION, prompt, inputBoard, n_completion
            )

        for i in range(0, n_completion):
            question = str(completion.choices[i].message.content)
            questions.append(question)
            saved.write(question + "\n")

            completion_translation = get_completion(
                OpenAIModels.TEXT, prompt_base_translator, questions[i], 1
            )
            saved.write(prompt_base_translator + "\n")
            question_code = (
                str(completion_translation.choices[0].message.content)
                .strip()
                .replace('"', "")
            )
            saved.write(question_code + "\n")
            score = compute_score(contextBoard, question_code)

            lines.append([questions[i], question_code, str(score)])

    return lines


def run_all_trials(mode, max_context, n_completion):
    os.makedirs("responses", exist_ok=True)
    with open(f"responses_{mode}.txt", "w+") as responseFile:
        for i in range(1, max_context + 1):
            response = [f"Context {i}"]
            llm_response = run_trial(i, mode, n_completion)
            for i in range(0, n_completion):
                response_write = response + llm_response[i]
                responseFile.write("\t".join(response_write) + "\n")


if __name__ == "__main__":
    scores = []
    mode, max_context, completions = "serial", 18, 10
    run_all_trials(mode, max_context, completions)
    with open(f"responses_{mode}.txt", "r") as responseFile:
        for line in responseFile:
            score = float(line.split("\t")[3].strip())
            scores.append(score)
    plt.hist(scores, [-1, -0.5, -0.1, 0, 0.25, 0.5, 0.75, 1, 1.25, 2, 3, 4, 5])
    plt.savefig(f"responses_{mode}.svg")
