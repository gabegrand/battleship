import os

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class Translator(object):
    def __init__(
        self,
        model_name: str = "Salesforce/codegen-350M-multi",
        max_new_tokens: int = 128,
    ):
        hf_auth_token = os.environ.get("HF_AUTH_TOKEN")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_auth_token
        )

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
