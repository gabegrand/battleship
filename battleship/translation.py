import os

from transformers import AutoTokenizer, AutoModelForCausalLM


class Translator(object):
    def __init__(self, model_name: str = "bigcode/starcoder"):
        hf_auth_token = os.environ.get("HF_AUTH_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_auth_token)

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)