from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re


class QwenMCQ:

    def __init__(self, model_name="Qwen/Qwen3.5-4B"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt, max_new_tokens=20):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_tokens = outputs[0][input_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return response.strip()

    def predict(self, prompt):

        response = self.generate(prompt)

        # normalize
        response = response.upper()

        # extract A/B/C/D
        match = re.search(r"\b([ABCD])\b", response)

        if match:
            return match.group(1)

        # fallback: first char
        if response and response[0] in ["A", "B", "C", "D"]:
            return response[0]

        return "UNKNOWN"