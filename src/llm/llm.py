from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class QwenMCQ:

    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", device="cuda"):

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt, max_new_tokens=10):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        input_length = inputs["input_ids"].shape[1]

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_tokens = output[0][input_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return response.strip()

    def predict(self, prompt):

        response = self.generate(prompt)

        match = re.search(r"[ABCD]", response)

        return match.group(0) if match else "UNKNOWN"