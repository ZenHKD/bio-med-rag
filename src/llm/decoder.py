"""
LLM Decoder: Qwen/Qwen3.5-4B loaded in 4-bit quantization (bitsandbytes NF4).

Usage:
    decoder = Decoder()
    answer = decoder.generate(context="...", question="A. ... B. ...")
"""

import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROMPT = str(PROJECT_ROOT / "scripts" / "prompt.txt")
DEFAULT_MODEL  = "Qwen/Qwen3.5-4B"
DEFAULT_MAX_NEW_TOKENS = 64    # enough for answer letter + surrounding text


def _load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

class Decoder:
    """
    Wraps a HuggingFace causal LM for single-letter MCQ answer generation.

    The model is loaded once in 4-bit NF4 quantization (bitsandbytes) to
    minimize VRAM usage.  After benchmarking, swap this class with the
    vLLM-backed implementation in src/serving/.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        prompt_path: str = DEFAULT_PROMPT,
        max_new_tokens: int = None,
        thinking: bool = False,
    ):
        self.model_name     = model_name
        self.prompt_template = _load_prompt(prompt_path)
        self.thinking       = thinking
        # Auto-scale token budget: thinking needs room for full reasoning chain
        if max_new_tokens is None:
            self.max_new_tokens = 2048 if thinking else 64
        else:
            self.max_new_tokens = max_new_tokens

        print(f"[Decoder] Loading {model_name} in 4-bit NF4...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 GB
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        ).eval()

    # ------------------------------------------------------------------
    def _build_messages(self, context: str, question: str) -> list:
        """Format context + question into the prompt template."""
        user_content = self.prompt_template.format(
            context=context.strip(),
            question=question.strip(),
        )
        return [{"role": "user", "content": user_content}]

    # ------------------------------------------------------------------
    def generate(self, context: str, question: str) -> str:
        """
        Generate an MCQ answer letter (A–E).

        Args:
            context:  Concatenated reranked document texts.
            question: Formatted question string (already includes A/B/C/D options).

        Returns:
            Single uppercase letter A–E, or "?" if extraction fails.
        """
        messages = self._build_messages(context, question)

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=self.thinking,
        )
        input_ids      = tokenized["input_ids"].to(self.model.device)
        attention_mask = tokenized["attention_mask"].to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

        thinking_text = ""
        answer_text   = raw

        if self.thinking:
            if "</think>" in raw:
                # Standard Qwen3 format: <think>...</think>\nAnswer
                parts         = raw.split("</think>", 1)
                thinking_text = parts[0].replace("<think>", "").strip()
                answer_text   = parts[1].strip()
            else:
                # Qwen3.5-4B: outputs plain reasoning text, no tags
                # Full raw IS the thinking; look for the answer at the very end
                thinking_text = raw
                answer_text   = raw[-300:]   # answer letter should be near the end

        return {
            "answer":  self._parse_answer(answer_text),
            "thinking": thinking_text,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_answer(text: str, thinking: bool = False) -> str:
        """Extract the answer letter (A–E) from model output.

        When thinking=True the output is '<think>...</think>\nA'.
        We strip the think block first so we don't match letters inside it.
        """
        if thinking and "</think>" in text:
            # Take everything after the closing think tag
            text = text.split("</think>", 1)[-1]

        match = re.search(r"\b([A-E])\b", text.upper())
        return match.group(1) if match else "?"
