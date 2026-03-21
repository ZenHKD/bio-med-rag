from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm


class Encoder():
    def __init__(self, texts, batch_size, model_name, device):
        self.texts = texts
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = device
        self.tokenizer, self.model = self.get_embed_model()

    def get_embed_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        model = AutoModel.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,  # guaranteed BF16 weights
            device_map=self.device
        ).eval()
        return tokenizer, model

    @staticmethod
    def _last_token_pool(hidden_states, attention_mask):
        """Last non-padding token pooling — correct for causal/decoder embedders."""
        seq_len = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_len]

    def encode(self):
        all_embeddings = []

        for i in tqdm(
            range(0, len(self.texts), self.batch_size),
            desc="Encoding",
        ):
            batch = self.texts[i: i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=4096,     
                return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                with torch.autocast(self.device, dtype=torch.bfloat16):
                    outputs = self.model(**encoded)

            embeddings = self._last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)  # float32 for FAISS
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string — used at search time (no tqdm)."""
        encoded = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            with torch.autocast(self.device, dtype=torch.bfloat16):
                outputs = self.model(**encoded)

        embedding = self._last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embedding = F.normalize(embedding.float(), p=2, dim=-1)
        return embedding.cpu().numpy()