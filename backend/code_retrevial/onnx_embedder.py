import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path

MODEL_DIR = Path("onnx_model")
MODEL_PATH = MODEL_DIR / "model.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

MAX_LENGTH = 256


session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))


def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask[..., None]
    summed = (token_embeddings * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    return summed / np.clip(counts, a_min=1e-9, a_max=None)


def embed_text(texts, batch_size=32):
    # Normalize input
    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        input_ids_batch = []
        attention_mask_batch = []

        for text in batch_texts:
            encoded = tokenizer.encode(text)

            input_ids = encoded.ids[:MAX_LENGTH]
            attention_mask = [1] * len(input_ids)

            pad_len = MAX_LENGTH - len(input_ids)
            if pad_len > 0:
                input_ids += [0] * pad_len
                attention_mask += [0] * pad_len

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)

        ort_inputs = {
            "input_ids": np.array(input_ids_batch, dtype=np.int64),
            "attention_mask": np.array(attention_mask_batch, dtype=np.int64),
        }

        # ONNX inference
        token_embeddings = session.run(None, ort_inputs)[0]
        # (batch, seq_len, hidden)

        # Mean pooling
        mask = ort_inputs["attention_mask"][:, :, None]
        pooled = np.sum(token_embeddings * mask, axis=1)
        pooled /= np.clip(mask.sum(axis=1), 1e-9, None)

        # L2 normalize
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)

        all_embeddings.append(pooled)

    return np.vstack(all_embeddings)
