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


def embed_text(text: str) -> np.ndarray:
    encoded = tokenizer.encode(text)

    input_ids = encoded.ids[:MAX_LENGTH]
    attention_mask = [1] * len(input_ids)

    pad_len = MAX_LENGTH - len(input_ids)
    if pad_len > 0:
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    ort_inputs = {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "attention_mask": np.array([attention_mask], dtype=np.int64),
    }

    outputs = session.run(None, ort_inputs)
    token_embeddings = outputs[0]  # (1, seq_len, hidden)

    sentence_embedding = mean_pooling(
        token_embeddings,
        ort_inputs["attention_mask"]
    )

    # normalize (must match build-time)
    sentence_embedding /= np.linalg.norm(sentence_embedding, axis=1, keepdims=True)

    return sentence_embedding
