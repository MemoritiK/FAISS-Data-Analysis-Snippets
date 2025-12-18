import json
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer


SNIPPETS_FILE = Path("..") / "snippets.jsonl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
TOP_K = 5

INDEX_DIR = Path("faiss_indices")
INDEX_DIR.mkdir(exist_ok=True)

snippets = [json.loads(line) for line in open(SNIPPETS_FILE, "r", encoding="utf-8")]
print(f"Loaded {len(snippets)} snippets")

model = SentenceTransformer(EMBEDDING_MODEL)


def build_index(texts, index_path):
    embeddings_path = index_path.with_suffix(".npy")
    
    if embeddings_path.exists() and index_path.exists():
        print(f"Loading cached index: {index_path}")
        index = faiss.read_index(str(index_path))
        q_embeddings = np.load(embeddings_path)
    else:
        # Generate embeddings and create FAISS index
        print(f"Generating embeddings for {len(texts)} items...")
        q_embeddings = model.encode(texts, convert_to_numpy=True, batch_size=BATCH_SIZE, show_progress_bar=True)
        q_embeddings = q_embeddings / np.linalg.norm(q_embeddings, axis=1, keepdims=True)
        
        d = q_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # cosine similarity
        index.add(q_embeddings)
        
        np.save(embeddings_path, q_embeddings)
        faiss.write_index(index, str(index_path))
        print(f"Saved index to {index_path} and embeddings to {embeddings_path}")
    return index, q_embeddings


texts_query = [s.get("question", "") for s in snippets]
texts_code = [s.get("code", "") for s in snippets]
texts_tags = [" ".join(s.get("tags", [])) for s in snippets]

index_query, _ = build_index(texts_query, INDEX_DIR / "query.index")
index_code, _ = build_index(texts_code, INDEX_DIR / "code.index")
index_tags, _ = build_index(texts_tags, INDEX_DIR / "tags.index")

metadata_file = INDEX_DIR / "metadata.pkl"
if not metadata_file.exists():
    with open(metadata_file, "wb") as f:
        pickle.dump(snippets, f)
    print(f"Saved metadata to {metadata_file}")