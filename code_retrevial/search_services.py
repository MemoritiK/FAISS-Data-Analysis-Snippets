from onnx_embedder import embed_text
from pathlib import Path
import faiss
import numpy as np
import json

SNIPPETS_FILE = Path("..") /"cleaned_snippets.jsonl"
TOP_K=3

INDEX_DIR = Path("faiss_indices")
snippets = [json.loads(line) for line in open(SNIPPETS_FILE, "r", encoding="utf-8")]

def get_index(index_path):
    embeddings_path = index_path.with_suffix(".npy")
    index = faiss.read_index(str(index_path))
    q_embeddings = np.load(embeddings_path)
    return index, q_embeddings

        
index_query, _ = get_index(INDEX_DIR / "query.index")
index_code, _ = get_index(INDEX_DIR / "code.index")
index_tags, _ = get_index(INDEX_DIR / "tags.index")

def detect_mode(text):
    if text.count(",") >= 2 and len(text.split()) < 4:
        return 3

    code_tokens = ["def ", "class ", "{", "}", "};", "==", "->", "#include", "import ", "for(", "while(",":","[","]"]
    if any(tok in text for tok in code_tokens):
        return 2

    return 1

def search_snippets(input_text, top_k=TOP_K, difficulty="all"):
    """
    Search snippets by mode and optional difficulty.
    mode: 1=query, 2=code, 3=tags
    difficulty: 'easy', 'medium', 'hard', or 'all'
    """
    mode = detect_mode(input_text)
    if mode == 1:
        index = index_query
    elif mode == 2:
        index = index_code
    elif mode == 3:
        index = index_tags
    else:
        raise ValueError("mode must be 1, 2, or 3")
    
    # Encode input
    q_emb = embed_text(input_text)
    
    D, I = index.search(q_emb, top_k)
    
    results = []
    for idx, score in zip(I[0], D[0]):
        snippet = snippets[idx].copy()
        snippet["score"] = float(score)
        results.append(snippet)
    
    if difficulty != "all":
        results = [r for r in results if r.get("difficulty") == difficulty]
    
    return results


if __name__ == "__main__":    
    query_text = '''df = pd.DataFrame({ 
        "A": [1, 2, None, 4],
        "B": [None, 2, 3, 4],
        "C": ["x", None, "y", "z"]
    })
    
    # Drop rows with any NaN
    df_clean = df.drop"'''
    results = search_snippets(query_text, top_k=3)
    
    for r in results:
        print(f"Category: {r['category']}, Score: {r['score']:.3f}")
        print(f"Question: {r['question']}")
        print("Difficulty:", r["difficulty"])
        print("Code:\n", r['code'].replace("\\n", "\n"))
        print("-"*50,"\n")