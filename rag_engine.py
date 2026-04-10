"""
rag_engine.py  –  RAG logic extracted from the notebook.
API keys are read from environment variables (never hardcode them).
"""

import json
import numpy as np
import re
import time
import os
from sentence_transformers import SentenceTransformer
from groq import Groq, RateLimitError, APIError

# ═══════════════════════════════════════════════
# CONFIG  (values come from environment variables)
# ═══════════════════════════════════════════════
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "merged_embeddings.json")
MODEL_NAME      = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL       = os.getenv("LLM_MODEL",   "llama-3.3-70b-versatile")
TOP_K           = int(os.getenv("TOP_K",   "3"))
CANDIDATES      = int(os.getenv("CANDIDATES", "20"))

# Read Groq keys from env  (GROQ_KEY_1 … GROQ_KEY_6)
GROQ_API_KEYS = [
    v for k, v in sorted(os.environ.items())
    if k.startswith("GROQ_KEY_") and v.strip()
]
if not GROQ_API_KEYS:
    raise RuntimeError(
        "No Groq API keys found. Set GROQ_KEY_1, GROQ_KEY_2 … in environment variables."
    )

# ── globals filled by load_data() ──────────────
texts      = []
sources    = []
embeddings = None
embedder   = None
key_manager = None


# ═══════════════════════════════════════════════
# API KEY MANAGER  (round-robin + fallback)
# ═══════════════════════════════════════════════
class GroqKeyManager:
    def __init__(self, keys):
        self.keys  = keys
        self.index = 0

    def _next_key(self):
        key = self.keys[self.index]
        self.index = (self.index + 1) % len(self.keys)
        return key

    def call_with_fallback(self, messages, model=None, **kwargs):
        model = model or LLM_MODEL
        tried = []
        for _ in range(len(self.keys)):
            key = self._next_key()
            tried.append(key[:8] + "...")
            client = Groq(api_key=key)
            try:
                return client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
            except (RateLimitError, APIError) as e:
                print(f"⚠️  Key {key[:8]}… failed: {e}. Trying next…")
                time.sleep(0.5)
        raise Exception(f"All {len(self.keys)} API keys failed. Tried: {tried}")


# ═══════════════════════════════════════════════
# STARTUP  – call once from FastAPI lifespan
# ═══════════════════════════════════════════════
def load_data():
    global texts, sources, embeddings, embedder, key_manager

    print("📂 Loading embeddings…")
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts    = [_clean(x["text"])           for x in data]
    sources  = [x.get("file", "unknown")    for x in data]
    raw_emb  = np.array([x["embedding"]     for x in data], dtype="float32")
    norms    = np.linalg.norm(raw_emb, axis=1, keepdims=True)
    embeddings = raw_emb / norms
    print(f"✅ Loaded {len(texts)} chunks")

    print("🤖 Loading sentence-transformer model…")
    embedder    = SentenceTransformer(MODEL_NAME)
    key_manager = GroqKeyManager(GROQ_API_KEYS)
    print("✅ Model ready")


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════
def _clean(text: str) -> str:
    text = text.replace('\u200b', '').replace('\u200c', '')
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _expand_query(query: str):
    return [
        query,
        f"definition: {query}",
        f"explanation: {query}",
        f"what is {query}",
        f"procedure for {query}",
    ]


def _search(query: str):
    scores = np.zeros(len(texts), dtype=np.float32)
    for q in _expand_query(query):
        vec = embedder.encode(q, normalize_embeddings=True).astype("float32")
        scores += embeddings @ vec
    top_idx = np.argsort(scores)[::-1][:CANDIDATES]
    return [{"text": texts[i], "source": sources[i], "score": float(scores[i])}
            for i in top_idx]


def _rerank(query: str, chunks: list):
    context = "\n\n".join(f"[{i}] {c['text'][:300]}" for i, c in enumerate(chunks))
    prompt  = (
        "You are a retrieval system.\n"
        "Pick the MOST relevant passages for answering the question.\n\n"
        f"Question: {query}\n\nPassages:\n{context}\n\n"
        "Return only indices like: 0,2,1,3"
    )
    try:
        resp = key_manager.call_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=50,
        )
        idxs = list(map(int, re.findall(r"\d+", resp.choices[0].message.content)))
        return [chunks[i] for i in idxs[:TOP_K]]
    except Exception:
        return chunks[:TOP_K]


def _build_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join(
        f"[{i+1}] ({c['source']})\n{c['text']}" for i, c in enumerate(chunks)
    )
    return (
        "You are a professional HR/legal assistant for an Algerian company.\n\n"
        "**CRITICAL INSTRUCTIONS:**\n"
        "- The context chunks may be in Arabic, French, English, or any other language.\n"
        "- You MUST read and understand EVERY chunk, regardless of its language.\n"
        "- Answer **only in the same language as the user's question**.\n"
        "- If information is missing from the context, say so clearly.\n\n"
        f"**Context:**\n{context}\n\n"
        f"**Question (answer in the same language as this question):**\n{query}\n\n"
        "**Answer:**"
    )


# ═══════════════════════════════════════════════
# PUBLIC  –  main RAG pipeline
# ═══════════════════════════════════════════════
def rag(query: str, verbose: bool = False) -> str:
    chunks = _search(query)
    chunks = _rerank(query, chunks)
    answer = key_manager.call_with_fallback(
        messages=[{"role": "user", "content": _build_prompt(query, chunks)}],
        temperature=0.1, max_tokens=800,
    )
    result = answer.choices[0].message.content.strip()
    if verbose:
        print(f"\nQUERY: {query}")
        for c in chunks:
            print(f"  [{c['score']:.3f}] {c['source']}")
    return result
