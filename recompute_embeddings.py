"""
Run this ONCE on your Mac to recompute embeddings with the small model.
It reads your existing merged_embeddings.json, keeps all text/source/file/language
fields, and replaces the embedding vectors with the small model's vectors.

Usage:
    pip install sentence-transformers
    python recompute_embeddings.py
"""

import json
from sentence_transformers import SentenceTransformer

INPUT_FILE  = "merged_embeddings.json"
OUTPUT_FILE = "merged_embeddings.json"   # overwrites in place
NEW_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"  # ~200MB, free-tier safe

print(f"📂 Loading {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} chunks")
print(f"🤖 Loading model {NEW_MODEL}...")
model = SentenceTransformer(NEW_MODEL)

texts = [item["text"] for item in data]

print("⚙️  Recomputing embeddings (this takes a few minutes)...")
new_embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
)

print("💾 Saving new embeddings...")
for i, item in enumerate(data):
    item["embedding"] = new_embeddings[i].tolist()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print(f"✅ Done! Saved {len(data)} chunks to {OUTPUT_FILE}")
print(f"   New embedding size: {len(new_embeddings[0])} dimensions")
