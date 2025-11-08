"""
build_index.py
Builds a FAISS inner-product (cosine) index from embeddings stored in metadata.db
"""

import sqlite3, os, json
import numpy as np
import faiss

DB_PATH = "metadata.db"
FAISS_PATH = "vectors.faiss"
IDMAP_PATH = "id_to_filename.json"


def load_embeddings(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute("SELECT filename, embedding FROM embeddings").fetchall()
    conn.close()
    items = []
    for fn, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        items.append((fn, emb))
    return items


def build_index(items, faiss_path=FAISS_PATH):
    if len(items) == 0:
        raise RuntimeError("No embeddings found.")
    dim = items[0][1].shape[0]
    xb = np.vstack([v for _, v in items]).astype("float32")

    # FAISS inner-product index (cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(xb)
    faiss.write_index(index, faiss_path)

    id_map = [fn for fn, _ in items]
    with open(IDMAP_PATH, "w") as f:
        json.dump(id_map, f)

    print(f"✅ Built FAISS index ({len(items)} items, dim={dim}) → {faiss_path}")


if __name__ == "__main__":
    items = load_embeddings()
    build_index(items)
