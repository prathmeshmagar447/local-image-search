"""
build_index.py
Builds a FAISS index (FlatIP or IVF) from embeddings stored in metadata.db
Refactored to use config and support larger datasets.
"""

import sqlite3, os, json
import numpy as np
import faiss
import yaml
from loguru import logger

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup logging
logger.add("image_search.log", rotation="10 MB", level="INFO")

DB_PATH = config["db_path"]
FAISS_PATH = config["faiss_path"]
IDMAP_PATH = config["idmap_path"]
INDEX_TYPE = config["faiss_index_type"]
IVF_NLIST = config["ivf_nlist"]
HNSW_M = config.get("hnsw_m", 32)  # HNSW parameter: number of neighbors
HNSW_EF_CONSTRUCTION = config.get("hnsw_ef_construction", 200)  # HNSW build parameter
HNSW_EF_SEARCH = config.get("hnsw_ef_search", 128)  # HNSW search parameter


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

    # Choose index type based on configuration and dataset size
    if INDEX_TYPE == "hnsw":
        # HNSW index for best search performance
        index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = HNSW_EF_SEARCH
        logger.info(f"Building HNSW index with {len(items)} items (M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION})...")
        index.add(xb)
        index_type_used = "hnsw"
    elif INDEX_TYPE == "ivf" or len(items) > 10000:
        # IVF index for larger datasets
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(IVF_NLIST, len(items)))
        logger.info(f"Training IVF index with {len(items)} items...")
        index.train(xb)
        index.add(xb)
        index_type_used = "ivf"
    else:
        # Flat index for smaller datasets - exact search
        index = faiss.IndexFlatIP(dim)
        index.add(xb)
        index_type_used = "flat"

    # Save index metadata for search optimization
    index_metadata = {
        "type": index_type_used,
        "dimension": dim,
        "num_vectors": len(items),
        "hnsw_m": HNSW_M if index_type_used == "hnsw" else None,
        "hnsw_ef_search": HNSW_EF_SEARCH if index_type_used == "hnsw" else None,
        "ivf_nlist": IVF_NLIST if index_type_used == "ivf" else None,
    }

    faiss.write_index(index, faiss_path)

    # Save metadata alongside index
    metadata_path = faiss_path + ".meta.json"
    with open(metadata_path, "w") as f:
        json.dump(index_metadata, f, indent=2)

    id_map = [fn for fn, _ in items]
    with open(IDMAP_PATH, "w") as f:
        json.dump(id_map, f)

    logger.info(f"✅ Built FAISS {index_type_used.upper()} index ({len(items)} items, dim={dim}) → {faiss_path}")
    logger.info(f"Index metadata saved to {metadata_path}")


if __name__ == "__main__":
    items = load_embeddings()
    build_index(items)
