"""
search_app.py
Streamlit UI for semantic image search with FAISS index + CLIP embeddings
"""

import streamlit as st
import sqlite3, json, os
import numpy as np
from PIL import Image
import torch
import faiss
from transformers import CLIPModel, CLIPProcessor

DB_PATH = "metadata.db"
FAISS_PATH = "vectors.faiss"
IDMAP_PATH = "id_to_filename.json"


@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device


@st.cache_data
def load_index():
    if not os.path.exists(FAISS_PATH):
        return None, None
    index = faiss.read_index(FAISS_PATH)
    with open(IDMAP_PATH, "r") as f:
        id2fn = json.load(f)
    return index, id2fn


@st.cache_data
def load_metadata():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT filename, path, width, height, format, exif, dominant_colors, llm_caption FROM images"
    ).fetchall()
    conn.close()
    meta = {
        r[0]: {
            "path": r[1],
            "width": r[2],
            "height": r[3],
            "format": r[4],
            "exif": json.loads(r[5]) if r[5] else {},
            "colors": json.loads(r[6]) if r[6] else [],
            "caption": r[7],
        }
        for r in rows
    }
    return meta


def embed_text(query, model, processor, device):
    inputs = processor(text=[query], return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy()


def show_results(filenames, metadata, top_k=20):
    cols = 4
    for i, fn in enumerate(filenames[:top_k]):
        if i % cols == 0:
            cols_row = st.columns(cols)
        meta = metadata.get(fn, {})
        path = meta.get("path")
        try:
            img = Image.open(path)
            cols_row[i % cols].image(img, caption=fn, use_container_width=True)
            if meta.get("caption"):
                cols_row[i % cols].markdown(f"📝 {meta['caption']}")
        except Exception as e:
            cols_row[i % cols].write(f"Error: {e}")


st.set_page_config(page_title="🔎 Local Image Search", layout="wide")
st.title("🔎 Local Semantic Image Search (CLIP + FAISS)")

model, processor, device = load_clip()
index, id2fn = load_index()
metadata = load_metadata()

query = st.text_input("Search images (natural language):", "")
num_results = st.slider("Results", 4, 64, 16)

if st.button("Search") and query.strip():
    if index is None:
        st.warning("No FAISS index found — run build_index.py first.")
    else:
        q_emb = embed_text(query, model, processor, device)
        D, I = index.search(q_emb.astype("float32"), k=num_results)
        filenames = [id2fn[idx] for idx in I[0]]
        show_results(filenames, metadata, top_k=num_results)

st.markdown("---")
st.subheader("📂 Browse all images")
if st.button("Show all thumbnails"):
    show_results(list(metadata.keys()), metadata, top_k=48)
