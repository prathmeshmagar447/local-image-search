"""
search_app.py
Streamlit UI for hybrid semantic image search with FAISS index + CLIP embeddings
Refactored for filters, pagination, and reverse image search.
"""

import streamlit as st
import sqlite3, json, os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import faiss
from transformers import CLIPModel, CLIPProcessor
import yaml
from loguru import logger
import requests
from collections import Counter

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DB_PATH = config["db_path"]
FAISS_PATH = config["faiss_path"]
IDMAP_PATH = config["idmap_path"]
DEFAULT_RESULTS = config["default_results"]
MAX_RESULTS = config["max_results"]
CLIP_MODEL_NAME = config["clip_model"]
DEVICE_CONFIG = config["device"]
CLIP_USE_FAST = config.get("clip_use_fast", True)
CLIP_MAX_LENGTH = config.get("clip_max_length", 77)


@st.cache_resource
def load_clip():
    if DEVICE_CONFIG == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE_CONFIG
    logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME} on device: {device}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=CLIP_USE_FAST)
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
        "SELECT filename, path, file_size, modified_time, width, height, format, exif, dominant_colors, llm_caption FROM images"
    ).fetchall()
    conn.close()
    meta = {}
    for r in rows:
        caption_data = json.loads(r[9]) if r[9] else {"caption": "", "tags": []}
        meta[r[0]] = {
            "path": r[1],
            "file_size": r[2],
            "modified_time": r[3],
            "width": r[4],
            "height": r[5],
            "format": r[6],
            "exif": json.loads(r[7]) if r[7] else {},
            "colors": json.loads(r[8]) if r[8] else [],
            "caption": caption_data.get("caption", ""),
            "tags": caption_data.get("tags", []),
        }
    return meta

@st.cache_data
def load_clusters():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        rows = cur.execute("SELECT id, label FROM clusters").fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}
    except sqlite3.OperationalError:
        conn.close()
        return {}



def embed_text(query, model, processor, device):
    inputs = processor(text=[query], return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy()


def embed_image(img, model, processor, device):
    """Enhanced image embedding with better preprocessing for CLIP."""
    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # CLIP works best with images around 224x224, but let's use the model's expected size
    if hasattr(model.config, 'vision_config'):
        expected_size = getattr(model.config.vision_config, 'image_size', 224)
        # Resize maintaining aspect ratio, then center crop
        img = img.resize((expected_size, expected_size), Image.BICUBIC)

    inputs = processor(images=[img], return_tensors="pt").to(device)

    with torch.no_grad():
        emb = model.get_image_features(**inputs)

    # Normalize embeddings
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy()


def hybrid_search(query_emb, query_text, metadata, index, id2fn, conn, top_k=20, filters=None, color_query=None, weights=None):
    if weights is None:
        weights = {"semantic": 0.7, "text": 0.2, "color": 0.1}
    if query_emb is not None:
        # Get initial candidates from FAISS
        D, I = index.search(query_emb.astype("float32"), k=top_k * 3)
        candidates = [(id2fn[idx], D[0][j]) for j, idx in enumerate(I[0])]
    else:
        # For browse all, all images
        candidates = [(fn, 0) for fn in metadata.keys()]

    # Apply filters
    if filters:
        filtered = []
        for fn, score in candidates:
            meta = metadata.get(fn, {})
            if filters.get("format") and meta.get("format", "").upper() not in filters["format"]:
                continue
            if filters.get("min_width") and meta.get("width", 0) < filters["min_width"]:
                continue
            if filters.get("min_height") and meta.get("height", 0) < filters["min_height"]:
                continue
            if filters.get("tags") and not any(tag.lower() in [t.lower() for t in meta.get("tags", [])] for tag in filters["tags"]):
                continue
            if filters.get("clusters") and meta.get("cluster_id") not in filters["clusters"]:
                continue
            filtered.append((fn, score))
        candidates = filtered
    candidates = candidates[:top_k * 2]  # Keep more for re-ranking

    # Filter with FTS if query_text
    if query_text:
        cur = conn.cursor()
        filtered = []
        for fn, score in candidates:
            match = cur.execute("SELECT rowid FROM captions_fts WHERE filename=? AND captions_fts MATCH ?", (fn, query_text)).fetchone()
            if match:
                filtered.append((fn, score))
        candidates = filtered[:top_k * 2]

    # Re-rank with improved caption similarity (fuzzy matching and better scoring)
    if query_text:
        query_words = set(query_text.lower().split())
        reranked = []
        for fn, score in candidates:
            meta = metadata.get(fn, {})
            caption = meta.get("caption", "").lower()
            tags = [t.lower() for t in meta.get("tags", [])]

            # Exact keyword matches
            exact_matches = len(query_words.intersection(set(caption.split()) | set(tags)))

            # Partial/fuzzy matches (word contains query word or vice versa)
            fuzzy_matches = 0
            for query_word in query_words:
                for word in caption.split() + tags:
                    word = word.lower()
                    if query_word in word or word in query_word:
                        fuzzy_matches += 0.5  # Partial match worth half

            # Phrase matching for multi-word queries
            phrase_score = 0
            if len(query_words) > 1:
                query_phrase = query_text.lower()
                if query_phrase in caption:
                    phrase_score = len(query_words) * 0.3  # Bonus for phrase matches

            text_match = exact_matches + fuzzy_matches + phrase_score
            combined_score = score + text_match * 0.15  # Increased boost for better text matching
            reranked.append((fn, combined_score))
        candidates = sorted(reranked, key=lambda x: x[1], reverse=True)

    # Compute weighted final scores
    final_candidates = []
    for fn, score in candidates:
        semantic_score = score  # FAISS similarity score
        text_score = 0
        color_score = 0

        if query_text:
            meta = metadata[fn]
            caption = meta.get("caption", "").lower()
            tags = [t.lower() for t in meta.get("tags", [])]
            query_words = set(query_text.lower().split())
            text_match = len(query_words.intersection(set(caption.split()) | set(tags)))
            text_score = min(1.0, text_match / 5.0)  # Normalize to 0-1

        if color_query:
            q_r = int(color_query[1:3], 16)
            q_g = int(color_query[3:5], 16)
            q_b = int(color_query[5:7], 16)
            colors = metadata[fn]['colors']
            if colors:
                min_dist = min((r - q_r)**2 + (g - q_g)**2 + (b - q_b)**2 for r,g,b in colors)
                color_score = 1 / (1 + min_dist / 10000)  # 0-1 score

        final_score = (weights["semantic"] * semantic_score +
                      weights["text"] * text_score +
                      weights["color"] * color_score)
        final_candidates.append((fn, final_score))

    final_candidates = sorted(final_candidates, key=lambda x: x[1], reverse=True)[:top_k]
    return [fn for fn, _ in final_candidates]


def show_results(filenames, metadata, top_k=20):
    cols = 4
    for i, fn in enumerate(filenames[:top_k]):
        if i % cols == 0:
            cols_row = st.columns(cols)
        meta = metadata.get(fn, {})
        path = meta.get("path")
        try:
            img = Image.open(path)
            cols_row[i % cols].image(img, caption=fn, width='stretch')
            if meta.get("caption"):
                cols_row[i % cols].markdown(f"📝 {meta['caption']}")
        except Exception as e:
            cols_row[i % cols].write(f"Error: {e}")


st.set_page_config(page_title="🔎 Local Image Search", layout="wide")
st.title("🔎 Local Semantic Image Search (CLIP + FAISS)")

model, processor, device = load_clip()
index, id2fn = load_index()
metadata = load_metadata()
clusters = load_clusters()
conn = sqlite3.connect(DB_PATH)

# Show model information
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** {CLIP_MODEL_NAME.split('/')[-1]}")
st.sidebar.markdown(f"**Device:** {device}")
if hasattr(model, 'config') and hasattr(model.config, 'vision_config'):
    vision_config = model.config.vision_config
    st.sidebar.markdown(f"**Vision:** {getattr(vision_config, 'image_size', 'N/A')}px")
st.sidebar.markdown(f"**Images:** {len(metadata) if metadata else 0}")

# Sidebar filters
st.sidebar.header("Filters")
filter_format = st.sidebar.multiselect("Image Format", options=["JPEG", "PNG", "BMP", "WEBP", "TIFF"], default=[])
filter_min_width = st.sidebar.number_input("Min Width", min_value=0, value=0)
filter_min_height = st.sidebar.number_input("Min Height", min_value=0, value=0)
tags_input = st.sidebar.text_input("Tags (comma separated)", "")
filter_tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []
cluster_options = [f"{cid}: {label}" for cid, label in clusters.items()]
filter_cluster = st.sidebar.multiselect("Clusters", options=cluster_options, default=[])
filter_clusters = [int(c.split(":")[0]) for c in filter_cluster] if filter_cluster else []

enable_color = st.sidebar.checkbox("Enable color search")
color_query = st.sidebar.color_picker("Select color") if enable_color else None

# Hybrid search weights
st.sidebar.header("Search Weights")
semantic_weight = st.sidebar.slider("Semantic Similarity", 0.0, 1.0, 0.7, 0.1)
text_weight = st.sidebar.slider("Text Match", 0.0, 1.0, 0.2, 0.1)
color_weight = st.sidebar.slider("Color Match", 0.0, 1.0, 0.1, 0.1)

weights = {
    "semantic": semantic_weight,
    "text": text_weight,
    "color": color_weight,
}

filters = {
    "format": filter_format,
    "min_width": filter_min_width if filter_min_width > 0 else None,
    "min_height": filter_min_height if filter_min_height > 0 else None,
    "tags": filter_tags,
    "clusters": filter_clusters,
}

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Search images (natural language):", "",
                         help="Try: 'sunset beach', 'mountain landscape', 'red car', or upload an image for reverse search")

    # Search suggestions based on common tags and captions
    if metadata and not query.strip():
        all_tags = []
        all_caption_words = []
        for meta in metadata.values():
            all_tags.extend(meta.get("tags", []))
            caption = meta.get("caption", "")
            all_caption_words.extend(caption.split())

        # Get most common words/tags for suggestions
        common_tags = [tag for tag, _ in Counter(all_tags).most_common(5)]
        common_words = [word for word, _ in Counter(all_caption_words).most_common(5) if len(word) > 3]

        suggestions = list(set(common_tags + common_words))[:8]
        if suggestions:
            st.markdown("**Popular search terms:** " + " • ".join(f"`{s}`" for s in suggestions))

with col2:
    num_results = st.slider("Results", 4, MAX_RESULTS, DEFAULT_RESULTS)

# Image upload for reverse search
uploaded_file = st.file_uploader("Or upload an image for reverse search", type=["jpg", "jpeg", "png", "bmp", "webp"])

# Quick search buttons for common queries
if metadata:
    st.markdown("**Quick searches:**")
    quick_searches = ["nature", "people", "buildings", "animals", "food", "vehicles"]
    cols = st.columns(len(quick_searches))
    for i, term in enumerate(quick_searches):
        if cols[i].button(f"🔍 {term}", key=f"quick_{term}"):
            query = term
            st.rerun()

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)
with col3:
    if st.button("📊 Stats", use_container_width=True):
        total_images = len(metadata)
        total_size = sum(meta.get("file_size", 0) for meta in metadata.values()) / (1024*1024)  # MB
        formats = Counter(meta.get("format", "Unknown") for meta in metadata.values())
        st.info(f"📸 {total_images} images • 💾 {total_size:.1f} MB • {dict(formats.most_common(3))}")

if clear_button:
    st.rerun()

if search_button:
    if index is None:
        st.warning("No FAISS index found — run build_index.py first.")
    elif query.strip() or uploaded_file:
        with st.spinner("Searching..."):
            if uploaded_file:
                img = Image.open(uploaded_file).convert("RGB")
                q_emb = embed_image(img, model, processor, device)
                query_text = ""  # No text for image search
                color_q = None
                st.success(f"🔍 Reverse search completed for uploaded image")
            else:
                q_emb = embed_text(query, model, processor, device)
                query_text = query
                color_q = color_query
                st.success(f"🔍 Found results for: '{query}'")

        filenames = hybrid_search(q_emb, query_text, metadata, index, id2fn, conn, top_k=num_results, filters=filters, color_query=color_q, weights=weights)
        if filenames:
            st.markdown(f"**Showing {len(filenames)} results:**")
            show_results(filenames, metadata, top_k=num_results)
        else:
            st.warning("No images found matching your search criteria.")
    else:
        st.warning("Enter a query or upload an image.")

st.markdown("---")
st.subheader("📂 Browse all images")
if st.button("Show all thumbnails"):
    all_filenames = hybrid_search(None, "", metadata, index, id2fn, conn, top_k=48, filters=filters, color_query=color_query, weights=weights)
    show_results(all_filenames, metadata, top_k=48)
