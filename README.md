# 🔎 Local Image Search App

A powerful semantic image search application using CLIP embeddings, FAISS indexing, and Streamlit UI. Search your local image collection with natural language queries.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Prathmesh Magar

### Features

- Semantic search using CLIP embeddings
- Fast similarity search with FAISS
- Streamlit web UI
- Supports multiple image formats
- Optional LLM-based captioning
- Offline operation after initial setup

### Steps

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Put images**

   ```bash
   mkdir images
   # drop your .jpg, .png, etc. into ./images
   ```

3. **Process images**

   ```bash
   python process_images.py
   ```

4. **Build FAISS index**

   ```bash
   python build_index.py
   ```

5. **Run Streamlit UI**

   ```bash
   streamlit run search_app.py
   ```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

### Notes

* Uses `openai/clip-vit-base-patch32` for embeddings.
* FAISS inner-product index → cosine similarity.
* Multiprocessing speeds up metadata extraction.
* Optional local LLM captioning (`USE_LLM = True` + `LLM_CMD` config).
* Handles `.jpg`, `.png`, `.bmp`, `.webp`, `.tif` etc.
* Works offline if models are cached (`transformers` will store them in `~/.cache/huggingface`).

---

### Optional: Large Datasets (FAISS IVF/HNSW)

Replace in `build_index.py`:

```python
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist=100, metric=faiss.METRIC_INNER_PRODUCT)
index.train(xb)
index.add(xb)
```

to use **inverted file (IVF)** for millions of vectors.

---

### Troubleshooting

* `CUDA out of memory` → use smaller batch size or CPU.
* No results? Ensure `build_index.py` ran successfully.
