# 🔎 Local Image Search App

A powerful semantic image search application using CLIP embeddings, FAISS indexing, and Streamlit UI. Search your local image collection with natural language queries, filters, and reverse image search.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Prathmesh Magar

### Features

- **Hybrid semantic search** using CLIP embeddings + text re-ranking + FTS5 full-text search
- **Reverse image search** by uploading query images
- **Color-based search** with color picker for finding similar colors
- **Semantic clustering** of images with LLM-labeled clusters
- **Timeline view** sorted by EXIF date
- **Hybrid search modes** with adjustable weights for semantic, text, and color similarity
- **Advanced filters**: by format, size, tags, clusters
- Fast similarity search with FAISS (Flat or IVF index)
- Streamlit web UI with pagination
- Structured JSON LLM captions with tags
- Incremental processing (skips existing images)
- Configurable via `config.yaml`
- Supports multiple image formats
- Offline operation after initial setup

### Steps

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure (optional)**

   Edit `config.yaml` for custom settings (batch sizes, models, etc.).

3. **Put images**

   ```bash
   mkdir images
   # drop your .jpg, .png, etc. into ./images
   ```

4. **Process images**

   ```bash
   python process_images.py
   ```

5. **Build FAISS index**

   ```bash
   python build_index.py
   ```

6. **Run Streamlit UI**

   ```bash
   streamlit run search_app.py
   ```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

### Configuration

Key settings in `config.yaml`:

- `use_llm`: Enable LLM captioning (requires Ollama)
- `llm_model`: Vision model (e.g., "moondream", "llava")
- `faiss_index_type`: "flat" or "ivf" for large datasets
- `thumbnail_size`: Resize images before embedding to save VRAM
- `batch_size`: CLIP batch processing size

---

### Notes

* Uses `openai/clip-vit-base-patch32` for embeddings by default.
* FAISS inner-product index → cosine similarity.
* Multiprocessing speeds up metadata extraction.
* Structured JSON prompts for clean LLM tags parsing.
* Incremental updates: re-run `process_images.py` to add new images.
* Handles `.jpg`, `.png`, `.bmp`, `.webp`, `.tif` etc.
* Works offline if models are cached (`transformers` will store them in `~/.cache/huggingface`).

---

### Large Datasets (FAISS IVF)

For large datasets, the system automatically switches to IVF indexing when >10K images are detected. IVF uses inverted files for scalable search. You can also manually set `faiss_index_type: "ivf"` in `config.yaml`.

---

### Troubleshooting

* `CUDA out of memory` → reduce `batch_size` or set `device: "cpu"`.
* No results? Ensure `build_index.py` ran successfully.
* LLM captions failing? Check Ollama is running (`ollama serve`).
