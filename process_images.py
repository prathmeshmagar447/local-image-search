"""
process_images.py
Extract metadata and CLIP embeddings for all images in ./images
Creates SQLite DB (metadata.db) and stores embeddings in it.
"""

import os, json, sqlite3, subprocess
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from transformers import CLIPProcessor, CLIPModel

DB_PATH = "metadata.db"
IMAGES_DIR = "images"
BATCH_SIZE = 32
N_WORKERS = max(1, os.cpu_count() - 1)

# Optional multimodal captioning
USE_LLM = True
LLM_MODEL = "moondream"  # or "llava", "bakllava", "llava-phi", etc.


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        filename TEXT UNIQUE,
        path TEXT,
        width INTEGER,
        height INTEGER,
        format TEXT,
        exif JSON,
        dominant_colors JSON,
        llm_caption TEXT,
        tags JSON
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY,
        filename TEXT UNIQUE,
        embedding BLOB
    )''')
    conn.commit()
    return conn


def get_dominant_colors(img, n_colors=3):
    """Compute up to n_colors dominant colors, safely."""
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img).reshape(-1, 3).astype(np.float32)

    # Avoid NaN / Inf problems
    if not np.isfinite(arr).all() or len(arr) == 0:
        return [[0, 0, 0]] * n_colors

    # If image is nearly monochrome, skip clustering
    std = np.std(arr, axis=0).mean()
    if std < 1e-3:
        mean_color = np.mean(arr, axis=0).astype(int).tolist()
        return [mean_color] * n_colors

    try:
        from sklearn.cluster import KMeans
        with np.errstate(all="ignore"):
            km = KMeans(n_clusters=n_colors, n_init=3, random_state=0).fit(arr)
        centers = np.clip(km.cluster_centers_, 0, 255).astype(int).tolist()
    except Exception:
        # Fallback random sampling
        idx = np.random.choice(len(arr), size=min(n_colors, len(arr)), replace=False)
        centers = arr[idx].astype(int).tolist()

    return centers


def llm_describe_image(image_path):
    """Generate a caption for an image using Ollama vision model via REST API."""
    if not USE_LLM:
        return None

    import requests, base64

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "model": LLM_MODEL,
            "prompt": "Describe this image briefly and list 5 descriptive tags.",
            "images": [img_b64],
        }
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60, stream=True)
        if r.status_code == 200:
            full_response = ""
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            full_response += data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            return full_response.strip()
    except Exception as e:
        print(f"⚠️ Ollama API caption failed for {image_path}: {e}")
    return None


def process_single(path):
    try:
        img = Image.open(path)
        w, h = img.size
        fmt = img.format
        colors = get_dominant_colors(img)
        exif = {}
        try:
            exif = {k: str(v) for k, v in (img._getexif() or {}).items()}
        except Exception:
            pass
        caption = llm_describe_image(path)
        # Parse tags from caption
        tags = []
        if caption:
            if "Tags:" in caption or "tags:" in caption:
                parts = caption.split("Tags:")
                if len(parts) > 1:
                    tags_str = parts[1].strip()
                    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        return dict(
            filename=os.path.basename(path),
            path=path,
            width=w,
            height=h,
            format=fmt,
            exif=exif,
            dominant_colors=colors,
            llm_caption=caption,
            tags=tags,
        )
    except Exception as e:
        return {"error": str(e), "path": path}


def compute_embeddings_batch(model, processor, pil_images):
    inputs = processor(images=pil_images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        embs = model.get_image_features(**inputs)
    embs = embs / embs.norm(p=2, dim=-1, keepdim=True)
    return embs.cpu().numpy()


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    conn = init_db()
    cur = conn.cursor()

    # Get already processed filenames
    existing = set(row[0] for row in cur.execute("SELECT filename FROM images"))

    all_paths = [str(p) for p in Path(IMAGES_DIR).glob("**/*") if p.suffix.lower() in
                 (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")]
    # Filter out already processed images
    all_paths = [p for p in all_paths if os.path.basename(p) not in existing]
    print(f"Found {len(all_paths)} new images.")

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(process_single, p) for p in all_paths]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting metadata"):
            r = fut.result()
            if r and "error" not in r:
                results.append(r)

    for r in results:
        cur.execute('''INSERT OR REPLACE INTO images
            (filename, path, width, height, format, exif, dominant_colors, llm_caption, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (r["filename"], r["path"], r["width"], r["height"], r["format"],
             json.dumps(r["exif"]), json.dumps(r["dominant_colors"]),
             r.get("llm_caption"), json.dumps(r.get("tags", []))))
    conn.commit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    for i in tqdm(range(0, len(results), BATCH_SIZE), desc="Computing embeddings"):
        batch = results[i:i+BATCH_SIZE]
        pil_images = [Image.open(r["path"]).convert("RGB") for r in batch]
        embs = compute_embeddings_batch(model, processor, pil_images)
        for r, emb in zip(batch, embs):
            cur.execute("INSERT OR REPLACE INTO embeddings (filename, embedding) VALUES (?, ?)",
                        (r["filename"], emb.tobytes()))
    conn.commit()
    conn.close()
    print("✅ Done: metadata.db created with metadata + embeddings.")


if __name__ == "__main__":
    main()
