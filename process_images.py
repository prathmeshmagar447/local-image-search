"""
process_images.py
Extract metadata and CLIP embeddings for all images in ./images
Creates SQLite DB (metadata.db) and stores embeddings in it.
Refactored for structured JSON captions, incremental updates, and optimizations.
"""

import os, json, sqlite3, subprocess, time
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import yaml
from loguru import logger

import torch
from transformers import CLIPProcessor, CLIPModel



# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup logging
logger.add("image_search.log", rotation="10 MB", level="INFO")

DB_PATH = config["db_path"]
IMAGES_DIR = config["images_dir"]
BATCH_SIZE = config["batch_size"]
THUMBNAIL_SIZE = config["thumbnail_size"]

USE_LLM = config["use_llm"]
LLM_MODEL = config["llm_model"]
LLM_TIMEOUT = config["llm_timeout"]
LLM_MAX_CONCURRENT = config["llm_max_concurrent"]
N_WORKERS = min(config.get("n_workers", max(1, os.cpu_count() - 1)), LLM_MAX_CONCURRENT)

# CLIP settings
CLIP_MODEL_NAME = config["clip_model"]
DEVICE_CONFIG = config["device"]
CLIP_USE_FAST = config.get("clip_use_fast", True)
CLIP_MAX_LENGTH = config.get("clip_max_length", 77)


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        filename TEXT UNIQUE,
        path TEXT,
        file_size INTEGER,
        modified_time REAL,
        width INTEGER,
        height INTEGER,
        format TEXT,
        exif JSON,
        dominant_colors JSON,
        llm_caption TEXT
    )''')
    # Create indexes for better performance
    c.execute('''CREATE INDEX IF NOT EXISTS idx_filename ON images(filename)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_format ON images(format)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_modified_time ON images(modified_time)''')
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY,
        filename TEXT UNIQUE,
        embedding BLOB
    )''')
    # FTS5 virtual table for full-text search on captions and tags
    c.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS captions_fts USING fts5(filename, caption, tags)''')
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
    """Generate structured JSON caption for an image using Ollama vision model with retry."""
    if not USE_LLM:
        return {"caption": "", "tags": []}

    import requests, base64, time

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Resize image to reduce processing time
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((512, 512))  # Resize to max 512x512
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            payload = {
                "model": LLM_MODEL,
                "prompt": '''Analyze this image and provide a detailed description. Focus on:
- Main subjects and objects
- Actions and interactions
- Setting and environment
- Colors, lighting, and mood
- Any text visible in the image

Output JSON with:
"caption": A detailed 2-3 sentence description
"tags": 8-10 relevant keywords/tags (strings)
"objects": List of objects detected in the image
"colors": Dominant colors described in words
"mood": Overall mood or atmosphere
"confidence": Your confidence score (0-1)

Example format:
{
  "caption": "A sunny beach scene with people playing volleyball...",
  "tags": ["beach", "volleyball", "ocean", "sunny", "people"],
  "objects": ["ball", "net", "sand", "water"],
  "colors": ["blue", "yellow", "white"],
  "mood": "energetic and joyful",
  "confidence": 0.95
}''',
                "images": [img_b64],
            }
            r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=LLM_TIMEOUT, stream=True)
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
                # Parse JSON response
                try:
                    parsed = json.loads(full_response.strip())
                    # Handle both old and new JSON formats
                    caption = parsed.get("caption", "")
                    tags = parsed.get("tags", [])
                    objects = parsed.get("objects", [])
                    colors_desc = parsed.get("colors", [])
                    mood = parsed.get("mood", "")
                    confidence = parsed.get("confidence", 0.5)
                    relationships = parsed.get("relationships", [])

                    # Combine objects and colors into tags for backward compatibility
                    enhanced_tags = tags + objects + colors_desc
                    if mood:
                        enhanced_tags.append(mood)

                    return {
                        "caption": caption,
                        "tags": enhanced_tags,
                        "objects": objects,
                        "colors": colors_desc,
                        "mood": mood,
                        "confidence": confidence,
                        "relationships": relationships
                    }
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from LLM for {image_path}: {full_response}")
                    return {"caption": full_response.strip(), "tags": [], "relationships": []}
            else:
                logger.warning(f"LLM API returned status {r.status_code} for {image_path}")
        except requests.exceptions.Timeout:
            logger.warning(f"LLM timeout for {image_path}, attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"Ollama API caption failed for {image_path}: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff

    logger.error(f"Failed to get LLM caption for {image_path} after {max_retries} attempts")
    return {"caption": "", "tags": []}


def process_single(path, conn=None):
    try:
        stat = os.stat(path)
        file_size = stat.st_size
        modified_time = stat.st_mtime

        img = Image.open(path)
        w, h = img.size
        fmt = img.format
        colors = get_dominant_colors(img)
        exif = {}
        lat, long = None, None
        try:
            exif_raw = img._getexif() or {}
            exif = {k: str(v) for k, v in exif_raw.items()}
            if 34853 in exif_raw:  # GPSInfo
                gps = exif_raw[34853]
                if 1 in gps and 2 in gps and 3 in gps and 4 in gps:
                    lat_ref = gps[1]
                    lat_d, lat_m, lat_s = gps[2]
                    lat = (lat_d + lat_m/60 + lat_s/3600) * (1 if lat_ref == b'N' else -1)
                    lon_ref = gps[3]
                    lon_d, lon_m, lon_s = gps[4]
                    long = (lon_d + lon_m/60 + lon_s/3600) * (1 if lon_ref == b'E' else -1)
        except Exception:
            pass

        filename = os.path.basename(path)
        # Check if LLM caption already exists
        if conn:
            cur = conn.cursor()
            existing_caption = cur.execute("SELECT llm_caption FROM images WHERE filename=?", (filename,)).fetchone()
            if existing_caption and existing_caption[0]:
                caption_data = json.loads(existing_caption[0])
            else:
                caption_data = llm_describe_image(path)
        else:
            caption_data = llm_describe_image(path)

        scene_graph = json.dumps({"relationships": caption_data.get("relationships", [])})

        return dict(
            filename=filename,
            path=path,
            file_size=file_size,
            modified_time=modified_time,
            width=w,
            height=h,
            format=fmt,
            exif=exif,
            dominant_colors=colors,
            llm_caption=json.dumps(caption_data),
        )
    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        return {"error": str(e), "path": path}


def compute_embeddings_batch(model, processor, pil_images, device):
    """Compute embeddings with memory optimization and error handling."""
    try:
        inputs = processor(images=pil_images, return_tensors="pt").to(device)

        # Use mixed precision if available for faster processing
        with torch.no_grad():
            if device.type == "cuda" and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    embs = model.get_image_features(**inputs)
            else:
                embs = model.get_image_features(**inputs)

        # Normalize embeddings
        embs = embs / embs.norm(p=2, dim=-1, keepdim=True)
        return embs.cpu().numpy()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"OOM error in batch processing, reducing batch size...")
            # Try with smaller batches recursively
            if len(pil_images) > 1:
                mid = len(pil_images) // 2
                emb1 = compute_embeddings_batch(model, processor, pil_images[:mid], device)
                emb2 = compute_embeddings_batch(model, processor, pil_images[mid:], device)
                return np.concatenate([emb1, emb2], axis=0)
            else:
                raise e
        else:
            raise e


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    conn = init_db()
    cur = conn.cursor()

    # Get already processed filenames (those with embeddings)
    existing_with_embeddings = set(row[0] for row in cur.execute("SELECT filename FROM embeddings"))
    # Get images that have metadata but no embeddings (need embedding computation)
    existing_with_metadata = set(row[0] for row in cur.execute("SELECT filename FROM images"))

    all_paths = [str(p) for p in Path(IMAGES_DIR).glob("**/*") if p.suffix.lower() in
                 (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")]
    # Filter out already fully processed images (those with embeddings)
    all_paths = [p for p in all_paths if os.path.basename(p) not in existing_with_embeddings]
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
            (filename, path, file_size, modified_time, width, height, format, exif, dominant_colors, llm_caption)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (r["filename"], r["path"], r["file_size"], r["modified_time"], r["width"], r["height"], r["format"],
             json.dumps(r["exif"]), json.dumps(r["dominant_colors"]), r["llm_caption"]))
        # Insert into FTS
        caption_data = json.loads(r["llm_caption"])
        caption = caption_data.get("caption", "")
        tags_text = ' '.join(caption_data.get("tags", []))
        relationships_text = ' '.join(caption_data.get("relationships", []))
        cur.execute("INSERT OR REPLACE INTO captions_fts (filename, caption, tags) VALUES (?, ?, ?)",
                    (r["filename"], caption + ' ' + relationships_text, tags_text))
    conn.commit()

    # Populate FTS for existing images if not already
    existing_fts = set(row[0] for row in cur.execute("SELECT filename FROM captions_fts"))
    all_images = cur.execute("SELECT filename, llm_caption FROM images").fetchall()
    for fn, llm_cap in all_images:
        if fn not in existing_fts and llm_cap:
            cap_data = json.loads(llm_cap)
            cap = cap_data.get("caption", "")
            tags = ' '.join(cap_data.get("tags", []))
            relationships = ' '.join(cap_data.get("relationships", []))
            cur.execute("INSERT OR REPLACE INTO captions_fts (filename, caption, tags) VALUES (?, ?, ?)", (fn, cap + ' ' + relationships, tags))
    conn.commit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME} on device: {device}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=CLIP_USE_FAST)

    for i in tqdm(range(0, len(results), BATCH_SIZE), desc="Computing embeddings"):
        batch = results[i:i+BATCH_SIZE]
        pil_images = []
        for r in batch:
            img = Image.open(r["path"]).convert("RGB")
            # Thumbnail to reduce memory usage
            img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            pil_images.append(img)
        embs = compute_embeddings_batch(model, processor, pil_images, device)
        for r, emb in zip(batch, embs):
            cur.execute("INSERT OR REPLACE INTO embeddings (filename, embedding) VALUES (?, ?)",
                        (r["filename"], emb.tobytes()))
    conn.commit()
    conn.close()
    print("✅ Done: metadata.db created with metadata + embeddings.")


if __name__ == "__main__":
    main()
