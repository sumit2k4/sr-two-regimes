"""Download UC Merced and EuroSAT-RGB from the HuggingFace parquet mirrors and
write them out as class-foldered PNGs under Paper3/data/."""
import io, os, sys, urllib.request, ssl
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CTX = ssl.create_default_context(); CTX.check_hostname=False; CTX.verify_mode=ssl.CERT_NONE

SPECS = {
    "ucmerced": ["https://huggingface.co/datasets/blanchon/UC_Merced/resolve/main/data/train-00000-of-00001.parquet"],
    "eurosat":  ["https://huggingface.co/datasets/blanchon/EuroSAT_RGB/resolve/main/data/train-00000-of-00001.parquet",
                 "https://huggingface.co/datasets/blanchon/EuroSAT_RGB/resolve/main/data/validation-00000-of-00001.parquet",
                 "https://huggingface.co/datasets/blanchon/EuroSAT_RGB/resolve/main/data/test-00000-of-00001.parquet"],
}

def grab(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        print("  cached", os.path.basename(dst)); return
    print("  GET", url)
    with urllib.request.urlopen(url, timeout=300, context=CTX) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    print("   ->", round(os.path.getsize(dst)/1e6, 1), "MB")

for name, urls in SPECS.items():
    raw = os.path.join(ROOT, "_parquet"); os.makedirs(raw, exist_ok=True)
    out = os.path.join(ROOT, name)
    files = []
    for i, u in enumerate(urls):
        dst = os.path.join(raw, f"{name}_{i}.parquet"); grab(u, dst); files.append(dst)
    if os.path.exists(os.path.join(out, "_DONE")):
        print(name, "already extracted"); continue
    n = 0
    for f in files:
        t = pq.read_table(f)
        cols = t.column_names
        print(name, "cols:", cols, "rows:", t.num_rows)
        lab_col = "label" if "label" in cols else cols[-1]
        img_col = "image" if "image" in cols else cols[0]
        # class names from parquet metadata if present
        names = None
        try:
            import json
            meta = t.schema.metadata or {}
            hf = json.loads(meta[b"huggingface"].decode()) if b"huggingface" in meta else {}
            names = hf["info"]["features"][lab_col]["names"]
        except Exception:
            pass
        imgs = t.column(img_col).to_pylist(); labs = t.column(lab_col).to_pylist()
        for im, lb in zip(imgs, labs):
            cls = names[lb] if names else str(lb)
            d = os.path.join(out, cls); os.makedirs(d, exist_ok=True)
            b = im["bytes"] if isinstance(im, dict) else im
            Image.open(io.BytesIO(b)).convert("RGB").save(os.path.join(d, f"{n:06d}.png"))
            n += 1
    open(os.path.join(out, "_DONE"), "w").write(str(n))
    print(name, "wrote", n, "images")
