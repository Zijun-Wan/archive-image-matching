# main.py
# deps: fastapi, uvicorn, pillow, python-multipart
# run: uvicorn main:app --reload
import gzip
import io
import tempfile
import uvicorn
from typing import List, Any, Dict, Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, Extra, ValidationError
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import json
import numpy as np
import cv2
from collections import defaultdict
from dataclasses import dataclass, field
import pickle
import os
from pathlib import Path
from tqdm import tqdm
from import_db import load_vt_model, patch_spatial_verify_for_tuples
# --- Optional FAISS for faster/more robust k-means ---
try:
    import faiss  # pip install faiss-cpu  (or faiss-gpu)
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False


app = FastAPI(title="Image+Object API", version="1.0")

# Allow browser clients (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Feature Extraction ----------
def extract_sift(gray_img, nfeatures=500):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, desc = sift.detectAndCompute(gray_img, None)
    if desc is None:
        return np.empty((0,128), np.float32), []
    return desc.astype(np.float32), kps
def to_rootsift(desc, eps=1e-12, l2_after=True):
    """
    Convert SIFT -> RootSIFT (Arandjelović & Zisserman, 2012).
    Steps: L1-normalize, then sqrt. Optionally L2-normalize after sqrt.
    desc: (N,128) float32
    """
    if desc is None or len(desc) == 0:
        return np.empty((0,128), np.float32)
    # L1-normalize
    desc /= (np.sum(desc, axis=1, keepdims=True) + eps)
    # element-wise sqrt
    desc = np.sqrt(desc, dtype=np.float32)
    if l2_after:
        # optional: stabilize numerics
        norms = np.linalg.norm(desc, axis=1, keepdims=True) + eps
        desc /= norms
    return desc.astype(np.float32)


def query_db(image, topk=200):
   # Query
    q_descs, q_kps = extract_sift(cv2.imread('vase.png', cv2.IMREAD_GRAYSCALE),nfeatures=1500)
    qdesc_root = to_rootsift(q_descs)
    cands = db.query(q_descs, topk=200)
    print(cands)
    reranked = db.spatial_verify(qdesc_root,q_kps, cands, ratio_thresh=0.75, cap=1000, lam=0.01)
    top10 = [(img, score) for img, score, inl in reranked[:10]]
    # print("Top 10 after RANSAC re-ranking:", top10)
    sv_map = {img:(final,inl) for img,final,inl in reranked}
    changes = []
    for img, base in cands:
        final, inl = sv_map[img]
        changes.append((img, base, inl, final-base))
    # print("any inliers >", any(inl>0 for _,_,inl,_ in changes))
    # print("top deltas:", sorted(changes, key=lambda x: x[3], reverse=True)[:5])
    print('top10',top10)
    return top10

def get_record_ids_by_page_ids(dataset, page_ids, unique=True):
    """
    Given:
      - dataset: a list of objects, each possibly with an "images" list like in your example
      - page_ids: list of pageId strings/ints to look up
    Returns:
      - a list of recordId strings associated with those pageIds.
        By default `unique=True` removes duplicates while preserving order.

    Example dataset shape:
      [
        {
          "images": [
            {
              "recordId": "27166539",
              "pageIds": ["27166540", "27166541", ...],
              ...
            },
            ...
          ]
        },
        ...
      ]
    """
    # 1) Build a mapping: pageId -> recordId
    page_to_record = {}
    for obj in dataset:
        for img in obj.get("images", []):
            rec = img.get("recordId")
            for pid in img.get("pageIds", []):
                if pid is None:
                    continue
                page_to_record[str(pid)] = rec

    # 2) Gather recordIds for the requested page_ids
    out = []
    for pid in page_ids:
        rid = page_to_record.get(str(pid))
        if rid is not None:
            out.append(rid)

    # 3) Optionally deduplicate while preserving order
    if unique:
        seen = set()
        out = [r for r in out if not (r in seen or seen.add(r))]

    return out


# ---------- Optional: faster for repeated queries ----------
def build_page_to_record_index(dataset):
    """
    Precompute and return a dict {pageId(str) -> recordId(str)}.
    Useful if you’ll call lookups many times.
    """
    index = {}
    for obj in dataset:
        for img in obj.get("images", []):
            rec = img.get("recordId")
            for pid in img.get("pageIds", []):
                if pid is None:
                    continue
                index[str(pid)] = rec
    return index

def lookup_record_ids(index, page_ids, unique=True):
    """Use the prebuilt index from build_page_to_record_index()."""
    out = [index.get(str(pid)) for pid in page_ids if index.get(str(pid)) is not None]
    if unique:
        seen = set()
        out = [r for r in out if not (r in seen or seen.add(r))]
    return out


# ---------- Example usage ----------
# pages = ["27166540", "27166551", "27166560"]
# record_ids = get_record_ids_by_page_ids(dataset, pages)          # one-off
# idx = build_page_to_record_index(dataset); record_ids2 = lookup_record_ids(idx, pages)  # repeated
# print(record_ids)


@app.post("/process")
async def process_image_and_meta(
    image: UploadFile = File(..., description="An image file"),
    dataset: List[str] = Form(..., description="List of meta values"),
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)

    img=cv2.imread(nparr, cv2.IMREAD_GRAYSCALE)
    page_ids=query_db(img, topk=20)
    record_ids=get_record_ids_by_page_ids(dataset=dataset,page_ids=page_ids, unique=True)
    """
    Accepts multipart/form-data with:
      - image: file
      - meta: JSON string matching the Meta model (can include extra fields)
    Returns parsed meta, image info, and a tiny derived result.
    """
    # # --- Read image and get basic info ---
    # try:
    #     raw = await image.read()
    #     img = Image.open(BytesIO(raw))
    #     img.load()  # ensure it's fully loaded
    #     width, height = img.size
    #     mode = img.mode
    #     fmt = img.format
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # img_cv=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
    # topk = query_db(img_cv, topk=20)
    

    # return {
    #     topk: topk,
    # }
    return record_ids,page_ids


# Optional: simple health check on root
@app.get("/")
def root():
    return {"status": "ok", "POST /process": "multipart form with fields: image (file), meta (JSON string)"}

def test():
    img=cv2.imread('vase.png', cv2.IMREAD_GRAYSCALE)
    results=query_db(img, topk=20)
    print(results)


if __name__ == "__main__":
    db=load_vt_model('vocab_db')
    db=patch_spatial_verify_for_tuples(db)
    test()
    
    uvicorn.run("app:app", host="localhost", port=8001, reload=True)
