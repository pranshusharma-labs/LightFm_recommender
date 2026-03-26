"""
api.py
──────
FastAPI serving LightFM hybrid recommendations.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /          → health check
    GET  /items     → list catalog items
    POST /recommend → get recommendations
    GET  /eval      → offline eval results
"""

import os
import pickle
import numpy as np
import scipy.sparse as sp
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

MODEL_DIR = os.environ.get('MODEL_DIR', 'model/')
state     = {}


# ── Load model on startup ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading LightFM model...")

    with open(f'{MODEL_DIR}/lightfm_collab.pkl', 'rb') as f:
        model_collab = pickle.load(f)

    with open(f'{MODEL_DIR}/lightfm_content.pkl', 'rb') as f:
        model_content = pickle.load(f)

    with open(f'{MODEL_DIR}/lightfm_hybrid.pkl', 'rb') as f:
        model_hybrid = pickle.load(f)

    with open(f'{MODEL_DIR}/artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)

    hybrid_features_path = f"{MODEL_DIR}/item_features_hybrid.npz"
    content_features_path = f"{MODEL_DIR}/item_features_content.npz"
    fallback_features_path = f"{MODEL_DIR}/item_features.npz"

    hybrid_item_features = sp.load_npz(
        hybrid_features_path if os.path.exists(hybrid_features_path) else fallback_features_path
    )
    content_item_features = sp.load_npz(
        content_features_path if os.path.exists(content_features_path) else fallback_features_path
    )

    state['models'] = {
        'collab': model_collab,
        'content': model_content,
        'hybrid': model_hybrid,
    }
    state['artifacts'] = artifacts
    state['item_features'] = {
        'content': content_item_features,
        'hybrid': hybrid_item_features,
    }

    a = artifacts
    print(f"Ready ✅ | Users: {len(a['user_id_map']):,} | Items: {len(a['item_id_map']):,}")
    yield
    state.clear()


app = FastAPI(
    title       = "LightFM Hybrid Recommender API",
    description = "Pure Collaborative / Content-Based / Hybrid via LightFM",
    version     = "1.0.0",
    lifespan    = lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    user_id:  str
    top_k:    int   = 10
    mode:     str   = "hybrid"   # "hybrid" | "collab" | "content"


class RecommendResponse(BaseModel):
    user_id:         str
    mode:            str
    recommendations: List[dict]
    total_items:     int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    a = state.get('artifacts', {})
    return {
        "status": "ok",
        "model":  "LightFM Hybrid Recommender",
        "users":  len(a.get('user_id_map', {})),
        "items":  len(a.get('item_id_map', {})),
        "docs":   "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": sorted(list(state.get('models', {}).keys()))
    }


@app.get("/items")
def list_items(limit: int = 50):
    a = state['artifacts']
    return {
        "total": len(a['item_id_map']),
        "items": [
            {"item_key": k, "title": a['item_to_title'].get(k, k)}
            for k in list(a['item_id_map'].keys())[:limit]
        ]
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if req.top_k < 1 or req.top_k > 50:
        raise HTTPException(400, "top_k must be between 1 and 50")
    if req.mode not in ("hybrid", "collab", "content"):
        raise HTTPException(400, "mode must be 'hybrid', 'collab', or 'content'")

    a             = state['artifacts']
    model         = state['models'][req.mode]
    mode_item_features = state['item_features']
    user_id_map   = a['user_id_map']
    item_id_map   = a['item_id_map']
    inv_item_map  = a['inv_item_map']
    all_item_idxs = a['all_item_idxs']
    item_to_title = a['item_to_title']
    user_seen_items = a.get('user_seen_items', {})
    popular_items = a.get('popular_items', [])

    if req.user_id not in user_id_map:
        recs = [
            {
                "rank": int(i + 1),
                "item_key": item['item_key'],
                "title": item['title'],
                "score": item['score'],
            }
            for i, item in enumerate(popular_items[:req.top_k])
        ]
        return RecommendResponse(
            user_id=req.user_id,
            mode=req.mode,
            recommendations=recs,
            total_items=len(item_id_map)
        )

    user_idx = user_id_map[req.user_id]
    seen_items = set(user_seen_items.get(req.user_id, []))

    if req.mode == "content":
        features_arg = mode_item_features['content']
    elif req.mode == "hybrid":
        features_arg = mode_item_features['hybrid']
    else:
        features_arg = None

    scores  = model.predict(user_idx, all_item_idxs, item_features=features_arg)
    ranked  = np.argsort(-scores)

    recs = []
    for idx in ranked:
        item_key = inv_item_map[idx]
        if item_key in seen_items:
            continue
        recs.append(
            {
                "rank": int(len(recs) + 1),
                "item_key": item_key,
                "title": item_to_title.get(item_key, ''),
                "score": round(float(scores[idx]), 4),
            }
        )
        if len(recs) == req.top_k:
            break

    return RecommendResponse(
        user_id         = req.user_id,
        mode            = req.mode,
        recommendations = recs,
        total_items     = len(item_id_map)
    )


@app.get("/eval")
def get_eval():
    a = state.get('artifacts', {})
    return a.get('eval_results', {"message": "No eval results — run train.py first"})
