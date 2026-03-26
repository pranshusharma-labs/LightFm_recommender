"""
train.py
────────
Trains LightFM in three modes and evaluates each:
  1. Pure Collaborative  — ID embeddings only
  2. Pure Content-Based  — item feature embeddings only
  3. Hybrid              — both combined
  4. Random Baseline     — lower bound

Usage:
    python train.py

Outputs:
  - model/lightfm_hybrid.pkl    ← best model (hybrid)
  - model/dataset.pkl           ← LightFM dataset mapping
  - model/artifacts.pkl         ← item maps, eval results
"""

import os
import pickle
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from tqdm import tqdm

from data_loader import (
    load_products, load_events, resolve_item_ids,
    apply_weights_and_decay, split_data, build_item_text
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PRODUCTS_PATH = 'data/products.csv'
EVENTS_PATH   = 'data/events.csv'
MODEL_DIR     = 'model/'
EPOCHS        = 20
NUM_THREADS   = 4
NUM_COMPONENTS = 64
K_VALUES      = [1, 5, 10]
SEED          = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Step 1: Loading data ─────────────────────────────────────")
products = load_products(PRODUCTS_PATH)
events   = load_events(EVENTS_PATH)
events   = resolve_item_ids(events, products)
events   = apply_weights_and_decay(events)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SPLIT (airtight temporal + session-boundary split)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Step 2: Splitting data ───────────────────────────────────")
train_df, test_df = split_data(events)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BUILD LIGHTFM DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Step 3: Building LightFM dataset ────────────────────────")

# Build richer item features — use category hierarchy, brand, tags
# Each product gets a list of feature strings
def get_item_features(row):
    features = []
    if row['category'] and row['category'] != 'unknown':
        # Add each level of category hierarchy as a separate feature
        cat = row['category'].replace('Home/', '')
        for part in cat.split('/'):
            p = part.strip()
            if p:
                features.append(f"cat:{p}")
    if row['brand'] and row['brand'] not in ['', 'nan', 'Google']:
        features.append(f"brand:{row['brand']}")
    if row['tags'] and row['tags'] not in ['', 'nan', '(not set)']:
        for tag in row['tags'].split(','):
            t = tag.strip()
            if t:
                features.append(f"tag:{t}")
    # Price bucket — gives model sense of price range
    if row['price'] > 0:
        if row['price'] < 15:
            features.append("price:low")
        elif row['price'] < 40:
            features.append("price:mid")
        else:
            features.append("price:high")
    return features if features else ['cat:unknown']

products['feature_list'] = products.apply(get_item_features, axis=1)

# Collect all unique feature strings
all_features = sorted(set(
    feat for feats in products['feature_list'] for feat in feats
))

user_ids = train_df['user_id'].drop_duplicates().tolist()
item_ids = products['product_id'].drop_duplicates().tolist()

# Hybrid dataset keeps LightFM's default item identity features.
dataset = Dataset()
dataset.fit(
    users         = user_ids,
    items         = item_ids,
    item_features = all_features
)

# Build interaction matrix from train only — no test leakage
interactions, weights = dataset.build_interactions(
    (row['user_id'], row['item_id'], row['weight'])
    for _, row in train_df.iterrows()
    if row['item_id'] in set(products['product_id'])
)

# Build hybrid item feature matrix (identity + metadata features)
hybrid_item_features = dataset.build_item_features(
    (row['product_id'], row['feature_list'])
    for _, row in products.iterrows()
)

user_id_map, _, item_id_map, _ = dataset.mapping()
inv_item_map  = {v: k for k, v in item_id_map.items()}
all_item_idxs = np.arange(len(item_id_map))

# Content dataset disables item identity features so recommendations are
# driven only by shared metadata features.
content_dataset = Dataset(item_identity_features=False)
content_dataset.fit(
    users         = user_ids,
    items         = item_ids,
    item_features = all_features
)
content_user_id_map, _, content_item_id_map, _ = content_dataset.mapping()
if user_id_map != content_user_id_map or item_id_map != content_item_id_map:
    raise ValueError("Content dataset mappings diverged from hybrid dataset mappings.")

content_item_features = content_dataset.build_item_features(
    (row['product_id'], row['feature_list'])
    for _, row in products.iterrows()
)

# Item metadata for API
item_to_title = dict(zip(products['product_id'], products['title']))
item_to_price = dict(zip(products['product_id'], products['price']))
item_to_text  = dict(zip(
    products['product_id'],
    products.apply(build_item_text, axis=1)
))

# User history and popularity metadata for serving
catalog_item_ids = set(item_id_map.keys())
user_seen_items = (
    train_df[train_df['item_id'].isin(catalog_item_ids)]
    .groupby('user_id')['item_id']
    .agg(lambda items: sorted(set(items)))
    .to_dict()
)
popular_items = (
    train_df[train_df['item_id'].isin(catalog_item_ids)]
    .groupby('item_id', as_index=False)['weight']
    .sum()
    .sort_values(['weight', 'item_id'], ascending=[False, True])
)
popular_items = [
    {
        'item_key': row['item_id'],
        'title': item_to_title.get(row['item_id'], row['item_id']),
        'score': round(float(row['weight']), 4),
    }
    for _, row in popular_items.iterrows()
]
popular_item_ids = [item['item_key'] for item in popular_items]
user_seen_item_idxs = {
    user: np.array(
        [item_id_map[item_id] for item_id in item_ids if item_id in item_id_map],
        dtype=np.int32
    )
    for user, item_ids in user_seen_items.items()
}

print(f"Users in dataset  : {len(user_id_map):,}")
print(f"Items in dataset  : {len(item_id_map):,}")
print(f"Item features     : {len(all_features)}")
print(f"Hybrid feature dim: {hybrid_item_features.shape[1]:,}")
print(f"Content feature dim: {content_item_features.shape[1]:,}")
print(f"Interaction nnz   : {interactions.nnz:,}")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def summarize_results(results, label, skipped=0):
    """Print and round evaluation metrics."""
    print(f"\n{'='*78}")
    print(f"  {label}")
    print(f"{'='*78}")
    print(f"  {'K':<5} {'HitRate@K':<12} {'Recall@K':<12} {'Precision@K':<14} {'NDCG@K'}")
    print(f"  {'-'*66}")

    summary = {}
    for k in K_VALUES:
        h = np.mean(results[k]['hit_rate'])
        r = np.mean(results[k]['recall'])
        p = np.mean(results[k]['precision'])
        n = np.mean(results[k]['ndcg'])
        summary[k] = {
            'hit_rate': round(h, 4),
            'recall': round(r, 4),
            'precision': round(p, 4),
            'ndcg': round(n, 4),
        }
        print(f"  K={k:<3} {h:<12.4f} {r:<12.4f} {p:<14.4f} {n:.4f}")

    print(f"{'='*78}")
    if skipped:
        print(f"  Skipped {skipped} cold-start users")
    return summary


def evaluate_model(model, test_df, item_features_arg, label):
    """Compute hit rate, true recall, precision, and NDCG with seen-item masking."""
    results = {
        k: {'hit_rate': [], 'recall': [], 'precision': [], 'ndcg': []}
        for k in K_VALUES
    }
    skipped = 0

    for user, group in tqdm(test_df.groupby('user_id'), desc=f"Eval [{label}]"):
        if user not in user_id_map:
            skipped += 1
            continue

        user_idx   = user_id_map[user]
        true_items = set(group['item_id'].tolist())
        seen_idxs  = user_seen_item_idxs.get(user, np.array([], dtype=np.int32))

        scores = model.predict(
            user_idx,
            all_item_idxs,
            item_features=item_features_arg
        )
        masked_scores = scores.copy()
        if seen_idxs.size:
            masked_scores[seen_idxs] = -np.inf
        ranked = np.argsort(-masked_scores)

        for k in K_VALUES:
            top_k  = [inv_item_map[i] for i in ranked[:k]]
            hits   = [1 if item in true_items else 0 for item in top_k]
            n_hits = sum(hits)

            results[k]['hit_rate'].append(1 if n_hits > 0 else 0)
            results[k]['recall'].append(n_hits / len(true_items) if true_items else 0)
            results[k]['precision'].append(n_hits / k)

            dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            results[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)

    return summarize_results(results, label, skipped=skipped)


def evaluate_random(test_df):
    """Random baseline with seen-item masking."""
    results = {
        k: {'hit_rate': [], 'recall': [], 'precision': [], 'ndcg': []}
        for k in K_VALUES
    }
    rng     = np.random.default_rng(SEED)
    n_items = len(item_id_map)

    for user, group in test_df.groupby('user_id'):
        if user not in user_id_map:
            continue
        true_items = set(group['item_id'].tolist())
        seen_items = set(user_seen_items.get(user, []))
        ranked     = [
            inv_item_map[i]
            for i in rng.permutation(n_items)
            if inv_item_map[i] not in seen_items
        ]

        for k in K_VALUES:
            top_k  = ranked[:k]
            hits   = [1 if item in true_items else 0 for item in top_k]
            n_hits = sum(hits)
            results[k]['hit_rate'].append(1 if n_hits > 0 else 0)
            results[k]['recall'].append(n_hits / len(true_items) if true_items else 0)
            results[k]['precision'].append(n_hits / k)
            dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            results[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)

    return summarize_results(results, "Random Baseline")


def evaluate_popularity(test_df):
    """Popularity baseline with seen-item masking."""
    results = {
        k: {'hit_rate': [], 'recall': [], 'precision': [], 'ndcg': []}
        for k in K_VALUES
    }

    for user, group in test_df.groupby('user_id'):
        if user not in user_id_map:
            continue
        true_items = set(group['item_id'].tolist())
        seen_items = set(user_seen_items.get(user, []))
        ranked     = [item_id for item_id in popular_item_ids if item_id not in seen_items]

        for k in K_VALUES:
            top_k  = ranked[:k]
            hits   = [1 if item in true_items else 0 for item in top_k]
            n_hits = sum(hits)
            results[k]['hit_rate'].append(1 if n_hits > 0 else 0)
            results[k]['recall'].append(n_hits / len(true_items) if true_items else 0)
            results[k]['precision'].append(n_hits / k)
            dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            results[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)

    return summarize_results(results, "Popularity Baseline")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN ALL THREE MODES
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Pure Collaborative ──────────────────────────────────────────────────────
# Only user/item ID embeddings — no content features
print("\n\n── Training: Pure Collaborative ──────────────────────────")
model_collab = LightFM(
    loss          = 'warp',
    no_components = NUM_COMPONENTS,
    random_state  = SEED
)
model_collab.fit(
    interactions,
    sample_weight = weights,
    epochs        = EPOCHS,
    num_threads   = NUM_THREADS,
    verbose       = True
)
results_collab = evaluate_model(
    model_collab, test_df,
    item_features_arg = None,
    label = "Pure Collaborative (ID embeddings only)"
)

# ── 2. Pure Content-Based ──────────────────────────────────────────────────────
# Trained on a content-only item feature matrix with no item identity features.
print("\n\n── Training: Pure Content-Based ──────────────────────────")
model_content = LightFM(
    loss          = 'warp',
    no_components = NUM_COMPONENTS,
    item_alpha    = 1e-6,
    random_state  = SEED
)
model_content.fit(
    interactions,
    sample_weight = weights,
    item_features = content_item_features,
    epochs        = EPOCHS,
    num_threads   = NUM_THREADS,
    verbose       = True
)
results_content = evaluate_model(
    model_content, test_df,
    item_features_arg = content_item_features,
    label = "Pure Content-Based (metadata features only)"
)

# ── 3. Hybrid ─────────────────────────────────────────────────────────────────
# Both ID embeddings AND feature embeddings contribute
# item_alpha=1e-6 = minimal regularization → both signals are free to learn
print("\n\n── Training: Hybrid ──────────────────────────────────────")
model_hybrid = LightFM(
    loss          = 'warp',
    no_components = NUM_COMPONENTS,
    item_alpha    = 1e-6,
    random_state  = SEED
)
model_hybrid.fit(
    interactions,
    sample_weight = weights,
    item_features = hybrid_item_features,
    epochs        = EPOCHS,
    num_threads   = NUM_THREADS,
    verbose       = True
)
results_hybrid = evaluate_model(
    model_hybrid, test_df,
    item_features_arg = hybrid_item_features,
    label = "Hybrid (ID embeddings + content features)"
)

# ── 4. Random Baseline ─────────────────────────────────────────────────────────
print("\n\n── Evaluating: Random Baseline ───────────────────────────")
results_random = evaluate_random(test_df)

# ── 5. Popularity Baseline ─────────────────────────────────────────────────────
print("\n\n── Evaluating: Popularity Baseline ───────────────────────")
results_popularity = evaluate_popularity(test_df)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("  FINAL COMPARISON SUMMARY")
print("="*70)

model_results = [
    ("Random Baseline",    results_random),
    ("Popularity Baseline", results_popularity),
    ("Pure Collaborative", results_collab),
    ("Pure Content",       results_content),
    ("Hybrid",             results_hybrid),
]

for k in K_VALUES:
    print(f"\n  @ K = {k}")
    print(f"  {'Model':<24} {'HitRate':<10} {'Recall':<10} {'Precision':<12} {'NDCG'}")
    print(f"  {'-'*74}")
    for name, res in model_results:
        h = res[k]['hit_rate']
        r = res[k]['recall']
        p = res[k]['precision']
        n = res[k]['ndcg']
        print(f"  {name:<24} {h:<10.4f} {r:<10.4f} {p:<12.4f} {n:.4f}")

print("\n" + "="*70)
print("  KEY:")
print("  HitRate@K   — did any relevant item appear in top-K?")
print("  Recall@K    — fraction of relevant test items recovered in top-K")
print("  Precision@K — fraction of top-K that are relevant")
print("  NDCG@K      — rank-aware: rewards relevant items at top positions")
print("  Seen items  — masked from candidate rankings during evaluation")
print("  Popularity  — global weighted-item ranking from train only")
print("  Random      — pure chance lower bound")
print("="*70)
 
# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVE BEST MODEL (Hybrid)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── Saving model to {MODEL_DIR} ──────────────────────────────")

with open(f'{MODEL_DIR}/lightfm_collab.pkl', 'wb') as f:
    pickle.dump(model_collab, f)

with open(f'{MODEL_DIR}/lightfm_content.pkl', 'wb') as f:
    pickle.dump(model_content, f)

with open(f'{MODEL_DIR}/lightfm_hybrid.pkl', 'wb') as f:
    pickle.dump(model_hybrid, f)

with open(f'{MODEL_DIR}/dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

import scipy.sparse as sp
sp.save_npz(f'{MODEL_DIR}/item_features.npz', hybrid_item_features)
sp.save_npz(f'{MODEL_DIR}/item_features_hybrid.npz', hybrid_item_features)
sp.save_npz(f'{MODEL_DIR}/item_features_content.npz', content_item_features)

artifacts = {
    'user_id_map':    user_id_map,
    'item_id_map':    item_id_map,
    'inv_item_map':   inv_item_map,
    'item_to_title':  item_to_title,
    'item_to_price':  item_to_price,
    'item_to_text':   item_to_text,
    'user_seen_items': user_seen_items,
    'popular_items':   popular_items,
    'all_item_idxs':  all_item_idxs,
    'feature_matrix_files': {
        'hybrid': 'item_features_hybrid.npz',
        'content': 'item_features_content.npz',
    },
    'eval_results': {
        'random':  results_random,
        'popularity': results_popularity,
        'collab':  results_collab,
        'content': results_content,
        'hybrid':  results_hybrid,
    }
}
with open(f'{MODEL_DIR}/artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

import os
for fname in os.listdir(MODEL_DIR):
    size = os.path.getsize(f'{MODEL_DIR}/{fname}') / (1024*1024)
    print(f"  {fname:<30} {size:.2f} MB")

print("\nDone ✅")
print(f"Hybrid HitRate@10 : {results_hybrid[10]['hit_rate']}")
print(f"Hybrid Recall@10  : {results_hybrid[10]['recall']}")
print(f"Hybrid NDCG@10    : {results_hybrid[10]['ndcg']}")
