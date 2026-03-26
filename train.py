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

dataset = Dataset()
dataset.fit(
    users         = train_df['user_id'].unique(),
    items         = products['product_id'].unique(),
    item_features = all_features
)

# Build interaction matrix from train only — no test leakage
interactions, weights = dataset.build_interactions(
    (row['user_id'], row['item_id'], row['weight'])
    for _, row in train_df.iterrows()
    if row['item_id'] in set(products['product_id'])
)

# Build item feature matrix
item_features = dataset.build_item_features(
    (row['product_id'], row['feature_list'])
    for _, row in products.iterrows()
)

user_id_map, _, item_id_map, _ = dataset.mapping()
inv_item_map  = {v: k for k, v in item_id_map.items()}
all_item_idxs = np.arange(len(item_id_map))

# Item metadata for API
item_to_title = dict(zip(products['product_id'], products['title']))
item_to_price = dict(zip(products['product_id'], products['price']))
item_to_text  = dict(zip(
    products['product_id'],
    products.apply(build_item_text, axis=1)
))

print(f"Users in dataset  : {len(user_id_map):,}")
print(f"Items in dataset  : {len(item_id_map):,}")
print(f"Item features     : {len(all_features)}")
print(f"Interaction nnz   : {interactions.nnz:,}")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, test_df, item_features_arg, label):
    """Compute Recall@K, Precision@K, NDCG@K."""
    results = {k: {'recall': [], 'precision': [], 'ndcg': []} for k in K_VALUES}
    skipped = 0

    for user, group in tqdm(test_df.groupby('user_id'), desc=f"Eval [{label}]"):
        if user not in user_id_map:
            skipped += 1
            continue

        user_idx   = user_id_map[user]
        true_items = set(group['item_id'].tolist())

        scores = model.predict(
            user_idx,
            all_item_idxs,
            item_features=item_features_arg
        )
        ranked = np.argsort(-scores)

        for k in K_VALUES:
            top_k  = [inv_item_map[i] for i in ranked[:k]]
            hits   = [1 if item in true_items else 0 for item in top_k]
            n_hits = sum(hits)

            results[k]['recall'].append(1 if n_hits > 0 else 0)
            results[k]['precision'].append(n_hits / k)

            dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            results[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)

    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(f"  {'K':<5} {'Recall@K':<12} {'Precision@K':<14} {'NDCG@K'}")
    print(f"  {'-'*46}")
    summary = {}
    for k in K_VALUES:
        r = np.mean(results[k]['recall'])
        p = np.mean(results[k]['precision'])
        n = np.mean(results[k]['ndcg'])
        summary[k] = {'recall': round(r,4), 'precision': round(p,4), 'ndcg': round(n,4)}
        print(f"  K={k:<3} {r:<12.4f} {p:<14.4f} {n:.4f}")
    print(f"{'='*58}")
    if skipped:
        print(f"  Skipped {skipped} cold-start users")
    return summary


def evaluate_random(test_df):
    """Random baseline — lower bound for all metrics."""
    results = {k: {'recall': [], 'precision': [], 'ndcg': []} for k in K_VALUES}
    rng     = np.random.default_rng(SEED)
    n_items = len(item_id_map)

    for user, group in test_df.groupby('user_id'):
        if user not in user_id_map:
            continue
        true_items = set(group['item_id'].tolist())
        ranked     = [inv_item_map[i] for i in rng.permutation(n_items)]

        for k in K_VALUES:
            top_k  = ranked[:k]
            hits   = [1 if item in true_items else 0 for item in top_k]
            n_hits = sum(hits)
            results[k]['recall'].append(1 if n_hits > 0 else 0)
            results[k]['precision'].append(n_hits / k)
            dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            results[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)

    print(f"\n{'='*58}")
    print(f"  Random Baseline")
    print(f"{'='*58}")
    print(f"  {'K':<5} {'Recall@K':<12} {'Precision@K':<14} {'NDCG@K'}")
    print(f"  {'-'*46}")
    summary = {}
    for k in K_VALUES:
        r = np.mean(results[k]['recall'])
        p = np.mean(results[k]['precision'])
        n = np.mean(results[k]['ndcg'])
        summary[k] = {'recall': round(r,4), 'precision': round(p,4), 'ndcg': round(n,4)}
        print(f"  K={k:<3} {r:<12.4f} {p:<14.4f} {n:.4f}")
    print(f"{'='*58}")
    return summary


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
# Features used, ID embeddings heavily regularized → suppressed
# item_alpha=1e-3 pushes ID embedding weights toward zero
# so model must rely on feature embeddings (category, brand, tags, price)
print("\n\n── Training: Pure Content-Based ──────────────────────────")
model_content = LightFM(
    loss          = 'warp',
    no_components = NUM_COMPONENTS,
    item_alpha    = 1e-3,
    random_state  = SEED
)
model_content.fit(
    interactions,
    sample_weight = weights,
    item_features = item_features,
    epochs        = EPOCHS,
    num_threads   = NUM_THREADS,
    verbose       = True
)
results_content = evaluate_model(
    model_content, test_df,
    item_features_arg = item_features,
    label = "Pure Content-Based (features only, ID embeddings suppressed)"
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
    item_features = item_features,
    epochs        = EPOCHS,
    num_threads   = NUM_THREADS,
    verbose       = True
)
results_hybrid = evaluate_model(
    model_hybrid, test_df,
    item_features_arg = item_features,
    label = "Hybrid (ID embeddings + content features)"
)

# ── 4. Random Baseline ─────────────────────────────────────────────────────────
print("\n\n── Evaluating: Random Baseline ───────────────────────────")
results_random = evaluate_random(test_df)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("  FINAL COMPARISON SUMMARY")
print("="*70)

model_results = [
    ("Random Baseline",    results_random),
    ("Pure Collaborative", results_collab),
    ("Pure Content",       results_content),
    ("Hybrid",             results_hybrid),
]

for k in K_VALUES:
    print(f"\n  @ K = {k}")
    print(f"  {'Model':<35} {'Recall':<10} {'Precision':<12} {'NDCG'}")
    print(f"  {'-'*65}")
    for name, res in model_results:
        r = res[k]['recall']
        p = res[k]['precision']
        n = res[k]['ndcg']
        print(f"  {name:<35} {r:<10.4f} {p:<12.4f} {n:.4f}")

print("\n" + "="*70)
print("  KEY:")
print("  Recall@K    — did any relevant item appear in top-K? (hit rate)")
print("  Precision@K — fraction of top-K that are relevant")
print("  NDCG@K      — rank-aware: rewards relevant items at top positions")
print("  Random      — pure chance lower bound")
print("="*70)
 
# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVE BEST MODEL (Hybrid)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── Saving model to {MODEL_DIR} ──────────────────────────────")

with open(f'{MODEL_DIR}/lightfm_hybrid.pkl', 'wb') as f:
    pickle.dump(model_hybrid, f)

with open(f'{MODEL_DIR}/dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

import scipy.sparse as sp
sp.save_npz(f'{MODEL_DIR}/item_features.npz', item_features)

artifacts = {
    'user_id_map':    user_id_map,
    'item_id_map':    item_id_map,
    'inv_item_map':   inv_item_map,
    'item_to_title':  item_to_title,
    'item_to_price':  item_to_price,
    'item_to_text':   item_to_text,
    'all_item_idxs':  all_item_idxs,
    'eval_results': {
        'random':  results_random,
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
print(f"Hybrid Recall@10  : {results_hybrid[10]['recall']}")
print(f"Hybrid NDCG@10    : {results_hybrid[10]['ndcg']}")
