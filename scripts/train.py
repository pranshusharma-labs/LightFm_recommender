"""
scripts/train.py
────────────────
Train LightFM collaborative, content, and hybrid recommenders.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.data import Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recommender.data_loader import (
    apply_weights_and_decay,
    build_item_text,
    load_events,
    load_products,
    resolve_item_ids,
    split_data,
)
from recommender.evaluate import (
    evaluate_model,
    evaluate_popularity,
    evaluate_random,
)
from recommender.features import attach_item_features


PRODUCTS_PATH = "data/products.csv"
EVENTS_PATH = "data/events.csv"
MODEL_DIR = "model/"
EPOCHS = 20
NUM_THREADS = 4
NUM_COMPONENTS = 64
K_VALUES = [1, 5, 10]
SEED = 42


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n── Step 1: Loading data ─────────────────────────────────────")
    products = load_products(PRODUCTS_PATH)
    events = load_events(EVENTS_PATH)
    events = resolve_item_ids(events, products)
    events = apply_weights_and_decay(events)

    print("\n── Step 2: Splitting data ───────────────────────────────────")
    train_df, test_df = split_data(events)

    print("\n── Step 3: Building LightFM dataset ────────────────────────")
    products, all_features = attach_item_features(products)

    user_ids = train_df["user_id"].drop_duplicates().tolist()
    item_ids = products["product_id"].drop_duplicates().tolist()

    dataset = Dataset()
    dataset.fit(users=user_ids, items=item_ids, item_features=all_features)

    product_id_set = set(products["product_id"])
    interactions, weights = dataset.build_interactions(
        (row["user_id"], row["item_id"], row["weight"])
        for _, row in train_df.iterrows()
        if row["item_id"] in product_id_set
    )

    hybrid_item_features = dataset.build_item_features(
        (row["product_id"], row["feature_list"])
        for _, row in products.iterrows()
    )

    user_id_map, _, item_id_map, _ = dataset.mapping()
    inv_item_map = {v: k for k, v in item_id_map.items()}
    all_item_idxs = np.arange(len(item_id_map))

    content_dataset = Dataset(item_identity_features=False)
    content_dataset.fit(users=user_ids, items=item_ids, item_features=all_features)
    content_user_id_map, _, content_item_id_map, _ = content_dataset.mapping()
    if user_id_map != content_user_id_map or item_id_map != content_item_id_map:
        raise ValueError("Content dataset mappings diverged from hybrid dataset mappings.")

    content_item_features = content_dataset.build_item_features(
        (row["product_id"], row["feature_list"])
        for _, row in products.iterrows()
    )

    item_to_title = dict(zip(products["product_id"], products["title"]))
    item_to_price = dict(zip(products["product_id"], products["price"]))
    item_to_text = dict(
        zip(products["product_id"], products.apply(build_item_text, axis=1))
    )

    catalog_item_ids = set(item_id_map.keys())
    user_seen_items = (
        train_df[train_df["item_id"].isin(catalog_item_ids)]
        .groupby("user_id")["item_id"]
        .agg(lambda items: sorted(set(items)))
        .to_dict()
    )
    popular_items_df = (
        train_df[train_df["item_id"].isin(catalog_item_ids)]
        .groupby("item_id", as_index=False)["weight"]
        .sum()
        .sort_values(["weight", "item_id"], ascending=[False, True])
    )
    popular_items = [
        {
            "item_key": row["item_id"],
            "title": item_to_title.get(row["item_id"], row["item_id"]),
            "score": round(float(row["weight"]), 4),
        }
        for _, row in popular_items_df.iterrows()
    ]
    popular_item_ids = [item["item_key"] for item in popular_items]
    user_seen_item_idxs = {
        user: np.array(
            [item_id_map[item_id] for item_id in item_ids if item_id in item_id_map],
            dtype=np.int32,
        )
        for user, item_ids in user_seen_items.items()
    }

    print(f"Users in dataset    : {len(user_id_map):,}")
    print(f"Items in dataset    : {len(item_id_map):,}")
    print(f"Item features       : {len(all_features)}")
    print(f"Hybrid feature dim  : {hybrid_item_features.shape[1]:,}")
    print(f"Content feature dim : {content_item_features.shape[1]:,}")
    print(f"Interaction nnz     : {interactions.nnz:,}")

    print("\n\n── Training: Pure Collaborative ──────────────────────────")
    model_collab = LightFM(
        loss="warp",
        no_components=NUM_COMPONENTS,
        random_state=SEED,
    )
    model_collab.fit(
        interactions,
        sample_weight=weights,
        epochs=EPOCHS,
        num_threads=NUM_THREADS,
        verbose=True,
    )
    results_collab = evaluate_model(
        model_collab,
        test_df,
        item_features_arg=None,
        label="Pure Collaborative (ID embeddings only)",
        k_values=K_VALUES,
        user_id_map=user_id_map,
        inv_item_map=inv_item_map,
        all_item_idxs=all_item_idxs,
        user_seen_item_idxs=user_seen_item_idxs,
    )

    print("\n\n── Training: Pure Content-Based ──────────────────────────")
    model_content = LightFM(
        loss="warp",
        no_components=NUM_COMPONENTS,
        item_alpha=1e-6,
        random_state=SEED,
    )
    model_content.fit(
        interactions,
        sample_weight=weights,
        item_features=content_item_features,
        epochs=EPOCHS,
        num_threads=NUM_THREADS,
        verbose=True,
    )
    results_content = evaluate_model(
        model_content,
        test_df,
        item_features_arg=content_item_features,
        label="Pure Content-Based (metadata features only)",
        k_values=K_VALUES,
        user_id_map=user_id_map,
        inv_item_map=inv_item_map,
        all_item_idxs=all_item_idxs,
        user_seen_item_idxs=user_seen_item_idxs,
    )

    print("\n\n── Training: Hybrid ──────────────────────────────────────")
    model_hybrid = LightFM(
        loss="warp",
        no_components=NUM_COMPONENTS,
        item_alpha=1e-6,
        random_state=SEED,
    )
    model_hybrid.fit(
        interactions,
        sample_weight=weights,
        item_features=hybrid_item_features,
        epochs=EPOCHS,
        num_threads=NUM_THREADS,
        verbose=True,
    )
    results_hybrid = evaluate_model(
        model_hybrid,
        test_df,
        item_features_arg=hybrid_item_features,
        label="Hybrid (ID embeddings + content features)",
        k_values=K_VALUES,
        user_id_map=user_id_map,
        inv_item_map=inv_item_map,
        all_item_idxs=all_item_idxs,
        user_seen_item_idxs=user_seen_item_idxs,
    )

    print("\n\n── Evaluating: Random Baseline ───────────────────────────")
    results_random = evaluate_random(
        test_df,
        k_values=K_VALUES,
        seed=SEED,
        item_id_map=item_id_map,
        user_id_map=user_id_map,
        inv_item_map=inv_item_map,
        user_seen_items=user_seen_items,
    )

    print("\n\n── Evaluating: Popularity Baseline ───────────────────────")
    results_popularity = evaluate_popularity(
        test_df,
        k_values=K_VALUES,
        user_id_map=user_id_map,
        user_seen_items=user_seen_items,
        popular_item_ids=popular_item_ids,
    )

    print("\n\n" + "=" * 70)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 70)

    model_results = [
        ("Random Baseline", results_random),
        ("Popularity Baseline", results_popularity),
        ("Pure Collaborative", results_collab),
        ("Pure Content", results_content),
        ("Hybrid", results_hybrid),
    ]

    for k in K_VALUES:
        print(f"\n  @ K = {k}")
        print(f"  {'Model':<24} {'HitRate':<10} {'Recall':<10} {'Precision':<12} {'NDCG'}")
        print(f"  {'-' * 74}")
        for name, result in model_results:
            hit_rate = result[k]["hit_rate"]
            recall = result[k]["recall"]
            precision = result[k]["precision"]
            ndcg = result[k]["ndcg"]
            print(f"  {name:<24} {hit_rate:<10.4f} {recall:<10.4f} {precision:<12.4f} {ndcg:.4f}")

    print("\n" + "=" * 70)
    print("  KEY:")
    print("  HitRate@K   — did any relevant item appear in top-K?")
    print("  Recall@K    — fraction of relevant test items recovered in top-K")
    print("  Precision@K — fraction of top-K that are relevant")
    print("  NDCG@K      — rank-aware: rewards relevant items at top positions")
    print("  Seen items  — masked from candidate rankings during evaluation")
    print("  Popularity  — global weighted-item ranking from train only")
    print("  Random      — pure chance lower bound")
    print("=" * 70)

    print(f"\n── Saving model to {MODEL_DIR} ──────────────────────────────")

    with open(f"{MODEL_DIR}/lightfm_collab.pkl", "wb") as f:
        pickle.dump(model_collab, f)

    with open(f"{MODEL_DIR}/lightfm_content.pkl", "wb") as f:
        pickle.dump(model_content, f)

    with open(f"{MODEL_DIR}/lightfm_hybrid.pkl", "wb") as f:
        pickle.dump(model_hybrid, f)

    with open(f"{MODEL_DIR}/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    sp.save_npz(f"{MODEL_DIR}/item_features.npz", hybrid_item_features)
    sp.save_npz(f"{MODEL_DIR}/item_features_hybrid.npz", hybrid_item_features)
    sp.save_npz(f"{MODEL_DIR}/item_features_content.npz", content_item_features)

    artifacts = {
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "inv_item_map": inv_item_map,
        "item_to_title": item_to_title,
        "item_to_price": item_to_price,
        "item_to_text": item_to_text,
        "user_seen_items": user_seen_items,
        "popular_items": popular_items,
        "all_item_idxs": all_item_idxs,
        "feature_matrix_files": {
            "hybrid": "item_features_hybrid.npz",
            "content": "item_features_content.npz",
        },
        "eval_results": {
            "random": results_random,
            "popularity": results_popularity,
            "collab": results_collab,
            "content": results_content,
            "hybrid": results_hybrid,
        },
    }
    with open(f"{MODEL_DIR}/artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    for fname in os.listdir(MODEL_DIR):
        size = os.path.getsize(f"{MODEL_DIR}/{fname}") / (1024 * 1024)
        print(f"  {fname:<30} {size:.2f} MB")

    print("\nDone ✅")
    print(f"Hybrid HitRate@10 : {results_hybrid[10]['hit_rate']}")
    print(f"Hybrid Recall@10  : {results_hybrid[10]['recall']}")
    print(f"Hybrid NDCG@10    : {results_hybrid[10]['ndcg']}")


if __name__ == "__main__":
    main()
