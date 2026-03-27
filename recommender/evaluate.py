"""Evaluation helpers for ranking metrics and baselines."""

import numpy as np
from tqdm import tqdm


def summarize_results(results, label, k_values, skipped=0):
    """Print and round evaluation metrics."""
    print(f"\n{'=' * 78}")
    print(f"  {label}")
    print(f"{'=' * 78}")
    print(f"  {'K':<5} {'HitRate@K':<12} {'Recall@K':<12} {'Precision@K':<14} {'NDCG@K'}")
    print(f"  {'-' * 66}")

    summary = {}
    for k in k_values:
        hit_rate = np.mean(results[k]["hit_rate"])
        recall = np.mean(results[k]["recall"])
        precision = np.mean(results[k]["precision"])
        ndcg = np.mean(results[k]["ndcg"])

        summary[k] = {
            "hit_rate": round(hit_rate, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "ndcg": round(ndcg, 4),
        }
        print(f"  K={k:<3} {hit_rate:<12.4f} {recall:<12.4f} {precision:<14.4f} {ndcg:.4f}")

    print(f"{'=' * 78}")
    if skipped:
        print(f"  Skipped {skipped} cold-start users")
    return summary


def evaluate_model(
    model,
    test_df,
    item_features_arg,
    label,
    *,
    k_values,
    user_id_map,
    inv_item_map,
    all_item_idxs,
    user_seen_item_idxs,
):
    """Compute hit rate, true recall, precision, and NDCG with seen-item masking."""
    results = {
        k: {"hit_rate": [], "recall": [], "precision": [], "ndcg": []}
        for k in k_values
    }
    skipped = 0

    for user, group in tqdm(test_df.groupby("user_id"), desc=f"Eval [{label}]"):
        if user not in user_id_map:
            skipped += 1
            continue

        user_idx = user_id_map[user]
        true_items = set(group["item_id"].tolist())
        seen_idxs = user_seen_item_idxs.get(user, np.array([], dtype=np.int32))

        scores = model.predict(
            user_idx,
            all_item_idxs,
            item_features=item_features_arg,
        )
        masked_scores = scores.copy()
        if seen_idxs.size:
            masked_scores[seen_idxs] = -np.inf
        ranked = np.argsort(-masked_scores)

        for k in k_values:
            top_k = [inv_item_map[i] for i in ranked[:k]]
            hits = [1 if item in true_items else 0 for item in top_k]
            n_hits = sum(hits)

            results[k]["hit_rate"].append(1 if n_hits > 0 else 0)
            results[k]["recall"].append(n_hits / len(true_items) if true_items else 0)
            results[k]["precision"].append(n_hits / k)

            dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            results[k]["ndcg"].append(dcg / idcg if idcg > 0 else 0)

    return summarize_results(results, label, k_values, skipped=skipped)


def evaluate_random(
    test_df,
    *,
    k_values,
    seed,
    item_id_map,
    user_id_map,
    inv_item_map,
    user_seen_items,
):
    """Random baseline with seen-item masking."""
    results = {
        k: {"hit_rate": [], "recall": [], "precision": [], "ndcg": []}
        for k in k_values
    }
    rng = np.random.default_rng(seed)
    n_items = len(item_id_map)

    for user, group in test_df.groupby("user_id"):
        if user not in user_id_map:
            continue

        true_items = set(group["item_id"].tolist())
        seen_items = set(user_seen_items.get(user, []))
        ranked = [
            inv_item_map[i]
            for i in rng.permutation(n_items)
            if inv_item_map[i] not in seen_items
        ]

        for k in k_values:
            top_k = ranked[:k]
            hits = [1 if item in true_items else 0 for item in top_k]
            n_hits = sum(hits)
            results[k]["hit_rate"].append(1 if n_hits > 0 else 0)
            results[k]["recall"].append(n_hits / len(true_items) if true_items else 0)
            results[k]["precision"].append(n_hits / k)
            dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            results[k]["ndcg"].append(dcg / idcg if idcg > 0 else 0)

    return summarize_results(results, "Random Baseline", k_values)


def evaluate_popularity(
    test_df,
    *,
    k_values,
    user_id_map,
    user_seen_items,
    popular_item_ids,
):
    """Popularity baseline with seen-item masking."""
    results = {
        k: {"hit_rate": [], "recall": [], "precision": [], "ndcg": []}
        for k in k_values
    }

    for user, group in test_df.groupby("user_id"):
        if user not in user_id_map:
            continue

        true_items = set(group["item_id"].tolist())
        seen_items = set(user_seen_items.get(user, []))
        ranked = [item_id for item_id in popular_item_ids if item_id not in seen_items]

        for k in k_values:
            top_k = ranked[:k]
            hits = [1 if item in true_items else 0 for item in top_k]
            n_hits = sum(hits)
            results[k]["hit_rate"].append(1 if n_hits > 0 else 0)
            results[k]["recall"].append(n_hits / len(true_items) if true_items else 0)
            results[k]["precision"].append(n_hits / k)
            dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            results[k]["ndcg"].append(dcg / idcg if idcg > 0 else 0)

    return summarize_results(results, "Popularity Baseline", k_values)

