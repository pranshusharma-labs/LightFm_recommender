"""Inference helpers shared by the API and other interfaces."""

import numpy as np


def build_popularity_recommendations(popular_items, top_k):
    """Return top-k popularity fallback recommendations."""
    return [
        {
            "rank": int(i + 1),
            "item_key": item["item_key"],
            "title": item["title"],
            "score": item["score"],
        }
        for i, item in enumerate(popular_items[:top_k])
    ]


def rank_recommendations(
    model,
    user_idx,
    all_item_idxs,
    inv_item_map,
    item_to_title,
    seen_items,
    top_k,
    *,
    item_features=None,
):
    """Rank candidate items for a user and filter already-seen items."""
    scores = model.predict(user_idx, all_item_idxs, item_features=item_features)
    ranked = np.argsort(-scores)

    recommendations = []
    for idx in ranked:
        item_key = inv_item_map[idx]
        if item_key in seen_items:
            continue

        recommendations.append(
            {
                "rank": int(len(recommendations) + 1),
                "item_key": item_key,
                "title": item_to_title.get(item_key, ""),
                "score": round(float(scores[idx]), 4),
            }
        )
        if len(recommendations) == top_k:
            break

    return recommendations
