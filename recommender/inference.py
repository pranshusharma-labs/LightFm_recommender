"""Inference helpers shared by the API and other interfaces."""

import re

import numpy as np


VARIANT_WORDS = {
    "xxs", "xs", "s", "m", "l", "xl", "xxl", "xxxl",
    "small", "medium", "large", "xlarge", "xxlarge",
    "red", "yellow", "blue", "green", "black", "white", "grey", "gray",
    "navy", "pink", "purple", "orange", "brown", "charcoal", "silver",
    "gold", "maroon", "heather", "multi", "multicolor",
}


def _base_product_key(title):
    """Collapse product title variants like size/color into one base key."""
    cleaned = re.sub(r"\([^)]*\)", " ", (title or "").lower())
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    base_tokens = [token for token in tokens if token not in VARIANT_WORDS]
    return " ".join(base_tokens).strip() or cleaned.strip()


def build_popularity_recommendations(popular_items, top_k):
    """Return top-k popularity fallback recommendations."""
    recommendations = []
    seen_base_products = set()

    for item in popular_items:
        base_key = _base_product_key(item["title"])
        if base_key in seen_base_products:
            continue

        seen_base_products.add(base_key)
        recommendations.append(
            {
                "rank": int(len(recommendations) + 1),
                "item_key": item["item_key"],
                "title": item["title"],
                "score": item["score"],
            }
        )
        if len(recommendations) == top_k:
            break

    return recommendations


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
    seen_base_products = set()
    for idx in ranked:
        item_key = inv_item_map[idx]
        if item_key in seen_items:
            continue

        title = item_to_title.get(item_key, "")
        base_key = _base_product_key(title)
        if base_key in seen_base_products:
            continue

        seen_base_products.add(base_key)
        recommendations.append(
            {
                "rank": int(len(recommendations) + 1),
                "item_key": item_key,
                "title": title,
                "score": round(float(scores[idx]), 4),
            }
        )
        if len(recommendations) == top_k:
            break

    return recommendations
