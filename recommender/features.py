"""Feature engineering helpers for the recommender models."""


def get_item_features(row):
    """Build sparse metadata features for a product row."""
    features = []

    if row["category"] and row["category"] != "unknown":
        cat = row["category"].replace("Home/", "")
        for part in cat.split("/"):
            part = part.strip()
            if part:
                features.append(f"cat:{part}")

    if row["brand"] and row["brand"] not in ["", "nan", "Google"]:
        features.append(f"brand:{row['brand']}")

    if row["tags"] and row["tags"] not in ["", "nan", "(not set)"]:
        for tag in row["tags"].split(","):
            tag = tag.strip()
            if tag:
                features.append(f"tag:{tag}")

    if row["price"] > 0:
        if row["price"] < 15:
            features.append("price:low")
        elif row["price"] < 40:
            features.append("price:mid")
        else:
            features.append("price:high")

    return features if features else ["cat:unknown"]


def attach_item_features(products):
    """Return a copy of products with feature_list plus all unique features."""
    products = products.copy()
    products["feature_list"] = products.apply(get_item_features, axis=1)
    all_features = sorted(
        set(feature for features in products["feature_list"] for feature in features)
    )
    return products, all_features

