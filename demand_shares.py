import numpy as np
import pandas as pd


def typical_model_price(model_prices, baseline_sku_share_for_model):
    """
    Compute a model’s representative price as a baseline-share-weighted average of its SKU prices.
    This stabilizes the representative price so it does not feed back on itself during optimization.
    """
    df = baseline_sku_share_for_model.copy()
    df["price"] = df["SKU_id"].map(model_prices).astype(float)
    df = df.dropna(subset=["price"])
    if df.empty:
        return 0.0
    return float((df["base_sku_share"] * df["price"]).sum())


def softmax(values):
    """
    Convert a list of scores into probabilities that sum to one.
    This is numerically stable by subtracting the maximum score before exponentiation.
    """
    arr = np.array(values, dtype=float)
    arr = arr - arr.max()
    exp = np.exp(arr)
    total = exp.sum()
    if total <= 0:
        return np.ones_like(exp) / len(exp)
    return exp / total


def split_between_models(model_representative_prices, baseline_model_share_df, k_model):
    """
    Convert model representative prices into model shares that sum to one.
    Cheaper models gain share relative to the cheapest model based on a single sensitivity k_model.
    """
    base_share_map = baseline_model_share_df.set_index("model_name")["base_model_share"].to_dict()
    names = list(model_representative_prices.keys())
    if not names:
        return {}

    min_rep_price = min(model_representative_prices.values())
    scores = []
    for model in names:
        base_share = max(1e-9, float(base_share_map.get(model, 1e-9)))
        price = float(model_representative_prices[model])
        score = np.log(base_share) - float(k_model) * (price - min_rep_price)
        scores.append(score)

    probs = softmax(scores)
    return {model: float(p) for model, p in zip(names, probs)}


def split_within_model(model_name, model_sku_prices, baseline_sku_share_df, k_sku, min_share_sku):
    """
    Convert SKU prices within a model into SKU shares that sum to one.
    Cheaper SKUs gain share relative to the cheapest SKU; a small floor keeps niche SKUs from collapsing to zero.
    """
    df = baseline_sku_share_df[baseline_sku_share_df["model_name"] == model_name][["SKU_id", "base_sku_share"]].copy()
    df = df[df["SKU_id"].isin(model_sku_prices.keys())].copy()
    if df.empty:
        return {}

    min_price_in_model = min([float(model_sku_prices[sku]) for sku in df["SKU_id"]])
    scores = []
    for _, row in df.iterrows():
        sku = row["SKU_id"]
        base_share = max(1e-9, float(row["base_sku_share"]))
        price = float(model_sku_prices[sku])
        score = np.log(base_share) - float(k_sku) * (price - min_price_in_model)
        scores.append(score)

    probs = softmax(scores)
    shares = pd.Series(probs, index=df["SKU_id"]).to_dict()

    floor = float(min_share_sku)
    adjusted = {k: max(floor, v) for k, v in shares.items()}
    total = sum(adjusted.values())
    if total <= 0:
        n = len(adjusted)
        return {k: 1.0 / n for k in adjusted}
    return {k: v / total for k, v in adjusted.items()}


def compute_demand_level(category_average_price, category_elasticity):
    """
    Compute a single demand multiplier from the category’s average price and elasticity.
    This multiplier scales all SKUs equally; absolute units are not required to rank price sets.
    """
    price = max(1e-6, float(category_average_price))
    return float(price ** float(category_elasticity))
