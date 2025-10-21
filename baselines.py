import logging
import numpy as np
import pandas as pd


def select_recent_window(transactions_df, baseline_days, use_promos=True, min_rows=50, max_expansions=2):
    """
    Select a recent slice of transactions for simple, stable baselines.
    If the recent slice is too small, the function expands the window a few times before falling back to all data.
    """
    df = transactions_df.copy()
    if not use_promos:
        df = df[df["promo_flag"] == 0]

    latest_time = df["date_time"].max()
    if pd.isna(latest_time):
        return df.iloc[0:0].copy()

    days = int(baseline_days)
    for _ in range(max_expansions + 1):
        cutoff = latest_time - pd.Timedelta(days=days)
        window = df[df["date_time"] >= cutoff].copy()
        if len(window) >= min_rows or days >= (baseline_days * (2 ** max_expansions)):
            if len(window) < min_rows:
                logging.warning("Baseline window is small (%d rows); proceeding anyway.", len(window))
            return window
        days *= 2

    logging.warning("Using full transaction history as baseline window due to sparse recent data.")
    return df.copy()


def compute_baseline_mix(window_df):
    """
    Compute recent model shares and recent SKU shares within each model from the selected window.
    Shares are floored at a tiny value and renormalized to avoid zero shares.
    """
    win = window_df.copy()

    sku_units = (
        win.groupby(["model_name", "SKU_id"], dropna=False)
        .size()
        .reset_index(name="units")
    )
    model_units = (
        sku_units.groupby("model_name", as_index=False)["units"].sum()
        .rename(columns={"units": "model_units"})
    )
    total_units = model_units["model_units"].sum()

    if total_units == 0:
        models = sorted(win["model_name"].dropna().unique().tolist())
        if not models:
            return (
                pd.DataFrame(columns=["model_name", "base_model_share"]),
                pd.DataFrame(columns=["model_name", "SKU_id", "base_sku_share"]),
            )
        base_model_share = pd.DataFrame({
            "model_name": models,
            "base_model_share": np.ones(len(models)) / len(models)
        })
        rows = []
        for m in models:
            skus = sorted(win.loc[win["model_name"] == m, "SKU_id"].unique().tolist())
            if skus:
                w = 1.0 / len(skus)
                for s in skus:
                    rows.append((m, s, w))
        base_sku_share = pd.DataFrame(rows, columns=["model_name", "SKU_id", "base_sku_share"])
        return base_model_share, base_sku_share

    base_model_share = model_units.copy()
    base_model_share["base_model_share"] = base_model_share["model_units"] / total_units
    base_model_share = base_model_share[["model_name", "base_model_share"]]

    sku_share = sku_units.merge(model_units, on="model_name", how="left")
    sku_share["base_sku_share"] = sku_share["units"] / sku_share["model_units"]
    base_sku_share = sku_share[["model_name", "SKU_id", "base_sku_share"]].copy()

    base_sku_share["base_sku_share"] = base_sku_share["base_sku_share"].clip(lower=1e-6)
    base_sku_share["base_sku_share"] = (
        base_sku_share
        .groupby("model_name")["base_sku_share"]
        .transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    )
    return base_model_share, base_sku_share


def compute_avg_price_per_sku(window_df):
    """
    Compute the average net price per SKU over the baseline window.
    This is used as a fallback for missing last prices.
    """
    avg_price = (
        window_df.groupby("SKU_id", dropna=False)["net_price"]
        .mean()
        .reset_index()
        .rename(columns={"net_price": "avg_price_sku"})
    )
    return avg_price


def compute_last_price(transactions_df, use_median_of_last_n=0, last_n=5):
    """
    Compute the most recent observed net price per SKU from all transactions.
    Optionally use the median of the last N sales for stability if recent prices are noisy.
    """
    df = transactions_df.sort_values("date_time").copy()
    rows = []
    for sku, group in df.groupby("SKU_id"):
        if group.empty:
            continue
        if use_median_of_last_n > 0:
            tail = group.tail(last_n)
            last_price = float(tail["net_price"].median())
        else:
            last_price = float(group["net_price"].iloc[-1])
        rows.append((sku, last_price))
    return pd.DataFrame(rows, columns=["SKU_id", "last_price"])


def derive_category_elasticity(elasticities_df, baseline_model_share_df, floor_share=0.01):
    """
    Compute a single category elasticity as the share-weighted average of model elasticities.
    If some models have zero share in the window, assign a small floor and renormalize; warn if the result is non-negative.
    """
    ela = elasticities_df[["model_name", "elasticity"]].copy()
    share = baseline_model_share_df[["model_name", "base_model_share"]].copy()
    merged = ela.merge(share, on="model_name", how="left")
    merged["base_model_share"] = merged["base_model_share"].fillna(0.0)

    if merged["base_model_share"].sum() == 0:
        weights = np.ones(len(merged)) / len(merged)
    else:
        weights = merged["base_model_share"].values
        weights = np.where(weights <= 0, floor_share, weights)
        weights = weights / weights.sum()

    epsilon_category = float((merged["elasticity"].values * weights).sum())
    if epsilon_category >= 0:
        logging.warning("Derived category elasticity was non-negative; forcing a small negative value.")
        epsilon_category = float(min(-0.1, epsilon_category))
    return epsilon_category
