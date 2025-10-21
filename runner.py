import logging
import yaml
import pandas as pd
from data_io import load_transactions, load_elasticities, load_min_price
from baselines import (
    select_recent_window,
    compute_baseline_mix,
    compute_avg_price_per_sku,
    compute_last_price,
    derive_category_elasticity,
)
from optimize import build_profit_objective, coordinate_descent
from constraints import make_prices_valid


def recommend_prices(transactions_df, elasticities_df, min_price_df, cfg):
    """
    Run the full pipeline and return a DataFrame of recommended prices per SKU.
    The function keeps the code path short and readable: prepare baselines, build the objective, and optimize under simple rules.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Validate configuration
    if cfg["step"] % cfg["price_tick"] != 0:
        raise ValueError("Config error: 'step' must be a multiple of 'price_tick'.")

    # 1) Load and clean inputs
    transactions = load_transactions(transactions_df)
    elasticities = load_elasticities(elasticities_df)
    min_price = load_min_price(min_price_df)

    # 2) Select recent window for simple baselines
    window = select_recent_window(
        transactions,
        baseline_days=cfg["baseline_days"],
        use_promos=cfg["use_promos"],
        min_rows=50,
        max_expansions=2
    )

    # 3) Baseline shares (models and SKUs within models)
    baseline_model_share_df, baseline_sku_share_df = compute_baseline_mix(window)

    # 4) Fallback average price per SKU (used only if last price is missing)
    avg_price_sku_df = compute_avg_price_per_sku(window)

    # 5) Last observed price per SKU
    last_price_df = compute_last_price(transactions)

    # 6) Category elasticity (weighted average of model elasticities)
    category_elasticity = derive_category_elasticity(elasticities, baseline_model_share_df)

    # 7) Assemble active sets and lookups
    models = sorted(baseline_model_share_df["model_name"].unique().tolist())
    skus_from_baseline = set(baseline_sku_share_df["SKU_id"])
    skus_from_min = set(min_price["SKU_id"])
    active_skus = skus_from_baseline | skus_from_min

    min_price = min_price[min_price["SKU_id"].isin(active_skus)].copy()
    last_price_df = last_price_df[last_price_df["SKU_id"].isin(active_skus)].copy()
    avg_price_sku_df = avg_price_sku_df[avg_price_sku_df["SKU_id"].isin(active_skus)].copy()

    # Ensure every active SKU is present in baseline_sku_share_df; assign a small share if missing
    missing_skus = active_skus - set(baseline_sku_share_df["SKU_id"])
    if missing_skus:
        rows = []
        for sku in sorted(missing_skus):
            model_name_series = transactions.loc[transactions["SKU_id"] == sku, "model_name"].tail(1)
            model_name = model_name_series.iloc[0] if len(model_name_series) else "unknown"
            rows.append({"model_name": model_name, "SKU_id": sku, "base_sku_share": cfg["min_share_sku"]})
        baseline_sku_share_df = pd.concat([baseline_sku_share_df, pd.DataFrame(rows)], ignore_index=True)
        baseline_sku_share_df["base_sku_share"] = (
            baseline_sku_share_df
            .groupby("model_name")["base_sku_share"]
            .transform(lambda x: x / x.sum() if x.sum() > 0 else x)
        )

    # Build mapping model -> list of SKUs
    skus_by_model = (
        baseline_sku_share_df.groupby("model_name")["SKU_id"]
        .apply(list)
        .to_dict()
    )
    if "unknown" in skus_by_model:
        logging.warning("Some SKUs had unknown model; excluding them from optimization.")
        del skus_by_model["unknown"]
        baseline_sku_share_df = baseline_sku_share_df[baseline_sku_share_df["model_name"] != "unknown"]
        models = sorted(baseline_sku_share_df["model_name"].unique().tolist())
        active_skus = set(baseline_sku_share_df["SKU_id"])
        min_price = min_price[min_price["SKU_id"].isin(active_skus)].copy()
        last_price_df = last_price_df[last_price_df["SKU_id"].isin(active_skus)].copy()
        avg_price_sku_df = avg_price_sku_df[avg_price_sku_df["SKU_id"].isin(active_skus)].copy()

    # Costs: take the most recent cost per SKU within the window, else from full data
    cost_df = window.sort_values("date_time").groupby("SKU_id", as_index=False).tail(1)[["SKU_id", "cost_unit"]]
    if cost_df.empty:
        cost_df = transactions.sort_values("date_time").groupby("SKU_id", as_index=False).tail(1)[["SKU_id", "cost_unit"]]
    cost_by_sku = {r["SKU_id"]: float(r["cost_unit"]) for _, r in cost_df.iterrows()}

    # Starting prices: last observed, with fallback to average price in window
    start_prices = {r["SKU_id"]: float(r["last_price"]) for _, r in last_price_df.iterrows()}
    for _, r in avg_price_sku_df.iterrows():
        sku = r["SKU_id"]
        if sku not in start_prices:
            start_prices[sku] = float(r["avg_price_sku"])

    # Keep only active SKUs in start prices
    start_prices = {k: v for k, v in start_prices.items() if k in active_skus}

    # 8) Build the profit objective (now with clearer naming)
    profit_fn = build_profit_objective(
        cost_by_sku=cost_by_sku,
        category_elasticity=category_elasticity,
        baseline_model_share_df=baseline_model_share_df,
        baseline_sku_share_df=baseline_sku_share_df,
        k_model=cfg["k_model"],
        k_sku=cfg["k_sku"],
        min_share_sku=cfg["min_share_sku"],
    )

    # 9) Optimize jointly with coordinate descent
    best_prices = coordinate_descent(
        models=models,
        skus_by_model=skus_by_model,
        start_prices=start_prices,
        evaluate_profit_fn=profit_fn,
        min_price_df=min_price,
        last_price_df=last_price_df,
        max_move_pct=cfg["max_price_move_pct"],
        price_tick=cfg["price_tick"],
        step=cfg["step"],
        max_iter=cfg["max_iter"],
    )

    # 10) Final validation pass through the rules
    best_prices = make_prices_valid(
        best_prices, min_price, last_price_df,
        max_move_pct=cfg["max_price_move_pct"], price_tick=cfg["price_tick"]
    )

    # 11) Return tidy DataFrame
    sku_to_model = (
        baseline_sku_share_df.drop_duplicates(subset=["SKU_id"])[["SKU_id", "model_name"]]
        .set_index("SKU_id")["model_name"].to_dict()
    )
    rows = [{"model_name": sku_to_model.get(sku, "unknown"),
             "SKU_id": sku,
             "recommended_price": float(price)}
            for sku, price in best_prices.items()]
    out = pd.DataFrame(rows).sort_values(["model_name", "SKU_id"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="S-series joint pricing MVP")
    parser.add_argument("--transactions", default="data/transactions.csv", help="Path to transactions CSV")
    parser.add_argument("--elasticities", default="data/elasticities.csv",
                        help="Path to model elasticities CSV")
    parser.add_argument("--min_price", default="data/min_price.csv", help="Path to minimum price CSV")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--out", default="output/recommended_prices.csv",
                        help="Path to write recommended_prices.csv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config["step"] % config["price_tick"] != 0:
        raise ValueError("Config error: 'step' must be a multiple of 'price_tick'.")

    tx = pd.read_csv(args.transactions)
    elas = pd.read_csv(args.elasticities)
    minp = pd.read_csv(args.min_price)

    rec = recommend_prices(tx, elas, minp, config)
    rec.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
