from copy import deepcopy
from constraints import make_prices_valid
from demand_shares import (
    typical_model_price,
    split_between_models,
    split_within_model,
    compute_demand_level,
)


def build_profit_objective(cost_by_sku,
                           category_elasticity,
                           baseline_model_share_df,
                           baseline_sku_share_df,
                           k_model,
                           k_sku,
                           min_share_sku):
    """
    Build and return a profit function that evaluates a full price vector across all SKUs.
    The function models cross-model switching, within-model switching, and a category-level demand response.
    """
    # Precompute which SKUs belong to which model for faster evaluation
    skus_by_model = {}
    for _, row in baseline_sku_share_df[["model_name", "SKU_id"]].drop_duplicates().iterrows():
        skus_by_model.setdefault(row["model_name"], []).append(row["SKU_id"])

    def profit(all_sku_prices):
        """
        Compute relative profit for a full set of SKU prices.
        Absolute units are not needed; only relative comparisons matter for selecting prices.
        """
        # 1) Representative price per model (using baseline SKU shares)
        model_rep_price = {}
        for model, model_skus in skus_by_model.items():
            prices_for_model = {s: all_sku_prices[s] for s in model_skus if s in all_sku_prices}
            if prices_for_model:
                baseline_shares = baseline_sku_share_df[baseline_sku_share_df["model_name"] == model][["SKU_id", "base_sku_share"]]
                model_rep_price[model] = typical_model_price(prices_for_model, baseline_shares)

        # 2) Model shares (cross-model cannibalization)
        model_share = split_between_models(model_rep_price, baseline_model_share_df, k_model)

        # 3) SKU shares within each model (within-model cannibalization)
        sku_share_by_model = {}
        for model, model_skus in skus_by_model.items():
            prices_for_model = {s: all_sku_prices[s] for s in model_skus if s in all_sku_prices}
            sku_share_by_model[model] = split_within_model(model, prices_for_model, baseline_sku_share_df, k_sku, min_share_sku)

        # 4) Category average price based on predicted model shares (clear, one-line definition)
        category_average_price = sum(model_share.get(model, 0.0) * model_rep_price.get(model, 0.0) for model in model_rep_price)

        # 5) Demand level multiplier from category average price and elasticity
        demand_level = compute_demand_level(category_average_price, category_elasticity)

        # 6) Profit as the sum of margins times allocated demand
        profit_value = 0.0
        for model, model_skus in skus_by_model.items():
            model_weight = float(model_share.get(model, 0.0))
            if model_weight <= 0:
                continue
            for sku in model_skus:
                if sku not in all_sku_prices:
                    continue
                sku_weight = float(sku_share_by_model[model].get(sku, 0.0))
                margin = float(all_sku_prices[sku] - cost_by_sku.get(sku, 0.0))
                profit_value += margin * model_weight * sku_weight * demand_level

        return float(profit_value)

    return profit


def coordinate_descent(models,
                       skus_by_model,
                       start_prices,
                       evaluate_profit_fn,
                       min_price_df,
                       last_price_df,
                       max_move_pct,
                       price_tick,
                       step,
                       max_iter):
    """
    Perform a joint coordinate descent over all models and SKUs with fixed-size steps.
    At each step the function tests raising or lowering a single SKU price, keeps improvements, and stops when no changes help.
    """
    prices = make_prices_valid(deepcopy(start_prices), min_price_df, last_price_df, max_move_pct, price_tick)
    best_profit = evaluate_profit_fn(prices)

    iteration = 0
    improved = True
    while improved and iteration < int(max_iter):
        improved = False
        iteration += 1

        for model in models:
            for sku in skus_by_model[model]:
                current_price = prices[sku]

                # Try moving up by one step
                try_up = deepcopy(prices)
                try_up[sku] = current_price + step
                try_up = make_prices_valid(try_up, min_price_df, last_price_df, max_move_pct, price_tick)
                profit_up = evaluate_profit_fn(try_up)

                # Try moving down by one step
                try_down = deepcopy(prices)
                try_down[sku] = current_price - step
                try_down = make_prices_valid(try_down, min_price_df, last_price_df, max_move_pct, price_tick)
                profit_down = evaluate_profit_fn(try_down)

                # Keep the best of current, up, down
                candidate_prices = [(best_profit, prices), (profit_up, try_up), (profit_down, try_down)]
                candidate_prices.sort(key=lambda t: t[0], reverse=True)
                top_profit, top_prices = candidate_prices[0]

                if top_profit > best_profit + 1e-9:
                    prices = top_prices
                    best_profit = top_profit
                    improved = True

    return prices
