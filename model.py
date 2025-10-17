import math
from .calibration import family_price_index

def predict_shares_with_outside(
    prices: dict[str, float],
    alphas: dict[str, float],
    betas: dict[str, float],
    alpha0: float,
    beta0: float,
    price_index_weights: dict[str, float]
) -> dict[str, float]:
    """
    Logit shares over SKUs + OUTSIDE. Returns shares that sum to 1 (including OUTSIDE).
    """
    utils: dict[str, float] = {k: alphas[k] - betas[k] * prices[k] for k in prices}
    p_idx = family_price_index(prices, price_index_weights)
    u0 = alpha0 - beta0 * p_idx

    max_u = max(max(utils.values()), u0)
    exp_u = {k: math.exp(v - max_u) for k, v in utils.items()}
    exp0 = math.exp(u0 - max_u)
    z = exp0 + sum(exp_u.values())

    shares = {k: exp_u[k] / z for k in prices}
    shares["OUTSIDE"] = exp0 / z
    return shares

def profit_and_revenue(
    prices: dict[str, float],
    costs: dict[str, float],
    shares_with_outside: dict[str, float],
    category_size: float
) -> tuple[float, float, dict[str, float], float]:
    """
    Returns (profit, revenue, sku_volumes, family_volume).
    family_volume = (1 - OUTSIDE) * category_size
    """
    s_out = shares_with_outside.get("OUTSIDE", 0.0)
    family_volume = (1.0 - s_out) * category_size

    sku_vols: dict[str, float] = {}
    revenue = 0.0
    profit = 0.0
    for sku, share in shares_with_outside.items():
        if sku == "OUTSIDE":
            continue
        q = share * category_size
        sku_vols[sku] = q
        revenue += prices[sku] * q
        profit += (prices[sku] - costs[sku]) * q

    return profit, revenue, sku_vols, family_volume