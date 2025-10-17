from src.calibration import (
    baseline_family_shares, beta_from_elasticity, alphas_from_baseline,
    family_price_index, calibrate_outside
)
from src.optimizer import best_price_grid_with_outside

# ---------- Inputs (replace with your real data) ----------
baseline_prices = {"S25": 999.0, "S25_Plus": 1099.0, "S25_Ultra": 1199.0}
baseline_units  = {"S25": 70_000.0, "S25_Plus": 25_000.0, "S25_Ultra": 15_000.0}
unit_costs      = {"S25": 520.0,   "S25_Plus": 560.0,    "S25_Ultra": 640.0}

mmm_elasticities_sku = {"S25": -1.2, "S25_Plus": -1.0, "S25_Ultra": -0.8}
mmm_elasticity_family = -1.1
baseline_family_share = 0.42
category_size = 300_000.0

candidate_grids = {
    "S25": [949.0, 999.0, 1049.0],
    "S25_Plus": [1049.0, 1099.0, 1149.0],
    "S25_Ultra": [1149.0, 1199.0, 1249.0, 1299.0],
}

# ---------- Calibration ----------
shares0 = baseline_family_shares(baseline_units)

betas = {
    sku: beta_from_elasticity(mmm_elasticities_sku[sku], baseline_prices[sku], shares0[sku])
    for sku in baseline_prices
}

reference_sku = "S25"
alphas = alphas_from_baseline(baseline_prices, shares0, betas, reference_sku)

price_index_weights = shares0
pidx0 = family_price_index(baseline_prices, price_index_weights)
alpha0, beta0 = calibrate_outside(
    alphas, betas, baseline_prices,
    baseline_family_share, mmm_elasticity_family, pidx0
)

# ---------- Optimize ----------
best_prices, best_profit, best_vols, best_family_volume = best_price_grid_with_outside(
    candidate_grids, alphas, betas, unit_costs,
    alpha0, beta0, price_index_weights,
    category_size, objective="profit"
)

print("Best prices:", best_prices)
print("Max profit:", round(best_profit))
print("SKU volumes (units):", {k: int(v) for k, v in best_vols.items()})
print("Family volume (units):", int(best_family_volume))