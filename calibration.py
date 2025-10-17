import math

def baseline_family_shares(units: dict[str, float]) -> dict[str, float]:
    total = sum(units.values())
    if total <= 0.0:
        raise ValueError("Total baseline units must be positive.")
    return {k: v / total for k, v in units.items()}

def beta_from_elasticity(elasticity: float, price: float, share: float) -> float:
    """
    beta = -elasticity / (price * (1 - share))
    """
    if not (0.0 < share < 1.0):
        raise ValueError("Share must be in (0,1).")
    if price <= 0.0:
        raise ValueError("Price must be positive.")
    return -elasticity / (price * (1.0 - share))

def alphas_from_baseline(
    prices: dict[str, float],
    shares: dict[str, float],
    betas: dict[str, float],
    reference_sku: str
) -> dict[str, float]:
    """
    alpha_k = ln(share_k/share_ref) + beta_k*price_k - beta_ref*price_ref
    with alpha_ref = 0.
    """
    if reference_sku not in prices:
        raise ValueError("Reference SKU not found in prices.")
    if shares[reference_sku] <= 0.0:
        raise ValueError("Reference share must be > 0.")
    alphas: dict[str, float] = {reference_sku: 0.0}
    for sku in prices:
        if sku == reference_sku:
            continue
        if shares[sku] <= 0.0:
            raise ValueError(f"Share must be > 0 for {sku}.")
        alphas[sku] = (
            math.log(shares[sku] / shares[reference_sku])
            + betas[sku] * prices[sku]
            - betas[reference_sku] * prices[reference_sku]
        )
    return alphas

def family_price_index(prices: dict[str, float], weights: dict[str, float]) -> float:
    """
    Weighted average price (weights should sum to 1; use baseline shares as weights).
    """
    return sum(prices[k] * weights[k] for k in prices)

def calibrate_outside(
    alphas: dict[str, float],
    betas: dict[str, float],
    baseline_prices: dict[str, float],
    baseline_family_share: float,
    family_elasticity: float,
    price_index_baseline: float
) -> tuple[float, float]:
    """
    Calibrate outside option parameters (alpha0, beta0) so that:
      - beta0 matches family-level MMM elasticity at baseline
      - alpha0 reproduces the baseline outside share

    Returns:
      (alpha0, beta0)
    """
    if not (0.0 < baseline_family_share < 1.0):
        raise ValueError("baseline_family_share must be in (0,1).")
    if price_index_baseline <= 0.0:
        raise ValueError("price_index_baseline must be positive.")
    if family_elasticity >= 0.0:
        raise ValueError("family_elasticity should be negative.")

    beta0 = -family_elasticity / (price_index_baseline * (1.0 - baseline_family_share))

    denom = 0.0
    for sku, a in alphas.items():
        denom += math.exp(a - betas[sku] * baseline_prices[sku])

    s0 = 1.0 - baseline_family_share
    alpha0 = beta0 * price_index_baseline + math.log(s0 / (1.0 - s0)) + math.log(denom)
    return alpha0, beta0