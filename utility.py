import math


def get_alpha(
    price: float,
    share: float,
    beta: float,
    ref_price: float,
    ref_share: float,
    ref_beta: float,
) -> float:
    """
    Compute alpha (baseline appeal) for a model relative to a reference model.

    Formula:
        alpha_k = ln(share_k / share_ref) + beta_k * price_k - beta_ref * price_ref

    Args:
        price: Model's (effective) price.
        share: Model's within-family share (0 < share < 1).
        beta:  Model's price sensitivity (> 0).
        ref_price: Reference model's price.
        ref_share: Reference model's within-family share (0 < share < 1).
        ref_beta: Reference model's price sensitivity (> 0).
    """
    if share <= 0.0 or ref_share <= 0.0:
        raise ValueError("Shares must be > 0 to compute log ratios.")
    return math.log(share / ref_share) + beta * price - ref_beta * ref_price


def get_beta(
    elasticity: float,
    price: float,
    share: float
) -> float:
    """
    Compute the price sensitivity parameter (beta) for a given model.

    Formula:
        beta = -elasticity / (price * (1 - share))

    Args:
        elasticity: Own-price elasticity from MMM (negative value, e.g. -1.2).
        price: Current or baseline price of the model.
        share: Model's within-family sales share (0 < share < 1).
    """
    if not (0.0 < share < 1.0):
        raise ValueError("Share must be between 0 and 1 (exclusive).")
    if price <= 0.0:
        raise ValueError("Price must be positive.")
    if elasticity >= 0.0:
        raise ValueError("Elasticity must be negative.")
    return -elasticity / (price * (1.0 - share))


alpha = get_alpha(
    price=1099.0, share=0.25, beta=0.0010,
    ref_price=999.0, ref_share=0.60, ref_beta=0.0013
)
beta: float = get_beta(
    elasticity=-1.2,
    price=999.0,
    share=0.60
)


print(alpha, beta)