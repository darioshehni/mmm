import logging
import math


def _round_to_tick(x, tick):
    """
    Round a number to the nearest multiple of a given tick.
    """
    return round(x / tick) * tick


def _floor_to_tick(x, tick):
    """
    Round a number down to the nearest multiple of a given tick.
    """
    return math.floor(x / tick) * tick


def _ceil_to_tick(x, tick):
    """
    Round a number up to the nearest multiple of a given tick.
    """
    return math.ceil(x / tick) * tick


def make_prices_valid(candidate_prices, min_price_df, last_price_df, max_move_pct, price_tick):
    """
    Adjust a set of candidate prices so they follow three simple rules.
    The rules are: respect the minimum price, stay within the allowed band around the last price, and use the price tick.
    """
    min_by_sku = {r["SKU_id"]: float(r["min_price"]) for _, r in min_price_df.iterrows()}
    last_by_sku = {r["SKU_id"]: float(r["last_price"]) for _, r in last_price_df.iterrows()}
    cap = float(max_move_pct)

    adjusted = {}
    for sku, target in candidate_prices.items():
        target = float(target)
        last = float(last_by_sku.get(sku, target))
        lower_band = last * (1.0 - cap)
        upper_band = last * (1.0 + cap)

        minimum_allowed = float(min_by_sku.get(sku, 0.0))
        if minimum_allowed > upper_band:
            logging.info(
                "SKU %s: minimum price (%.2f) is above the allowed change band [%.2f, %.2f]. "
                "Using the minimum price and ignoring the band for this SKU.",
                sku, minimum_allowed, lower_band, upper_band
            )

        lower = max(lower_band, minimum_allowed)
        upper = upper_band

        chosen = min(max(target, lower), upper)

        lower_tick = _ceil_to_tick(lower, price_tick)
        upper_tick = _floor_to_tick(upper, price_tick)

        if lower_tick > upper_tick:
            chosen_tick = max(_ceil_to_tick(lower, price_tick), _floor_to_tick(upper, price_tick))
        else:
            chosen_tick = _round_to_tick(chosen, price_tick)
            chosen_tick = min(max(chosen_tick, lower_tick), upper_tick)

        adjusted[sku] = float(chosen_tick)

    return adjusted
