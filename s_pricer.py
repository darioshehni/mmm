"""
s_pricer.py — lean price recommendation engine for Samsung S-series SKUs.

Design goals:
- Clear, modular functions (load → validate → baselines → demand → split → optimize).
- Explainable, tried-and-tested math (isoelastic demand, historical-mix splitter).
- Guardrails: bounds, ladders, smoothing.
- Minimal deps: pandas, numpy.

Notes:
- Type hints use built-in generics (dict[str, float]) — no typing.List/Dict imports.
- Currency-agnostic (treats prices as numeric).
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------
# Data loading / normalization
# -------------------------------

REQUIRED_COLUMNS = {
    "sku_master": ["model_id", "sku_id", "storage_gb", "cost"],
    "sales": ["date", "sku_id", "price", "units"],
    "elasticities": ["model_id", "elasticity", "ref_price", "ref_volume"],
    "limits": ["sku_id", "min_price", "max_price", "last_price"],
}


@dataclass
class DataBundle:
    sku_master: pd.DataFrame
    sales: pd.DataFrame
    elasticities: pd.DataFrame
    limits: pd.DataFrame
    last_shares: pd.DataFrame | None = None  # optional, columns: model_id, sku_id, last_share


def load_data(paths: dict[str, str | Path]) -> DataBundle:
    """Load CSVs into dataframes with minimal fuss. Hard-fails on missing files."""

    def _load(name: str) -> pd.DataFrame:
        p = Path(paths[name])
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
        df = pd.read_csv(p)
        missing = [c for c in REQUIRED_COLUMNS.get(name, []) if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns: {missing}")
        return df

    sku_master = _load("sku_master")
    sales = _load("sales")
    elasticities = _load("elasticities")
    limits = _load("limits")
    last_shares = None
    if "last_shares" in paths and Path(paths["last_shares"]).exists():
        last_shares = pd.read_csv(paths["last_shares"])
        if not {"model_id", "sku_id", "last_share"}.issubset(last_shares.columns):
            raise ValueError("last_shares must include columns: model_id, sku_id, last_share")

    return DataBundle(sku_master, sales, elasticities, limits, last_shares)


def normalize_inputs(d: DataBundle) -> DataBundle:
    """Light normalization: ensure dtypes and keys align; parse dates."""
    sales = d.sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    # Ensure key casing and numeric types
    for col in ("price", "units"):
        sales[col] = pd.to_numeric(sales[col], errors="coerce")
    if "promo_flag" in sales.columns:
        sales["promo_flag"] = sales["promo_flag"].fillna(0).astype(int)

    sku_master = d.sku_master.copy()
    sku_master["storage_gb"] = pd.to_numeric(sku_master["storage_gb"], errors="coerce")
    sku_master["cost"] = pd.to_numeric(sku_master["cost"], errors="coerce")

    elasticities = d.elasticities.copy()
    for col in ("elasticity", "ref_price", "ref_volume"):
        elasticities[col] = pd.to_numeric(elasticities[col], errors="coerce")

    limits = d.limits.copy()
    for col in ("min_price", "max_price", "last_price"):
        limits[col] = pd.to_numeric(limits[col], errors="coerce")

    last_shares = d.last_shares.copy() if d.last_shares is not None else None
    if last_shares is not None:
        last_shares["last_share"] = pd.to_numeric(last_shares["last_share"], errors="coerce")

    return DataBundle(sku_master, sales, elasticities, limits, last_shares)


# -------------------------------
# Validation
# -------------------------------


def validate_data(d: DataBundle) -> list[str]:
    """Return a list of human-readable errors; empty if OK."""
    errors: list[str] = []

    # Required keys
    for name, cols in REQUIRED_COLUMNS.items():
        df = getattr(d, name)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            errors.append(f"{name}: missing columns {missing}")

    # Duplicates
    if d.sku_master.duplicated(["sku_id"]).any():
        errors.append("sku_master: duplicate sku_id found")
    if d.elasticities.duplicated(["model_id"]).any():
        errors.append("elasticities: duplicate model_id found")
    if d.limits.duplicated(["sku_id"]).any():
        errors.append("limits: duplicate sku_id in limits")

    # Orphan SKUs
    sales_skus = set(d.sales["sku_id"].unique())
    master_skus = set(d.sku_master["sku_id"].unique())
    if not sales_skus.issubset(master_skus):
        missing = list(sales_skus - master_skus)[:10]
        errors.append(f"sales contains SKU(s) missing from sku_master: sample {missing}")

    # Price >= 0, cost >= 0
    if (d.sku_master["cost"] < 0).any():
        errors.append("sku_master: negative costs present")
    if (d.sales["price"] < 0).any():
        errors.append("sales: negative prices present")

    # Price < cost warning (not hard error)
    joined = d.sales.merge(d.sku_master[["sku_id", "cost"]], on="sku_id", how="left")
    if (joined["price"] < joined["cost"]).any():
        errors.append("warning: some historical prices below cost")

    # Recency check: require at least some data in recent 26 weeks
    if d.sales["date"].max() < (pd.Timestamp.today() - pd.Timedelta(weeks=26)):
        errors.append("sales: last transaction older than 26 weeks; baselines may be stale")

    return errors


# -------------------------------
# Baselines
# -------------------------------


def compute_baseline_mix(
    sales: pd.DataFrame, sku_master: pd.DataFrame, lookback_weeks: int, exclude_promos: bool
) -> pd.DataFrame:
    """
    Compute baseline SKU mix per model over the recent stable period.
    Returns columns: model_id, sku_id, baseline_share.
    """
    cutoff = sales["date"].max() - pd.Timedelta(weeks=lookback_weeks)
    df = sales[sales["date"] >= cutoff].copy()
    if exclude_promos and "promo_flag" in df.columns:
        df = df[df["promo_flag"] == 0].copy()

    df = df.merge(sku_master[["sku_id", "model_id"]], on="sku_id", how="left")
    mix = (
        df.groupby(["model_id", "sku_id"], as_index=False)["units"].sum()
        .rename(columns={"units": "units_sum"})
    )
    model_tot = (
        mix.groupby("model_id", as_index=False)["units_sum"].sum().rename(columns={"units_sum": "model_units"})
    )
    mix = mix.merge(model_tot, on="model_id", how="left")
    mix["baseline_share"] = np.where(
        mix["model_units"] > 0, mix["units_sum"] / mix["model_units"], 0.0
    )
    return mix[["model_id", "sku_id", "baseline_share"]]


def compute_ref_price_volume(
    sales: pd.DataFrame, sku_master: pd.DataFrame, lookback_weeks: int, exclude_promos: bool
) -> pd.DataFrame:
    """
    Compute reference average price per model and reference volume (units per period).
    Returns columns: model_id, ref_price, ref_volume.
    """
    cutoff = sales["date"].max() - pd.Timedelta(weeks=lookback_weeks)
    df = sales[sales["date"] >= cutoff].copy()
    if exclude_promos and "promo_flag" in df.columns:
        df = df[df["promo_flag"] == 0].copy()

    df = df.merge(sku_master[["sku_id", "model_id"]], on="sku_id", how="left")
    # Approximate model average price: simple average of SKU average prices weighted by units
    sku_agg = (
        df.groupby(["model_id", "sku_id"], as_index=False)
        .agg(units=("units", "sum"), price=("price", "mean"))
    )
    model_price = (
        sku_agg.assign(weighted_price=lambda x: x["price"] * x["units"])  # type: ignore[index]
        .groupby("model_id", as_index=False)
        .agg(
            ref_price=(
                "weighted_price",
                lambda s: s.sum() / max(sku_agg.loc[s.index, "units"].sum(), 1e-9),
            ),
            ref_volume=("units", "sum"),
        )
    )
    return model_price[["model_id", "ref_price", "ref_volume"]]


# -------------------------------
# Demand (model-level)
# -------------------------------


def isoelastic_volume(price: float, ref_price: float, ref_volume: float, elasticity: float) -> float:
    """
    Classic isoelastic demand pivoted at (ref_price, ref_volume):
    V(p) = V_ref * (p / p_ref) ** elasticity
    elasticity should be negative for normal goods.
    """
    if ref_price <= 0:
        return max(ref_volume, 0.0)
    ratio = max(price, 1e-9) / ref_price
    return max(ref_volume * (ratio ** elasticity), 0.0)


def model_avg_price(sku_prices: dict[str, float], method: str = "simple") -> float:
    """Aggregate SKU prices to a model-level price. MVP: simple average."""
    if not sku_prices:
        return 0.0
    if method == "simple":
        return float(np.mean(list(sku_prices.values())))
    # Placeholder for alternatives (e.g., share-weighted)
    return float(np.mean(list(sku_prices.values())))


# -------------------------------
# SKU split (Level 1 — simple & robust)
# -------------------------------


def sku_split_level1(
    sku_prices: dict[str, float],
    baseline_mix: dict[str, float],
    last_share: dict[str, float] | None,
    k: float,
    min_share: float,
    max_change: float,
) -> dict[str, float]:
    """
    Historical-mix anchored, price-gap penalized share splitter.
    Scores = log(baseline_mix) - k * (price - min_price)
    Then softmax → shares; apply floors and optional per-run change cap vs last_share.
    """
    if not sku_prices:
        return {}

    min_p = min(sku_prices.values())
    premiums = {sku: max(0.0, sku_prices[sku] - min_p) for sku in sku_prices}

    eps = 1e-12
    scores: dict[str, float] = {
        sku: math.log(max(baseline_mix.get(sku, eps), eps)) - k * premiums[sku]
        for sku in sku_prices
    }
    # Softmax
    max_score = max(scores.values())
    exps = {sku: math.exp(scores[sku] - max_score) for sku in scores}
    total = sum(exps.values()) or 1.0
    shares = {sku: exps[sku] / total for sku in exps}

    # Floor & renormalize
    def apply_floor(sh: dict[str, float]) -> dict[str, float]:
        skus = list(sh.keys())
        clamped = {sku: max(val, min_share) for sku, val in sh.items()}
        s = sum(clamped.values())
        return {sku: val / s for sku, val in clamped.items()}

    shares = apply_floor(shares)

    # Optional per-run cap vs last share
    if last_share:
        capped: dict[str, float] = {}
        for sku, curr in shares.items():
            prev = last_share.get(sku, curr)
            delta = curr - prev
            if delta > max_change:
                capped[sku] = prev + max_change
            elif delta < -max_change:
                capped[sku] = prev - max_change
            else:
                capped[sku] = curr
        s = sum(capped.values()) or 1.0
        shares = {sku: v / s for sku, v in capped.items()}

    return shares


# -------------------------------
# Constraints (bounds, ladders, smoothing)
# -------------------------------


def apply_price_bounds(sku_prices: dict[str, float], limits_df: pd.DataFrame) -> dict[str, float]:
    """Clamp prices to [min_price, max_price] per SKU."""
    lim = limits_df.set_index("sku_id")[
        ["min_price", "max_price"]
    ].to_dict(orient="index")
    bounded: dict[str, float] = {}
    for sku, p in sku_prices.items():
        if sku in lim:
            lo = lim[sku]["min_price"]
            hi = lim[sku]["max_price"]
            bounded[sku] = float(min(max(p, lo), hi))
        else:
            bounded[sku] = float(p)
    return bounded


def enforce_ladder(
    sku_prices: dict[str, float],
    ladder_gap_by_step: float,
    storage_map: dict[str, int],
) -> dict[str, float]:
    """
    Enforce non-decreasing prices by storage order, with minimum gap per step.
    Greedy forward pass on SKUs sorted by storage_gb.
    """
    ordered = sorted(sku_prices.items(), key=lambda kv: storage_map.get(kv[0], 0))
    adjusted = dict(ordered)
    for i in range(1, len(ordered)):
        prev_sku, prev_price = ordered[i - 1][0], adjusted[ordered[i - 1][0]]
        sku, price = ordered[i][0], adjusted[ordered[i][0]]
        min_required = prev_price + ladder_gap_by_step
        if price < min_required:
            adjusted[sku] = min_required
    return adjusted


def cap_price_moves(
    sku_prices: dict[str, float],
    last_prices: dict[str, float],
    max_pct_move: float,
) -> dict[str, float]:
    """
    Cap each price to be within ±max_pct_move of last price.
    """
    capped: dict[str, float] = {}
    for sku, p in sku_prices.items():
        last = last_prices.get(sku, p)
        lo = last * (1.0 - max_pct_move)
        hi = last * (1.0 + max_pct_move)
        capped[sku] = float(min(max(p, lo), hi))
    return capped


def validate_constraints(
    sku_prices: dict[str, float],
    limits_df: pd.DataFrame,
    ladder_gap_by_step: float,
    storage_map: dict[str, int],
) -> list[str]:
    """Return list of violations (empty if all good)."""
    issues: list[str] = []

    # Bounds
    lim = limits_df.set_index("sku_id")[
        ["min_price", "max_price"]
    ].to_dict(orient="index")
    for sku, p in sku_prices.items():
        if sku in lim:
            lo = lim[sku]["min_price"]
            hi = lim[sku]["max_price"]
            if p < lo - 1e-6 or p > hi + 1e-6:
                issues.append(f"{sku}: price {p:.2f} outside [{lo:.2f}, {hi:.2f}]")

    # Ladder
    ordered = sorted(sku_prices.items(), key=lambda kv: storage_map.get(kv[0], 0))
    for i in range(1, len(ordered)):
        prev_sku, prev_price = ordered[i - 1]
        sku, price = ordered[i]
        if price < prev_price + ladder_gap_by_step - 1e-6:
            issues.append(
                f"ladder: {sku} < {prev_sku} + gap ({price:.2f} < {prev_price + ladder_gap_by_step:.2f})"
            )

    return issues


# -------------------------------
# Objective & prediction
# -------------------------------


def predict_sku_units(
    model_id: str,
    sku_prices: dict[str, float],
    refs: dict,  # expects keys: ref_price, ref_volume
    elasticity: float,
    split_params: dict,
    baseline_mix_row: pd.DataFrame,
    last_share_row: pd.DataFrame | None,
) -> dict[str, float]:
    """
    Predict SKU units:
    1) avg model price
    2) total units via isoelastic
    3) shares via splitter
    """
    avg_price = model_avg_price(sku_prices, method="simple")
    total_units = isoelastic_volume(avg_price, refs["ref_price"], refs["ref_volume"], elasticity)

    # Prepare per-SKU baselines and last shares
    baseline = {r["sku_id"]: r["baseline_share"] for _, r in baseline_mix_row.iterrows()}
    last_share = None
    if last_share_row is not None and not last_share_row.empty:
        last_share = {r["sku_id"]: r["last_share"] for _, r in last_share_row.iterrows()}

    shares = sku_split_level1(
        sku_prices=sku_prices,
        baseline_mix=baseline,
        last_share=last_share,
        k=split_params["k"],
        min_share=split_params["min_share"],
        max_change=split_params["max_change"],
    )
    return {sku: float(total_units * shares.get(sku, 0.0)) for sku in sku_prices}


def compute_profit(
    sku_prices: dict[str, float],
    sku_units: dict[str, float],
    costs: dict[str, float],
) -> float:
    """Σ (price - cost) * units."""
    profit = 0.0
    for sku, units in sku_units.items():
        price = sku_prices.get(sku, 0.0)
        cost = costs.get(sku, 0.0)
        profit += max(price - cost, 0.0) * units
    return float(profit)


def evaluate_solution(model_id: str, context: dict, sku_prices: dict[str, float]) -> dict:
    """
    Evaluate a price vector for a model. Returns dict with:
    - units_by_sku, profit, revenue, avg_model_price, shares, constraint_flags
    """
    sku_list = context["sku_list"]  # list of sku_ids for the model
    # clamp to known SKUs
    sku_prices = {sku: sku_prices[sku] for sku in sku_list}

    # Constraints pass (validate post-hoc only; solver applies proactively)
    issues = validate_constraints(
        sku_prices=sku_prices,
        limits_df=context["limits_df"],
        ladder_gap_by_step=context["cfg"]["constraints"]["ladder_gap_by_storage_step"],
        storage_map=context["storage_map"],
    )

    units_by_sku = predict_sku_units(
        model_id=model_id,
        sku_prices=sku_prices,
        refs=context["refs"],
        elasticity=context["elasticity"],
        split_params=context["cfg"]["splitter"],
        baseline_mix_row=context["baseline_mix_row"],
        last_share_row=context["last_share_row"],
    )

    costs = context["costs"]
    profit = compute_profit(sku_prices, units_by_sku, costs)
    revenue = sum(sku_prices[sku] * units_by_sku[sku] for sku in sku_list)
    avg_price = model_avg_price(sku_prices)
    return {
        "model_id": model_id,
        "sku_prices": sku_prices,
        "units_by_sku": units_by_sku,
        "profit": profit,
        "revenue": revenue,
        "avg_model_price": avg_price,
        "constraint_flags": issues,
    }


# -------------------------------
# Solvers (coordinate descent + grid)
# -------------------------------


def _apply_all_constraints(candidate: dict[str, float], context: dict) -> dict[str, float]:
    """Helper: apply bounds → ladder → move caps, in that order."""
    candidate = apply_price_bounds(candidate, context["limits_df"])
    candidate = enforce_ladder(
        candidate, context["cfg"]["constraints"]["ladder_gap_by_storage_step"], context["storage_map"]
    )
    candidate = cap_price_moves(
        candidate, context["last_prices"], context["cfg"]["constraints"]["max_price_move_pct"]
    )
    # final bounds after ladder/move
    candidate = apply_price_bounds(candidate, context["limits_df"])
    return candidate


def coordinate_descent(
    model_id: str,
    start_prices: dict[str, float],
    step: float,
    eval_fn,
    context: dict,
    max_iter: int,
    patience: int,
) -> dict:
    """
    Greedy CD over SKUs. For each SKU, try +/- step; accept if profit improves.
    Applies constraints on every candidate move.
    """
    current = _apply_all_constraints(start_prices.copy(), context)
    best_eval = eval_fn(model_id, context, current)
    best_profit = best_eval["profit"]
    no_improve_rounds = 0

    sku_ids = list(current.keys())

    for _ in range(max_iter):
        improved = False
        for sku in sku_ids:
            for direction in (+1, -1):
                candidate = current.copy()
                candidate[sku] = candidate[sku] + direction * step
                candidate = _apply_all_constraints(candidate, context)
                ev = eval_fn(model_id, context, candidate)
                if ev["profit"] > best_profit + 1e-6:
                    current = candidate
                    best_profit = ev["profit"]
                    best_eval = ev
                    improved = True
        if not improved:
            no_improve_rounds += 1
            if no_improve_rounds >= patience:
                break
        else:
            no_improve_rounds = 0

    return best_eval


def grid_search(
    model_id: str,
    price_grid_by_sku: dict[str, list[float]],
    eval_fn,
    context: dict,
) -> dict:
    """
    Cartesian product over per-SKU grids (use for small SKU counts).
    Applies constraints to each candidate before evaluating.
    """
    from itertools import product

    skus = list(price_grid_by_sku.keys())
    best = None
    best_profit = -1e18

    for combo in product(*[price_grid_by_sku[k] for k in skus]):
        candidate = {sku: price for sku, price in zip(skus, combo)}
        candidate = _apply_all_constraints(candidate, context)
        ev = eval_fn(model_id, context, candidate)
        if ev["profit"] > best_profit + 1e-9:
            best = ev
            best_profit = ev["profit"]

    return best


# -------------------------------
# Orchestration
# -------------------------------


def _build_storage_map(sku_master: pd.DataFrame, model_id: str) -> dict[str, int]:
    row = sku_master[sku_master["model_id"] == model_id][["sku_id", "storage_gb"]].copy()
    row = row.sort_values("storage_gb")
    return {r.sku_id: i for i, r in enumerate(row.itertuples())}


def recommend_for_model(model_id: str, d: DataBundle, cfg: dict) -> dict:
    """End-to-end for a single model_id; returns evaluation dict augmented with diagnostics/metadata."""
    model_skus = d.sku_master[d.sku_master["model_id"] == model_id].copy()
    sku_list = list(model_skus["sku_id"])
    costs = {r.sku_id: float(r.cost) for r in model_skus.itertuples()}
    storage_map = _build_storage_map(d.sku_master, model_id)

    # Limits and last prices
    limits_df = d.limits[d.limits["sku_id"].isin(sku_list)].copy()
    last_prices = {r.sku_id: float(r.last_price) for r in limits_df.itertuples()}

    # Baselines
    base_mix_row = d.baseline_mix[d.baseline_mix["model_id"] == model_id]  # set by caller
    last_share_row = None
    if d.last_shares is not None:
        last_share_row = d.last_shares[d.last_shares["model_id"] == model_id]

    # Elasticity + refs
    el_row = d.elasticities[d.elasticities["model_id"] == model_id].iloc[0]
    refs = {"ref_price": float(el_row.ref_price), "ref_volume": float(el_row.ref_volume)}
    elasticity = float(el_row.elasticity)

    # Start prices = last prices (safe)
    start_prices = last_prices.copy()

    context = {
        "cfg": cfg,
        "limits_df": limits_df,
        "storage_map": storage_map,
        "last_prices": last_prices,
        "sku_list": sku_list,
        "refs": refs,
        "elasticity": elasticity,
        "costs": costs,
        "baseline_mix_row": base_mix_row,
        "last_share_row": last_share_row,
    }

    method = cfg["optimizer"]["method"]
    if method == "coordinate_descent":
        best_eval = coordinate_descent(
            model_id=model_id,
            start_prices=start_prices,
            step=cfg["optimizer"]["step"],
            eval_fn=evaluate_solution,
            context=context,
            max_iter=cfg["optimizer"]["max_iter"],
            patience=cfg["optimizer"]["patience"],
        )
    elif method == "grid":
        # Build per-SKU grids using limits + step
        step = cfg["optimizer"]["step"]
        price_grid_by_sku: dict[str, list[float]] = {}
        for sku in sku_list:
            row = limits_df[limits_df["sku_id"] == sku].iloc[0]
            lo = max(row.min_price, last_prices[sku] * (1.0 - cfg["constraints"]["max_price_move_pct"]))
            hi = min(row.max_price, last_prices[sku] * (1.0 + cfg["constraints"]["max_price_move_pct"]))
            grid = list(np.arange(lo, hi + 0.5 * step, step))
            price_grid_by_sku[sku] = grid
        best_eval = grid_search(model_id, price_grid_by_sku, evaluate_solution, context)
    else:
        raise ValueError(f"Unknown optimizer method: {method}")

    # Attach diagnostics/metadata
    best_eval["prev_prices"] = last_prices
    best_eval["profit_uplift_vs_last"] = best_eval["profit"] - evaluate_solution(
        model_id, context, last_prices
    )["profit"]

    return best_eval


def recommend_all_models(d: DataBundle, cfg: dict) -> list[dict]:
    """Run recommendations for all models in data bundle."""
    results: list[dict] = []
    models = sorted(d.sku_master["model_id"].unique())
    for model_id in models:
        res = recommend_for_model(model_id, d, cfg)
        results.append(res)
    return results


# -------------------------------
# Small helpers to attach baselines to the bundle
# -------------------------------


def attach_baselines(d: DataBundle, cfg: dict) -> DataBundle:
    """Compute or attach baseline mix and ref price/volume into the bundle (adds attributes)."""
    # Compute if not provided
    base_mix = compute_baseline_mix(
        sales=d.sales,
        sku_master=d.sku_master,
        lookback_weeks=cfg["baselines"]["lookback_weeks"],
        exclude_promos=cfg["baselines"]["exclude_promos"],
    )
    ref_pv = compute_ref_price_volume(
        sales=d.sales,
        sku_master=d.sku_master,
        lookback_weeks=cfg["baselines"]["lookback_weeks"],
        exclude_promos=cfg["baselines"]["exclude_promos"],
    )
    # Merge ref_pv into elasticities if missing
    el = d.elasticities.merge(ref_pv, on="model_id", suffixes=("", "_computed"), how="left")
    el["ref_price"] = np.where(el["ref_price"].isna(), el["ref_price_computed"], el["ref_price"])  # type: ignore[index]
    el["ref_volume"] = np.where(
        el["ref_volume"].isna(), el["ref_volume_computed"], el["ref_volume"]
    )  # type: ignore[index]
    d.elasticities = el[["model_id", "elasticity", "ref_price", "ref_volume"]].copy()

    # Stash baseline mix on the bundle for quick access
    d.baseline_mix = base_mix  # type: ignore[attr-defined]
    return d


# -------------------------------
# Config helpers
# -------------------------------


def load_config(path: str | Path) -> dict:
    """Load a small YAML or JSON config (JSON supported out of the box)."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return json.loads(Path(path).read_text())
    # Tiny YAML loader without extra deps: accept a limited subset
    try:
        import yaml  # optional dependency

        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(
            "Failed to load YAML config. Install pyyaml or provide JSON config.\n"
            f"Underlying error: {e}"
        )

