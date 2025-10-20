"""
reporting.py — compact evaluation and output helpers.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def summarize_model(rec: dict) -> dict:
    """Return a tidy dict with key KPIs and any constraint flags."""
    return {
        "model_id": rec["model_id"],
        "profit_total": float(rec["profit"]),
        "revenue_total": float(rec["revenue"]),
        "units_total": float(sum(rec["units_by_sku"].values())),
        "avg_model_price": float(rec["avg_model_price"]),
        "profit_uplift_vs_last": float(rec.get("profit_uplift_vs_last", 0.0)),
        "constraint_flags": "; ".join(rec.get("constraint_flags", [])),
    }


def summarize_all(recs: list[dict]) -> pd.DataFrame:
    rows = [summarize_model(r) for r in recs]
    df = pd.DataFrame(rows)
    df = df.sort_values("profit_uplift_vs_last", ascending=False)
    return df


def ladder_table(rec: dict) -> pd.DataFrame:
    data = []
    for sku, price in rec["sku_prices"].items():
        prev = rec["prev_prices"].get(sku, None)
        units = rec["units_by_sku"].get(sku, 0.0)
        data.append({"sku_id": sku, "prev_price": prev, "reco_price": price, "units": units})
    df = pd.DataFrame(data).sort_values("reco_price")
    # Add gaps
    df["gap_vs_prev_sku"] = df["reco_price"].diff()
    return df


def write_outputs(recs: list[dict], out_dir: str) -> None:
    """Write two CSVs and a minimal run summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Detailed per-SKU recommendations
    rows = []
    for rec in recs:
        model_id = rec["model_id"]
        for sku, price in rec["sku_prices"].items():
            rows.append(
                {
                    "model_id": model_id,
                    "sku_id": sku,
                    "prev_price": rec["prev_prices"].get(sku, None),
                    "reco_price": price,
                    "expected_units": rec["units_by_sku"].get(sku, 0.0),
                    "expected_revenue": price * rec["units_by_sku"].get(sku, 0.0),
                }
            )
    pd.DataFrame(rows).to_csv(out / "recommended_prices.csv", index=False)

    # Per-model summaries
    summarize_all(recs).to_csv(out / "model_summaries.csv", index=False)

    # Minimal text summary
    lines = ["# Pricing Run Summary", ""]
    for rec in recs:
        s = summarize_model(rec)
        lines.append(
            f"- {s['model_id']}: profit={s['profit_total']:.0f}, uplift_vs_last={s['profit_uplift_vs_last']:.0f}, "
            f"avg_price={s['avg_model_price']:.2f}, flags=({s['constraint_flags']})"
        )
    (out / "RUN_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")

