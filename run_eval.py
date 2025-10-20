"""
run_eval.py — tiny CLI to run recommendations and sanity checks.

Usage:
  python run_eval.py recommend --config config.yaml --data_dir ./data --out ./out/2025-10-20
  python run_eval.py sanity --config config.yaml --data_dir ./data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from s_pricer import (
    load_data,
    normalize_inputs,
    validate_data,
    attach_baselines,
    load_config,
    recommend_all_models,
)
from reporting import write_outputs, summarize_all


def _read_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "sku_master": data_dir / "sku_master.csv",
        "sales": data_dir / "sales.csv",
        "elasticities": data_dir / "elasticities.csv",
        "limits": data_dir / "limits.csv",
        "last_shares": data_dir / "last_shares.csv",  # optional
    }


def cmd_recommend(args):
    cfg = load_config(args.config)
    paths = _read_paths(Path(args.data_dir))
    data = load_data(paths)
    data = normalize_inputs(data)

    errs = validate_data(data)
    if errs and any(not e.startswith("warning") for e in errs):
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errs))
    elif errs:
        print("Validation warnings:")
        for e in errs:
            if e.startswith("warning"):
                print(" -", e)

    data = attach_baselines(data, cfg)

    # Run recommendations
    recs = recommend_all_models(data, cfg)

    # Write outputs
    write_outputs(recs, args.out)

    # Print a quick leaderboard
    df = summarize_all(recs)
    print("\n=== Model leaderboard (by profit uplift vs last) ===")
    print(
        df[
            ["model_id", "profit_uplift_vs_last", "profit_total", "revenue_total", "units_total"]
        ].to_string(index=False)
    )


def cmd_sanity(args):
    cfg = load_config(args.config)
    paths = _read_paths(Path(args.data_dir))
    data = load_data(paths)
    data = normalize_inputs(data)
    errs = validate_data(data)
    if errs:
        print("Sanity report:")
        for e in errs:
            print(" -", e)
    else:
        print("All core validations passed.")

    # quick baselines preview
    from s_pricer import compute_baseline_mix, compute_ref_price_volume

    bm = compute_baseline_mix(
        data.sales, data.sku_master, cfg["baselines"]["lookback_weeks"], cfg["baselines"]["exclude_promos"]
    )
    pv = compute_ref_price_volume(
        data.sales, data.sku_master, cfg["baselines"]["lookback_weeks"], cfg["baselines"]["exclude_promos"]
    )
    print("\nBaseline mix sample:")
    print(bm.head(10).to_string(index=False))
    print("\nRef price/volume:")
    print(pv.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="S-series SKU pricing tool")
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("recommend", help="Run price recommendations")
    p1.add_argument("--config", required=True)
    p1.add_argument("--data_dir", required=True)
    p1.add_argument("--out", required=True)
    p1.set_defaults(func=cmd_recommend)

    p2 = sub.add_parser("sanity", help="Run basic data validations")
    p2.add_argument("--config", required=True)
    p2.add_argument("--data_dir", required=True)
    p2.set_defaults(func=cmd_sanity)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

