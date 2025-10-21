import logging
import pandas as pd

REQUIRED_TXN_COLS = [
    "SKU_id", "model_name", "cost_unit",
    "price_sold", "promo_flag", "discount_value", "date_time"
]
REQUIRED_ELAS_COLS = ["model_name", "elasticity"]
REQUIRED_MINP_COLS = ["SKU_id", "min_price"]


def assert_required_columns(df, required):
    """
    Check that all required columns are present in a DataFrame.
    Raises a ValueError if any required columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def load_transactions(path_or_df):
    """
    Load transactions and create a 'net_price' column if needed.
    The function parses timestamps, normalizes the promo flag, and ensures non-negative net prices.
    """
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    assert_required_columns(df, REQUIRED_TXN_COLS)
    df = df.copy()

    df["date_time"] = pd.to_datetime(df["date_time"], utc=False, errors="coerce")
    if df["date_time"].isna().any():
        raise ValueError("Some date_time values could not be parsed.")

    df["net_price"] = df["price_sold"] - df["discount_value"].fillna(0.0)
    bad = df["net_price"] <= 0
    if bad.any():
        logging.warning("Found %d rows with non-positive net_price; using price_sold for those rows.", bad.sum())
        df.loc[bad, "net_price"] = df.loc[bad, "price_sold"]

    df["promo_flag"] = df["promo_flag"].fillna(0).astype(int)
    return df


def load_elasticities(path_or_df):
    """
    Load model elasticities and validate that they are negative.
    Warn if any elasticity is very close to zero, as that can drive counterintuitive results.
    """
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    assert_required_columns(df, REQUIRED_ELAS_COLS)
    df = df.copy()
    if (df["elasticity"] >= 0).any():
        raise ValueError("Elasticity values must be negative.")
    near_zero = df["elasticity"].abs() < 0.05
    if near_zero.any():
        logging.warning("Some elasticities are near zero (almost inelastic): %s",
                        df.loc[near_zero, ["model_name", "elasticity"]].to_dict("records"))
    return df


def load_min_price(path_or_df):
    """
    Load minimum prices per SKU and validate basic constraints.
    Minimum prices must be strictly greater than zero.
    """
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    assert_required_columns(df, REQUIRED_MINP_COLS)
    df = df.copy()
    if (df["min_price"] <= 0).any():
        raise ValueError("min_price must be > 0 for all SKUs.")
    return df
