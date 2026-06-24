"""
Data loading, preprocessing, and caching for Criteo Uplift dataset.
Handles CSV -> parquet conversion, float32 downcasting, and train/test splits.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent.parent / "data"
PARQUET_PATH = DATA_DIR / "criteo_uplift.parquet"
FEATURE_COLS = [f"f{i}" for i in range(12)]
# exposure is post-treatment; conditioning on it induces collider bias — excluded
DROP_COLS = ["exposure"]


def load_data(percent10: bool = True, use_cache: bool = True) -> pd.DataFrame:
    """Load Criteo uplift dataset with parquet caching and float32 downcasting."""
    DATA_DIR.mkdir(exist_ok=True)

    cache_key = "10pct" if percent10 else "full"
    cache_path = DATA_DIR / f"criteo_uplift_{cache_key}.parquet"

    # also check for manually placed parquet
    manual = DATA_DIR / "criteo_uplift_10pct.parquet"
    if use_cache and manual.exists() and percent10:
        return pd.read_parquet(manual)
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        from sklift.datasets import fetch_criteo
        print(f"Downloading Criteo dataset (percent10={percent10})...")
        df = fetch_criteo(percent10=percent10, return_X_y_t=False)
        if not isinstance(df, pd.DataFrame):
            # fetch_criteo returns bunch — reconstruct DataFrame
            bunch = fetch_criteo(percent10=percent10)
            df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
            df["treatment"] = bunch.treatment
            df["visit"] = bunch.target
            if hasattr(bunch, "conversion"):
                df["conversion"] = bunch.conversion
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset: {e}\n"
            "Alternatively place criteo-uplift-v2.1.csv.gz in data/ and re-run."
        ) from e

    # drop post-treatment collider
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # downcast to float32 — halves memory on 14M rows
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    int_cols = df.select_dtypes(include="int64").columns
    df[int_cols] = df[int_cols].astype("int8")

    df.to_parquet(cache_path, index=False)
    print(f"Cached to {cache_path}  shape={df.shape}")
    return df


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """One-time CSV -> parquet conversion for the full 297 MB dataset."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    out = DATA_DIR / "criteo_uplift_full.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {out}  shape={df.shape}")
    return df


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    label: str = "visit",
):
    """Stratified split preserving treatment ratio. Returns (train, val, test)."""
    strat = df["treatment"].astype(str) + "_" + df[label].astype(str)
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=strat, random_state=seed
    )
    strat2 = (
        train_val["treatment"].astype(str) + "_" + train_val[label].astype(str)
    )
    val_frac = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=strat2, random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def get_Xyt(df: pd.DataFrame, label: str = "visit"):
    """Return feature matrix, outcome, treatment arrays as numpy float32."""
    X = df[FEATURE_COLS].values.astype("float32")
    y = df[label].values.astype("float32")
    t = df["treatment"].values.astype("float32")
    return X, y, t
