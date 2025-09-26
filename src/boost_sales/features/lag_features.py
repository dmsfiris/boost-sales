# src/sales_forecast/features/lag_features.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")


def add_lag_roll_features(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("store_id", "item_id"),
    target_col: str = "sales",
    lag_steps: Iterable[int] = (1, 7, 14, 28),
    roll_windows: Iterable[int] = (7, 28),
    roll_use_target: bool = True,
    roll_min_periods: Optional[int] = None,  # kept for API parity; not used directly
    include_std: bool = True,
    include_price_roll: bool = True,
    price_col: str = "price",
    date_col: str = "date",
) -> pd.DataFrame:
    """Add legacy-compatible lag and rolling features; returns a new DataFrame."""
    if not isinstance(group_cols, (list, tuple)):
        group_cols = tuple(group_cols)

    out = df.copy()
    _ensure_datetime(out, date_col)

    g = out.groupby(list(group_cols), sort=False, observed=True)

    # Lags for target
    for k in lag_steps:
        out[f"{target_col}_lag_{k}"] = g[target_col].shift(k)

    # Rolling over shifted target (exclude current row)
    if roll_use_target:
        shifted_target = g[target_col].shift(1)
        for w in roll_windows:
            out[f"{target_col}_roll_mean_{w}"] = shifted_target.rolling(window=w, min_periods=1).mean()
            if include_std:
                out[f"{target_col}_roll_std_{w}"] = shifted_target.rolling(window=w, min_periods=2).std()

    # Rolling price (mean) over shifted price
    if include_price_roll and price_col in out.columns:
        shifted_price = g[price_col].shift(1)
        for w in roll_windows:
            out[f"{price_col}_roll_{w}"] = shifted_price.rolling(window=w, min_periods=1).mean()

    return out


# Backwards-compat alias
def add_lags_and_rollups(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("store_id", "item_id"),
    target_col: str = "sales",
    lag_steps: Iterable[int] = (1, 7, 14, 28),
    roll_windows: Iterable[int] = (7, 28),
    roll_use_target: bool = True,
    roll_min_periods: Optional[int] = None,
    include_std: bool = True,
    include_price_roll: bool = True,
    price_col: str = "price",
    date_col: str = "date",
) -> pd.DataFrame:
    return add_lag_roll_features(
        df,
        group_cols=group_cols,
        target_col=target_col,
        lag_steps=lag_steps,
        roll_windows=roll_windows,
        roll_use_target=roll_use_target,
        roll_min_periods=roll_min_periods,
        include_std=include_std,
        include_price_roll=include_price_roll,
        price_col=price_col,
        date_col=date_col,
    )
