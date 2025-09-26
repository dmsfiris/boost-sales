# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union, IO

import pandas as pd

from boost_sales.config import Columns  # for type hints (optional)

# Accept file paths or file-like objects (BytesIO/StringIO)
PathOrBuf = Union[str, Path, IO[bytes], IO[str]]

# Canonical default dtypes for non-date columns (final casts happen after read).
DEFAULT_DTYPES: Dict[str, str] = {
    "store_id": "string",
    "item_id": "string",
    "price": "float64",
    "promo": "int8",
    "sales": "int64",
}


def _order_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    return df[present]


def load_sales_csv(
    path_or_buf: PathOrBuf,
    schema: Optional[Columns] = None,
    *,
    usecols: Optional[Sequence[str]] = None,
    extra_dtypes: Optional[Dict[str, str]] = None,
    # column names (overrides; if schema is provided these are ignored)
    date_col: str = "date",
    store_col: str = "store_id",
    item_col: str = "item_id",
    sales_col: str = "sales",
    price_col: str = "price",
    promo_col: str = "promo",
    parse_dates: bool = True,
    keep_tz: bool = False,
    # if you prefer forgiving date parsing set to "coerce"; default "raise" is stricter
    datetime_errors: str = "raise",  # "raise" | "coerce" | "ignore"
) -> pd.DataFrame:
    """
    Load a sales CSV (or buffer) using either a provided schema (preferred) or explicit
    column-name overrides.

    Supports both:
        load_sales_csv(path, schema=cfg.columns, parse_dates=True)
        load_sales_csv(path, cfg.columns, parse_dates=True)
        load_sales_csv(BytesIO(data), schema=cfg.columns, parse_dates=True)

    Behavior:
      - Parses date column to datetime (strict by default).
      - Normalizes IDs → pandas 'string' (strip whitespace).
      - price → float64 (rounded to 2 decimals)
      - promo → int8 (NaNs → 0)
      - sales → int64 (NaNs → 0, rounded)
      - Returns only the canonical columns in canonical order.
    """
    # Resolve column names from schema if present
    if schema is not None:
        date_col = schema.date
        store_col = schema.store
        item_col = schema.item
        sales_col = schema.sales
        price_col = schema.price
        promo_col = schema.promo

    # Build dtype mapping for non-date columns (respect any extras)
    dtypes = DEFAULT_DTYPES.copy()
    if extra_dtypes:
        dtypes.update(extra_dtypes)

    wanted_cols = usecols or [date_col, store_col, item_col, price_col, promo_col, sales_col]

    # Try fast/robust engine where available, fall back gracefully
    try:
        df = pd.read_csv(
            path_or_buf,  # type: ignore[arg-type]
            usecols=wanted_cols if usecols is not None else None,
            dtype={k: v for k, v in dtypes.items() if k in wanted_cols and k != date_col},
            engine="pyarrow",
        )
    except Exception:
        df = pd.read_csv(
            path_or_buf,  # type: ignore[arg-type]
            usecols=wanted_cols if usecols is not None else None,
            dtype={k: v for k, v in dtypes.items() if k in wanted_cols and k != date_col},
            engine="c",
            low_memory=False,
        )

    # Validate required columns exist
    required = {date_col, store_col, item_col, price_col, promo_col, sales_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    # Date parsing
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors=datetime_errors, utc=keep_tz)
        if not keep_tz and pd.api.types.is_datetime64tz_dtype(df[date_col].dtype):
            df[date_col] = df[date_col].dt.tz_localize(None)

    # Normalize IDs
    for c in (store_col, item_col):
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    # Numeric casts
    if price_col in df.columns:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce").astype("float64").round(2)
    if promo_col in df.columns:
        df[promo_col] = (
            pd.to_numeric(df[promo_col], errors="coerce")
            .fillna(0)
            .round(0)
            .astype("int8")
        )
    if sales_col in df.columns:
        df[sales_col] = (
            pd.to_numeric(df[sales_col], errors="coerce")
            .fillna(0)
            .round(0)
            .astype("int64")
        )

    # Canonical order (drop any extras silently, keep known order)
    df = _order_cols(df, [date_col, store_col, item_col, price_col, promo_col, sales_col])
    return df
