# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Iterator, Tuple

import pandas as pd


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> None:
    """Coerce `date_col` to datetime if needed (in place)."""
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col!r}")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")


def iter_pairs(
    df: pd.DataFrame,
    *,
    store_col: str = "store_id",
    item_col: str = "item_id",
    date_col: str = "date",
    copy_groups: bool = True,
) -> Iterator[Tuple[str, str, pd.DataFrame]]:
    """
    Yield (store_id, item_id, group_df) for each (store,item), with group_df sorted by date.
    """
    for col in (store_col, item_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col!r}")

    _ensure_datetime(df, date_col)

    # presort once; groupby(sort=False) keeps the within-group order
    df_sorted = df.sort_values([store_col, item_col, date_col], kind="stable")

    for (s, i), g in df_sorted.groupby([store_col, item_col], sort=False, observed=True):
        yield str(s), str(i), (g.copy() if copy_groups else g)


def iter_items(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    date_col: str = "date",
    copy_groups: bool = True,
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Yield (item_id, group_df) for each item_id, with group_df sorted by date.
    """
    if item_col not in df.columns:
        raise ValueError(f"Missing required column: {item_col!r}")

    _ensure_datetime(df, date_col)
    df_sorted = df.sort_values([item_col, date_col], kind="stable")

    for i, g in df_sorted.groupby(item_col, sort=False, observed=True):
        yield str(i), (g.copy() if copy_groups else g)


def iter_stores(
    df: pd.DataFrame,
    *,
    store_col: str = "store_id",
    date_col: str = "date",
    copy_groups: bool = True,
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Yield (store_id, group_df) for each store_id, with group_df sorted by date.
    """
    if store_col not in df.columns:
        raise ValueError(f"Missing required column: {store_col!r}")

    _ensure_datetime(df, date_col)
    df_sorted = df.sort_values([store_col, date_col], kind="stable")

    for s, g in df_sorted.groupby(store_col, sort=False, observed=True):
        yield str(s), (g.copy() if copy_groups else g)


__all__ = ["iter_pairs", "iter_items", "iter_stores"]
