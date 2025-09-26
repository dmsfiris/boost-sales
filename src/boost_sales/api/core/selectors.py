# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Optional

import pandas as pd

ForecastScope = Literal[
    "single",
    "latest_per_pair",
    "latest_per_store",
    "latest_per_item",
    "last_n_days",
    "since_date",
    "at_date",
]


def _latest_per_pair(df: pd.DataFrame, date_col: str, group_cols: tuple[str, str]) -> pd.DataFrame:
    """
    Keep only the row with the max date for each (store,item).
    """
    g = df.groupby(list(group_cols), sort=False)
    idx = g[date_col].idxmax()
    return df.loc[idx.values].copy().reset_index(drop=True)


def select_rows(
    df: pd.DataFrame,
    scope: ForecastScope,
    c,  # columns config object with .store, .item, .date
    *,
    store_id: Optional[str] = None,
    item_id: Optional[str] = None,
    n_days: Optional[int] = None,
    since_date: Optional[date] = None,
    at_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Scope-aware selection of rows to score. Returns a new dataframe slice.

    Scopes:
      - single: latest row for the given (store_id, item_id)
      - latest_per_pair: latest row for every (store_id, item_id)
      - latest_per_store: for the given store_id, latest row for each item in that store
      - latest_per_item: for the given item_id, latest row for each store that carries it
      - last_n_days: rows in the last n_days (optional store_id/item_id filters)
      - since_date: rows on/after since_date (optional filters)
      - at_date: rows exactly equal to at_date (optional filters)
    """
    date_col = c.date
    store_col = c.store
    item_col = c.item

    # --- required columns present?
    missing = [col for col in (date_col, store_col, item_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Ensure datetime dtype (coerce invalids to NaT)
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if scope == "single":
        if not store_id or not item_id:
            raise ValueError("scope=single requires both store_id and item_id.")
        sub = df[(df[store_col] == store_id) & (df[item_col] == item_id)]
        if sub.empty:
            return sub.copy().reset_index(drop=True)
        latest_dt = sub[date_col].max()
        return sub[sub[date_col].eq(latest_dt)].copy().reset_index(drop=True)

    if scope == "latest_per_pair":
        return _latest_per_pair(df, date_col, (store_col, item_col))

    if scope == "latest_per_store":
        if not store_id:
            raise ValueError("scope=latest_per_store requires store_id.")
        sub = df[df[store_col] == store_id]
        if sub.empty:
            return sub.copy().reset_index(drop=True)
        return _latest_per_pair(sub, date_col, (store_col, item_col))

    if scope == "latest_per_item":
        if not item_id:
            raise ValueError("scope=latest_per_item requires item_id.")
        sub = df[df[item_col] == item_id]
        if sub.empty:
            return sub.copy().reset_index(drop=True)
        return _latest_per_pair(sub, date_col, (store_col, item_col))

    if scope == "last_n_days":
        if not n_days or n_days < 1:
            raise ValueError("scope=last_n_days requires n_days >= 1.")
        latest_dt = df[date_col].max()
        # If df is empty, latest_dt is NaT and comparison yields empty slice (fine)
        cutoff = latest_dt - timedelta(days=n_days - 1)
        sub = df[df[date_col] >= cutoff]
        if store_id:
            sub = sub[sub[store_col] == store_id]
        if item_id:
            sub = sub[sub[item_col] == item_id]
        return sub.copy().reset_index(drop=True)

    if scope == "since_date":
        if since_date is None:
            raise ValueError("scope=since_date requires since_date.")
        sd = pd.Timestamp(since_date)
        sub = df[df[date_col] >= sd]
        if store_id:
            sub = sub[sub[store_col] == store_id]
        if item_id:
            sub = sub[sub[item_col] == item_id]
        return sub.copy().reset_index(drop=True)

    if scope == "at_date":
        if at_date is None:
            raise ValueError("scope=at_date requires at_date.")
        # Normalize both sides so time components don't prevent matches
        ad = pd.Timestamp(at_date).normalize()
        dts = df[date_col].dt.normalize()
        sub = df[dts == ad]
        if store_id:
            sub = sub[sub[store_col] == store_id]
        if item_id:
            sub = sub[sub[item_col] == item_id]
        return sub.copy().reset_index(drop=True)

    raise ValueError(f"Unsupported scope: {scope}")
