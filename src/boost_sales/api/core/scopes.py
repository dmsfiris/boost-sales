# SPDX-License-Identifier: MIT
from __future__ import annotations

import pandas as pd

from boost_sales.config import AppConfig
from boost_sales.api.schemas import ForecastRequest


def apply_scope(df: pd.DataFrame, cfg: AppConfig, req: ForecastRequest) -> pd.DataFrame:
    """
    Filter rows according to the requested scope, returning eligible base rows.
    For "latest_*" scopes, reduce to the most recent base date per key.

    Supported scopes (aligned with ForecastRequest):
      - "single": latest row for the specified (store_id, item_id)
      - "latest_per_pair": latest row per (store_id, item_id) across the dataset
      - "latest_per_store": latest row per (store_id, item_id) within a given store_id
      - "latest_per_item": latest row per (store_id, item_id) within a given item_id
      - "last_n_days": rows within the last n_days (optional filters: store_id, item_id)
      - "since_date": rows on/after since_date (optional filters: store_id, item_id)
      - "at_date": rows exactly at at_date (optional filters: store_id, item_id)
    """
    c = cfg.columns
    if df.empty:
        return df.copy()

    # ensure deterministic ordering before groupby operations
    df = df.sort_values([c.store, c.item, c.date]).copy()

    scope = req.scope

    # ---- single: require both ids; return the latest date for that pair
    if scope == "single":
        if not req.store_id or not req.item_id:
            raise ValueError("scope=single requires both 'store_id' and 'item_id'.")
        g = df[(df[c.store] == req.store_id) & (df[c.item] == req.item_id)].copy()
        if g.empty:
            return g
        latest = g[c.date].max()
        return g[g[c.date].eq(latest)]

    # ---- latest per pair: one row per (store,item), the latest date
    if scope == "latest_per_pair":
        idx = df.groupby([c.store, c.item], sort=False)[c.date].transform("max").eq(df[c.date])
        return df[idx].copy()

    # ---- latest per store: require store_id; latest row for each item within that store
    if scope == "latest_per_store":
        if not req.store_id:
            raise ValueError("scope=latest_per_store requires 'store_id'.")
        g = df[df[c.store] == req.store_id].copy()
        if g.empty:
            return g
        idx = g.groupby([c.store, c.item], sort=False)[c.date].transform("max").eq(g[c.date])
        return g[idx].copy()

    # ---- latest per item: require item_id; latest row for each store for that item
    if scope == "latest_per_item":
        if not req.item_id:
            raise ValueError("scope=latest_per_item requires 'item_id'.")
        g = df[df[c.item] == req.item_id].copy()
        if g.empty:
            return g
        idx = g.groupby([c.store, c.item], sort=False)[c.date].transform("max").eq(g[c.date])
        return g[idx].copy()

    # ---- last_n_days: require n_days; optional store_id/item_id filters
    if scope == "last_n_days":
        if not req.n_days or req.n_days < 1:
            raise ValueError("scope=last_n_days requires a positive 'n_days'.")
        maxd = pd.to_datetime(df[c.date]).max()
        cutoff = maxd - pd.Timedelta(days=int(req.n_days) - 1)
        mask = pd.to_datetime(df[c.date]) >= cutoff
        if req.store_id:
            mask &= df[c.store].eq(req.store_id)
        if req.item_id:
            mask &= df[c.item].eq(req.item_id)
        return df[mask].copy()

    # ---- since_date: require since_date; optional store_id/item_id filters
    if scope == "since_date":
        if not req.since_date:
            raise ValueError("scope=since_date requires 'since_date'.")
        cutoff = pd.to_datetime(req.since_date)
        mask = pd.to_datetime(df[c.date]) >= cutoff
        if req.store_id:
            mask &= df[c.store].eq(req.store_id)
        if req.item_id:
            mask &= df[c.item].eq(req.item_id)
        return df[mask].copy()

    # ---- at_date: require at_date; optional store_id/item_id filters
    if scope == "at_date":
        if not req.at_date:
            raise ValueError("scope=at_date requires 'at_date'.")
        target = pd.to_datetime(req.at_date)
        # compare on normalized dates to ignore time component if present
        dts = pd.to_datetime(df[c.date])
        mask = dts.dt.normalize().eq(pd.Timestamp(target).normalize())
        if req.store_id:
            mask &= df[c.store].eq(req.store_id)
        if req.item_id:
            mask &= df[c.item].eq(req.item_id)
        return df[mask].copy()

    # Defensive default: no rows
    return df.head(0).copy()
