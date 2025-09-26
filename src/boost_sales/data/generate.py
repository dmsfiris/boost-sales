from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from boost_sales.config import AppConfig


@dataclass
class Schema:
    date: str = "date"
    store: str = "store_id"
    item: str = "item_id"
    price: str = "price"    # float, <= 2 decimals
    promo: str = "promo"    # 0/1 int
    sales: str = "sales"    # integer

    # transactions/raw names (if using tx -> panel)
    qty: str = "quantity"
    unit_price: str = "unit_price"
    promo_tx: Optional[str] = None  # e.g., "promo_flag" in raw lines


def _order_cols(df: pd.DataFrame, sch: Schema) -> pd.DataFrame:
    """Ensure legacy column order."""
    cols = [sch.date, sch.store, sch.item, sch.price, sch.promo, sch.sales]
    present = [c for c in cols if c in df.columns]
    return df[present]


# -------------------------------
# Synthetic panel (legacy-like)
# -------------------------------

def generate_synthetic(
    *,
    n_stores: int = 5,
    n_items: int = 50,
    start: str = "2024-01-01",
    periods: int = 365,
    seed: int = 42,
    schema: Schema = Schema(),
) -> pd.DataFrame:
    """
    Synthetic daily panel matching legacy scales:
      price ~ 100..600 (2 decimals), promo in {0,1} ~10%, sales integer.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="D")
    stores = [f"S{i:02d}" for i in range(1, n_stores + 1)]
    items = [f"I{i:02d}" for i in range(1, n_items + 1)]

    idx = pd.MultiIndex.from_product([stores, items, dates], names=[schema.store, schema.item, schema.date])
    df = pd.DataFrame(index=idx).reset_index()

    dow = df[schema.date].dt.dayofweek
    day_fx = 6 * (dow.isin([5, 6]).astype(int))  # weekend boost
    store_fx = df[schema.store].str[-2:].astype(int).to_numpy() / 2.5
    item_fx = df[schema.item].str[-2:].astype(int).to_numpy() / 2.0

    promo = (rng.random(len(df)) < 0.10).astype(np.int8)  # numpy int8
    promo_s = pd.Series(promo, dtype="int8")              # pandas series int8

    base_sales = 18 + day_fx + store_fx + item_fx
    sales = base_sales + rng.normal(0, 3, len(df)) + promo * 5
    sales = np.clip(np.rint(sales), 0, None).astype("int64")  # plain int64

    price = 280 + (item_fx * 6) + (store_fx * 3) + rng.normal(0, 4, len(df))
    price = np.clip(price, 90, 600)
    price = np.round(price, 2)  # two decimals

    out = pd.DataFrame({
        schema.date: df[schema.date],
        schema.store: df[schema.store].astype("string"),
        schema.item: df[schema.item].astype("string"),
        schema.price: price.astype(float),
        schema.promo: promo_s,            # int8 series
        schema.sales: sales,              # int64 array
    }).sort_values([schema.store, schema.item, schema.date]).reset_index(drop=True)

    return _order_cols(out, schema)


# --------------------------------
# Flat panel pass-through
# --------------------------------

def generate_from_flat(
    flat_csv: Path,
    *,
    schema: Schema = Schema(),
    parse_dates: bool = True,
) -> pd.DataFrame:
    engine = "pyarrow"
    try:
        import pyarrow  # noqa
    except Exception:
        engine = "c"

    df = pd.read_csv(flat_csv, engine=engine, low_memory=False)

    if parse_dates and schema.date in df.columns:
        df[schema.date] = pd.to_datetime(df[schema.date], errors="raise")
    for c in (schema.store, schema.item):
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    if schema.price in df.columns:
        df[schema.price] = pd.to_numeric(df[schema.price], errors="coerce").round(2)
    if schema.promo in df.columns:
        df[schema.promo] = pd.to_numeric(df[schema.promo], errors="coerce").fillna(0).astype("int8")
    if schema.sales in df.columns:
        df[schema.sales] = pd.to_numeric(df[schema.sales], errors="coerce").round(0).astype("int64")

    df = df.sort_values([schema.store, schema.item, schema.date]).reset_index(drop=True)
    return _order_cols(df, schema)


# --------------------------------
# Transactions -> daily panel
# --------------------------------

def _price_aggregate(
    lines: pd.DataFrame,
    groupers: Sequence[str],
    *,
    qty_col: str,
    unit_price_col: str,
    strategy: str = "weighted_avg",  # 'weighted_avg' | 'mean' | 'median' | 'last'
) -> pd.Series:
    q = pd.to_numeric(lines[qty_col], errors="coerce").fillna(0.0)
    p = pd.to_numeric(lines[unit_price_col], errors="coerce").fillna(np.nan)

    if strategy == "weighted_avg":
        val = q * p
        agg = lines.assign(_val=val, _q=q).groupby(groupers, as_index=True)[["_val", "_q"]].sum()
        price = agg["_val"] / agg["_q"]
    elif strategy == "mean":
        price = lines.groupby(groupers, as_index=True)[unit_price_col].mean()
    elif strategy == "median":
        price = lines.groupby(groupers, as_index=True)[unit_price_col].median()
    elif strategy == "last":
        last = lines.sort_values(groupers + [unit_price_col]).groupby(groupers, as_index=True)[unit_price_col].last()
        price = last
    else:
        raise ValueError("price strategy must be one of: weighted_avg, mean, median, last")
    return price


def generate_from_transactions(
    tx_csv: Path,
    *,
    schema: Schema = Schema(),
    parse_dates: bool = True,
    group_extra: Optional[Sequence[str]] = None,
    sales_as: str = "sum",            # 'sum' (qty) | 'count' (lines)
    price_strategy: str = "weighted_avg",
    promo_strategy: str = "column",   # 'column' | 'price_drop_vs_roll'
    promo_col_name: Optional[str] = None,
    promo_roll_window: int = 28,
    promo_drop_threshold: float = 0.10,
) -> pd.DataFrame:
    engine = "pyarrow"
    try:
        import pyarrow  # noqa
    except Exception:
        engine = "c"

    head = pd.read_csv(tx_csv, engine=engine, nrows=0)
    required = [schema.date, schema.store, schema.item]
    if sales_as == "sum":
        required.append(schema.qty)
    missing = [c for c in required if c not in head.columns]
    if missing:
        raise ValueError(f"Transactions CSV missing required columns: {missing}")

    df = pd.read_csv(tx_csv, engine=engine, low_memory=False)
    if parse_dates:
        df[schema.date] = pd.to_datetime(df[schema.date], errors="raise")
    df[schema.store] = df[schema.store].astype("string").str.strip()
    df[schema.item] = df[schema.item].astype("string").str.strip()

    groupers = [schema.store, schema.item, schema.date]
    if group_extra:
        groupers = list(groupers) + list(group_extra)

    # SALES (integer)
    if sales_as == "sum":
        df[schema.qty] = pd.to_numeric(df[schema.qty], errors="coerce").fillna(0.0)
        g_sales = df.groupby(groupers, as_index=True)[schema.qty].sum()
    elif sales_as == "count":
        g_sales = df.groupby(groupers, as_index=True).size().astype(float)
    else:
        raise ValueError("sales_as must be 'sum' or 'count'")
    g_sales = np.rint(g_sales).astype("int64").rename(schema.sales)

    # PRICE (<= 2 decimals)
    if schema.unit_price in df.columns:
        g_price = _price_aggregate(
            df, groupers, qty_col=schema.qty, unit_price_col=schema.unit_price, strategy=price_strategy
        ).round(2).rename(schema.price)
    else:
        g_price = pd.Series(index=g_sales.index, dtype="float64", name=schema.price)

    # PROMO
    if promo_strategy == "column" and (promo_col_name or schema.promo_tx) and ((promo_col_name or schema.promo_tx) in df.columns):
        pcol = promo_col_name or schema.promo_tx  # type: ignore[assignment]
        g_promo = pd.to_numeric(df[pcol], errors="coerce").fillna(0).astype("int8").groupby(groupers).max()
    elif promo_strategy == "price_drop_vs_roll" and schema.unit_price in df.columns:
        daily_price = df.groupby(groupers, as_index=False)[schema.unit_price].mean().sort_values(groupers)
        key = [schema.store, schema.item]
        daily_price["_roll"] = daily_price.groupby(key)[schema.unit_price].transform(
            lambda s: s.rolling(promo_roll_window, min_periods=1).mean()
        )
        daily_price["_promo"] = (daily_price[schema.unit_price] <= (1 - promo_drop_threshold) * daily_price["_roll"]).astype("int8")
        g_promo = daily_price.set_index(groupers)["_promo"]
    else:
        g_promo = pd.Series(0, index=g_sales.index, dtype="int8")

    g_promo = g_promo.astype("int8").rename(schema.promo)

    panel = pd.concat([g_price, g_promo, g_sales], axis=1).reset_index()
    panel = panel.sort_values([schema.store, schema.item, schema.date]).reset_index(drop=True)
    return _order_cols(panel, schema)


# --------------------------------
# Public entry points
# --------------------------------

def build_dataset(
    *,
    mode: str,
    cfg: Optional[AppConfig] = None,
    # schema overrides (names)
    date_col: Optional[str] = None,
    store_col: Optional[str] = None,
    item_col: Optional[str] = None,
    price_col: Optional[str] = None,
    promo_col: Optional[str] = None,
    sales_col: Optional[str] = None,
    # synthetic
    n_stores: int = 5,
    n_items: int = 50,
    start: str = "2024-01-01",
    periods: int = 365,
    seed: int = 42,
    # flat
    flat_csv: Optional[Path] = None,
    # transactions
    tx_csv: Optional[Path] = None,
    qty_col: Optional[str] = None,
    unit_price_col: Optional[str] = None,
    group_extra: Optional[Sequence[str]] = None,
    sales_as: str = "sum",
    price_strategy: str = "weighted_avg",
    promo_strategy: str = "column",
    promo_col_name: Optional[str] = None,
    promo_roll_window: int = 28,
    promo_drop_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Build a training-ready panel matching legacy layout:
      date, store_id, item_id, price, promo, sales
    """
    cfg = cfg or AppConfig()
    schema = Schema(
        date=date_col or cfg.cols.date,
        store=store_col or cfg.cols.store,
        item=item_col or cfg.cols.item,
        price=price_col or cfg.cols.price,
        promo=promo_col or cfg.cols.promo,
        sales=sales_col or cfg.cols.sales,
        qty=qty_col or Schema.qty,
        unit_price=unit_price_col or Schema.unit_price,
        promo_tx=promo_col_name,
    )

    if mode == "synthetic":
        return generate_synthetic(
            n_stores=n_stores, n_items=n_items, start=start, periods=periods, seed=seed, schema=schema
        )
    if mode == "from-flat":
        if not flat_csv:
            raise ValueError("mode='from-flat' requires flat_csv")
        return generate_from_flat(flat_csv, schema=schema)
    if mode == "from-transactions":
        if not tx_csv:
            raise ValueError("mode='from-transactions' requires tx_csv")
        return generate_from_transactions(
            tx_csv,
            schema=schema,
            group_extra=group_extra,
            sales_as=sales_as,
            price_strategy=price_strategy,
            promo_strategy=promo_strategy,
            promo_col_name=promo_col_name,
            promo_roll_window=promo_roll_window,
            promo_drop_threshold=promo_drop_threshold,
        )
    raise ValueError("mode must be one of: synthetic | from-flat | from-transactions")


def write_dataset(df: pd.DataFrame, out_csv: Path) -> Path:
    """
    Write CSV (keep integers for `sales`; `price` already rounded to 2 decimals).
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _order_cols(df, Schema()).to_csv(out_csv, index=False)
    return out_csv
