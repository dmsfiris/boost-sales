# SPDX-License-Identifier: MIT
"""
Training pipeline (global and per-group) with parity controls.

It wires together:
- data.io.load_sales_csv
- api.core.features.prepare_features        ← unified horizon-agnostic features
- api.core.controls.add_horizon_controls    ← unified horizon-specific controls
- models.xgb.train_one_horizon / save_booster
- pipeline.artifacts.outdir_for / write_categories
- pipeline.groupers.iter_* (for per-group loops)

All behavior is controlled by AppConfig (see config.py).

Notes on validation:
- If cfg.train.valid_cutoff_date is provided, we use it as a time-based split.
- Otherwise, if cfg.train.valid_tail_days is set (e.g., 28), we derive a cutoff as:
    max(date) - valid_tail_days
  (Per-group for train-per-group.)
- If both are None, we train on all data without a validation set.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from boost_sales.api.core.controls import add_horizon_controls
from boost_sales.api.core.features import prepare_features
from boost_sales.config import AppConfig, xgb_params_from
from boost_sales.data.io import load_sales_csv
from boost_sales.models.xgb import save_booster, train_one_horizon
from boost_sales.pipeline.artifacts import outdir_for, write_categories
from boost_sales.pipeline.groupers import iter_items, iter_pairs, iter_stores


def _save_holidays_meta(outdir: Path, cfg: AppConfig) -> None:
    """Persist the holiday region so serving can mirror it."""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "holidays_meta.json").write_text(
        json.dumps(
            {"country": cfg.train.hol_country, "subdiv": cfg.train.hol_subdiv},
            indent=2,
        ),
        encoding="utf-8",
    )


def _schema_kwargs(cfg: AppConfig) -> dict:
    """Expand cfg.columns into explicit keyword args for the CSV loader."""
    c = cfg.columns
    return {
        "date_col": c.date,
        "store_col": c.store,
        "item_col": c.item,
        "sales_col": c.sales,
        "price_col": c.price,
        "promo_col": c.promo,
    }


def _derive_auto_cutoff(df: pd.DataFrame, date_col: str, tail_days: Optional[int]) -> Optional[str]:
    """
    Compute a YYYY-MM-DD validation cutoff as max(date) - tail_days.
    Returns None if tail_days is falsy or dates are missing.
    """
    if not tail_days or tail_days <= 0 or date_col not in df.columns:
        return None
    dates = pd.to_datetime(df[date_col], errors="coerce")
    last = dates.max()
    if pd.isna(last):
        return None
    cutoff = (last - pd.Timedelta(days=int(tail_days))).date()
    return cutoff.isoformat()


# ----------------------------
# Global training
# ----------------------------
def train_global(cfg: AppConfig) -> None:
    """
    Train a single global model per horizon on the entire dataset.
    Artifacts in cfg.paths.models_dir:
      - model_h{h}.xgb.json
      - feature_order_h{h}.json
      - holidays_meta.json
    """
    # Load data with schema awareness and parsed dates
    df = load_sales_csv(cfg.paths.data_csv, parse_dates=True, **_schema_kwargs(cfg))
    c = cfg.columns

    # Horizon-agnostic features (exact parity with serving)
    df = prepare_features(
        df,
        cfg,
        hol_country=cfg.train.hol_country,
        hol_subdiv=cfg.train.hol_subdiv,
    )

    _save_holidays_meta(cfg.paths.models_dir, cfg)

    # Validation cutoff resolution:
    # 1) explicit valid_cutoff_date if provided
    # 2) else derive from valid_tail_days (e.g., last 28 days)
    auto_cutoff = cfg.train.valid_cutoff_date or _derive_auto_cutoff(df, c.date, cfg.train.valid_tail_days)

    params = xgb_params_from(cfg)
    for h in cfg.train.horizons:
        # Unified builder (no-op if future controls disabled in cfg.future)
        dfh = add_horizon_controls(
            df,
            horizon=h,
            cfg=cfg,
            date_col=c.date,
            price_col=c.price,
            promo_col=c.promo,
        )

        booster, feats = train_one_horizon(
            dfh,
            h,
            params,
            required_feature_notna=cfg.train.required_feature_notna,
            valid_cutoff_date=auto_cutoff,
            early_stopping_rounds=cfg.train.early_stopping_rounds,
            verbose_eval=cfg.train.verbose_eval,
        )

        save_booster(booster, feats, cfg.paths.models_dir, h)


# ----------------------------
# Per-group training
# ----------------------------
def _resolve_group_info(scope: str, keys) -> Tuple[Optional[str], Optional[str]]:
    """Returns (store_id, item_id) strings according to the chosen scope."""
    if scope == "pair":
        store = str(keys[0])
        item = str(keys[1])
    elif scope == "item":
        store = None
        item = str(keys[0])
    elif scope == "store":
        store = str(keys[0])
        item = None
    else:
        raise ValueError("scope must be one of: pair, item, store")
    return store, item


def train_per_group(cfg: AppConfig, scope: str) -> None:
    """
    Train one model per horizon for each group defined by `scope`.

    Layout:
      models/
        by_pair/{store}__{item}/model_h{h}.xgb.json
        by_item/{item}/model_h{h}.xgb.json
        by_store/{store}/model_h{h}.xgb.json

      Also per group:
        - feature_order_h{h}.json
        - holidays_meta.json
        - categories.json
    """
    # Load data with schema awareness and parsed dates
    df = load_sales_csv(cfg.paths.data_csv, parse_dates=True, **_schema_kwargs(cfg))
    c = cfg.columns

    # Horizon-agnostic features (exact parity with serving)
    df = prepare_features(
        df,
        cfg,
        hol_country=cfg.train.hol_country,
        hol_subdiv=cfg.train.hol_subdiv,
    )

    # Choose iterator based on scope (schema-aware, stable sorting inside)
    if scope == "pair":
        iterator = iter_pairs(df, store_col=c.store, item_col=c.item, date_col=c.date)
    elif scope == "item":
        iterator = iter_items(df, item_col=c.item, date_col=c.date)
    elif scope == "store":
        iterator = iter_stores(df, store_col=c.store, date_col=c.date)
    else:
        raise ValueError("scope must be one of: pair, item, store")

    params = xgb_params_from(cfg)

    for group_info in iterator:
        if scope == "pair":
            store_id, item_id, g = group_info
        elif scope == "item":
            item_id, g = group_info
            store_id = None
        else:  # scope == "store"
            store_id, g = group_info
            item_id = None

        outdir = outdir_for(scope, cfg.paths.models_dir, store_id, item_id)
        outdir.mkdir(parents=True, exist_ok=True)

        _save_holidays_meta(outdir, cfg)
        write_categories(outdir, g[c.store].unique(), g[c.item].unique())

        # Per-group dynamic cutoff (if explicit cutoff not given)
        auto_cutoff = cfg.train.valid_cutoff_date or _derive_auto_cutoff(g, c.date, cfg.train.valid_tail_days)

        for h in cfg.train.horizons:
            # Unified builder (no-op if future controls disabled in cfg.future)
            dfh = add_horizon_controls(
                g,
                horizon=h,
                cfg=cfg,
                date_col=c.date,
                price_col=c.price,
                promo_col=c.promo,
            )

            booster, feats = train_one_horizon(
                dfh,
                h,
                params,
                required_feature_notna=cfg.train.required_feature_notna,
                valid_cutoff_date=auto_cutoff,
                early_stopping_rounds=cfg.train.early_stopping_rounds,
                verbose_eval=cfg.train.verbose_eval,
            )

            save_booster(booster, feats, outdir, h)
