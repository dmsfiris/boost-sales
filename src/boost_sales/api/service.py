# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from boost_sales.api.core.controls import (
    add_horizon_controls,
    assume_no_change_fill,
)
from boost_sales.api.core.features import prepare_features

# ---- Core utilities (reuse, don't duplicate) ----
from boost_sales.api.core.horizons import parse_horizons_opt
from boost_sales.api.core.paginate import paginate_df
from boost_sales.api.core.selectors import select_rows
from boost_sales.config import AppConfig
from boost_sales.models.xgb import load_booster

from .schemas import ForecastRequest, ForecastResponse, PageMeta, PredictionRow, UnitType


# -----------------------------
# Helpers
# -----------------------------
def _schema_kwargs(cfg: AppConfig) -> dict:
    """Expand cfg.columns into explicit keyword args for the CSV loader."""
    c = cfg.columns
    return dict(
        date_col=c.date,
        store_col=c.store,
        item_col=c.item,
        sales_col=c.sales,
        price_col=c.price,
        promo_col=c.promo,
    )


def _format_sales_series(vals: pd.Series, unit_type: UnitType, decimal_places: int) -> pd.Series:
    s = pd.Series(vals, dtype="float64")
    if unit_type == "integer":
        # Round to int and represent as float for schema compatibility (sales: float)
        return s.round(0).astype("int64").astype("float64")
    return s.round(int(decimal_places)).astype("float64")


def _models_require_future_controls(models_dir: Path, horizons: List[int]) -> bool:
    """
    Return True if any horizon model expects price_h*/promo_h* features.

    Fast path: read feature_order_h*.json (no need to load boosters).
    Fallback: load_booster if the feature file is missing.
    """
    mdir = Path(models_dir)
    for h in horizons:
        feats: Optional[List[str]] = None
        feats_path = mdir / f"feature_order_h{h}.json"
        if feats_path.exists():
            try:
                feats = json.loads(feats_path.read_text(encoding="utf-8"))
            except Exception:
                feats = None  # fall through to load_booster
        if feats is None:
            try:
                _, feats = load_booster(mdir, h)
            except Exception:
                continue  # model missing; ignore for this check
        if any(str(f).startswith(("price_h", "promo_h")) for f in feats):
            return True
    return False


# -----------------------------
# Optional: align holiday region with training
# -----------------------------
def _load_holiday_meta_if_present(models_dir: Path) -> Optional[Tuple[str, Optional[str]]]:
    """
    If training saved holidays_meta.json in models_dir, return (country, subdiv).
    Otherwise, return None and keep current cfg settings.
    """
    meta_path = Path(models_dir) / "holidays_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("country", "US"), meta.get("subdiv")
    except Exception:
        # Never fail serving on a metadata read issue
        return None


# -----------------------------
# Data loading + feature prep
# -----------------------------
def _load_df_from_source(cfg: AppConfig, req: ForecastRequest, csv_bytes: Optional[bytes]) -> pd.DataFrame:
    """
    Load raw CSV (server/demo or uploaded), then build horizon-agnostic features.

    NOTE: boost_sales.data.io.load_sales_csv expects a Path; for uploaded bytes
    we write a temporary file (Windows-safe), read it, then unlink it here.
    """
    from boost_sales.data.io import load_sales_csv  # path-based reader

    if csv_bytes is not None:
        import os
        import tempfile

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(csv_bytes)
                tmp_path = Path(tmp.name)
            df = load_sales_csv(
                tmp_path,
                parse_dates=True,
                **_schema_kwargs(cfg),
            )
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    elif req.use_server_csv:
        if not getattr(cfg.paths, "data_csv", None):
            raise ValueError("Service has no demo CSV configured on the server.")
        df = load_sales_csv(
            cfg.paths.data_csv,
            parse_dates=True,
            **_schema_kwargs(cfg),
        )
    else:
        raise ValueError("No input data provided. Upload a CSV or set 'use_server_csv=true'.")

    # Single source of truth for feature building (parity with training)
    return prepare_features(
        df,
        cfg,
        hol_country=cfg.train.hol_country,
        hol_subdiv=cfg.train.hol_subdiv,
    )


# ---- Future plan parsing & application ---------------------------------------
def _parse_price_plan(plan: Optional[str], horizons: List[int]) -> tuple[str, dict[int, float]]:
    """
    Returns ("none" | "scalar" | "list", mapping)
    - "scalar": dict {h: scalar_multiplier}
    - "list":   dict {h: absolute_price}  (values per horizon)
    - "none":   {}
    Supports percent scalar like "50%" -> 0.5.
    """
    if not plan:
        return "none", {}
    s = plan.strip()

    # percent scalar like "50%"
    if s.endswith("%"):
        try:
            pct = float(s[:-1].strip()) / 100.0
            return "scalar", {h: pct for h in horizons}
        except Exception:
            pass

    # scalar multiplier
    try:
        val = float(s)
        return "scalar", {h: float(val) for h in horizons}
    except Exception:
        pass

    # CSV list of absolutes
    parts = [p.strip() for p in s.split(",") if p.strip()]
    vals: list[float] = []
    for p in parts:
        vals.append(float(p))
    if not vals:
        return "none", {}
    m: dict[int, float] = {}
    for i, h in enumerate(horizons):
        if i < len(vals):
            m[h] = float(vals[i])
    return "list", m


def _parse_promo_plan(plan: Optional[str], horizons: List[int]) -> tuple[str, dict[int, float]]:
    """
    Returns ("none" | "scalar" | "list", mapping) of promo intensities in [0,1].
    Accepts:
      - "0", "1", "true", "false"
      - scalar floats like "0.5"
      - percent like "50%"
      - CSV list like "0,0,0,0.5,0,0,0"
    Values are clamped to [0,1].
    """
    if not plan:
        return "none", {}
    s = plan.strip().lower()

    def _to_float(tok: str) -> float:
        t = tok.strip().lower()
        if t in {"1", "true"}:
            return 1.0
        if t in {"0", "false"}:
            return 0.0
        if t.endswith("%"):
            return max(0.0, min(1.0, float(t[:-1]) / 100.0))
        return max(0.0, min(1.0, float(t)))

    # Try scalar
    try:
        v = _to_float(s)
        return "scalar", {h: v for h in horizons}
    except Exception:
        pass

    # Try CSV list
    parts = [p for p in s.split(",") if p.strip()]
    if not parts:
        return "none", {}
    m: dict[int, float] = {}
    for i, h in enumerate(horizons):
        if i < len(parts):
            try:
                m[h] = _to_float(parts[i])
            except Exception:
                m[h] = 0.0
    return "list", m


def _has_nonbinary_intensity(promo_map: dict[int, float]) -> bool:
    """Return True if map contains any value strictly between 0 and 1."""
    for v in promo_map.values():
        if v > 0.0 + 1e-12 and v < 1.0 - 1e-12:
            return True
    return False


def _apply_future_overrides(
    dfh: pd.DataFrame,
    h: int,
    cfg: AppConfig,
    price_mode: str,
    price_map: dict[int, float],
    promo_mode: str,
    promo_map: dict[int, float],
    *,
    derive_price_from_promo: bool,
) -> None:
    """
    Mutates dfh in place for horizon h using parsed plans:
      - price_h{h} (scalar multiplies current price; list sets absolute)
      - promo_h{h} (scalar/list sets float intensity 0..1)
      - optionally derive price from promo intensity if no price plan given
      - price_ratio_h{h} recomputed when available
    """
    c = cfg.columns
    ph = f"price_h{h}"
    pmh = f"promo_h{h}"
    prh = f"price_ratio_h{h}"
    denom = cfg.future.denom_col

    # promo override (float intensity)
    if promo_mode in {"scalar", "list"} and h in promo_map:
        dfh[pmh] = float(promo_map[h])

    # price override explicit
    if price_mode == "scalar" and h in price_map:
        m = price_map[h]
        if c.price in dfh.columns:
            dfh[ph] = pd.to_numeric(dfh[c.price], errors="coerce").astype("float64") * float(m)
    elif price_mode == "list" and h in price_map:
        dfh[ph] = float(price_map[h])
    # else: optional derive from promo intensity if requested
    elif derive_price_from_promo and (promo_mode in {"scalar", "list"}) and h in promo_map:
        if c.price in dfh.columns:
            intensity = float(promo_map[h])  # 0..1
            dfh[ph] = pd.to_numeric(dfh[c.price], errors="coerce").astype("float64") * (1.0 - intensity)

    # price ratio recompute if possible and requested by config
    if cfg.future.add_price_ratio and denom in dfh.columns and ph in dfh.columns:
        num = pd.to_numeric(dfh[ph], errors="coerce").astype("float64")
        den = pd.to_numeric(dfh[denom], errors="coerce").astype("float64")
        safe = den.notna() & num.notna()
        if cfg.future.safe_zero_denominator:
            safe &= den.ne(0)
        dfh[prh] = np.nan
        dfh.loc[safe, prh] = (num[safe] / den[safe]).astype("float64")


def _predict_for_horizons(
    df_scope: pd.DataFrame,
    cfg: AppConfig,
    horizons: List[int],
    *,
    price_future: Optional[str],
    promo_future: Optional[str],
    unit_type: UnitType,
    decimal_places: int,
    fill_missing_with_current: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """
    For each horizon, build horizon controls, apply plan overrides (including optional
    price derivation from promo intensity), optionally fill missing *_h{h} with
    assume-no-change, then score with XGB, and return a concatenated dataframe with 'sales'.
    Also returns diagnostics (e.g., missing features, dropped rows).
    """
    c = cfg.columns
    all_out: List[pd.DataFrame] = []
    notes: list[str] = []

    price_mode, price_map = _parse_price_plan(price_future, horizons)
    promo_mode, promo_map = _parse_promo_plan(promo_future, horizons)

    # If no explicit price plan but promo intensities include a non-binary value,
    # auto-derive price from intensity for those horizons.
    derive_price_from_promo = (price_mode == "none") and _has_nonbinary_intensity(promo_map)

    for h in horizons:
        dfh = add_horizon_controls(
            df_scope,
            horizon=h,
            cfg=cfg,
            date_col=c.date,
            price_col=c.price,
            promo_col=c.promo,
        )

        # Apply plan overrides (in place)
        _apply_future_overrides(
            dfh,
            h,
            cfg,
            price_mode,
            price_map,
            promo_mode,
            promo_map,
            derive_price_from_promo=derive_price_from_promo,
        )

        # Optionally ensure remaining NaNs are filled from current price/promo
        if fill_missing_with_current:
            dfh = assume_no_change_fill(dfh, horizon=h, cfg=cfg)

        # Load booster & ensure features exist
        booster, feats = load_booster(cfg.paths.models_dir, h)
        missing = set(feats) - set(dfh.columns)
        if missing:
            notes.append(f"h{h}: missing features {sorted(missing)[:6]}{'...' if len(missing)>6 else ''}")
            continue

        ok = dfh[feats].notna().all(axis=1)
        ready = dfh.loc[ok].copy()
        dropped = int((~ok).sum())
        if dropped:
            notes.append(f"h{h}: dropped {dropped} row(s) due to NaNs in required features")
        if ready.empty:
            continue

        # Local import to keep import time fast
        import xgboost as xgb  # type: ignore

        dmat = xgb.DMatrix(ready[feats], feature_names=feats)
        yhat = booster.predict(dmat)

        # Format per requested unit settings
        yhat_fmt = _format_sales_series(pd.Series(yhat), unit_type, decimal_places).values

        out = pd.DataFrame(
            {
                "store_id": ready[c.store].astype("string"),
                "item_id": ready[c.item].astype("string"),
                "base_date": pd.to_datetime(ready[c.date]).dt.date,
                "target_date": (pd.to_datetime(ready[c.date]) + pd.to_timedelta(h, unit="D")).dt.date,
                "horizon": h,
                "sales": yhat_fmt,
            }
        )
        all_out.append(out)

    if not all_out:
        return pd.DataFrame(columns=["store_id", "item_id", "base_date", "target_date", "horizon", "sales"]), notes

    return (
        pd.concat(all_out, ignore_index=True)
        .sort_values(["store_id", "item_id", "target_date", "horizon"])
        .reset_index(drop=True),
        notes,
    )


# -----------------------------
# Public API used by server
# -----------------------------
def forecast(cfg: AppConfig, req: ForecastRequest, csv_bytes: Optional[bytes] = None) -> ForecastResponse:
    """
    Main API entrypoint:
      - Parse horizons (core)
      - Load data & build features (core)
      - Select scope rows (core)
      - Build horizon controls, apply future plans (optionally derive price from promo & assume-no-change), predict
      - Paginate (core)
      - Format response using cfg.output defaults unless overridden in request
    """
    # Use config output defaults unless overridden
    unit_type: UnitType = cast(UnitType, req.unit_type or cfg.output.unit_type)
    decimal_places: int = (
        req.decimal_places if (req.decimal_places is not None and unit_type == "float") else cfg.output.decimal_places
    )

    # Align holiday region with training if metadata is present (optional but recommended)
    hol_meta = _load_holiday_meta_if_present(cfg.paths.models_dir)
    if hol_meta is not None:
        # mutate a shallow copy to avoid surprising global changes
        cfg = AppConfig(**cfg.model_dump())
        cfg.train.hol_country, cfg.train.hol_subdiv = hol_meta

    # 1) Horizons
    hs = parse_horizons_opt(req.horizons, cfg.train.horizons)

    # 2) Load + features
    df = _load_df_from_source(cfg, req, csv_bytes)

    # 3) Scope selection
    c = cfg.columns
    df_scope = select_rows(
        df,
        req.scope,  # type: ignore[arg-type]
        c,
        store_id=req.store_id,
        item_id=req.item_id,
        n_days=req.n_days,
        since_date=req.since_date,
        at_date=req.at_date,
    )

    # 3.5) Decide how to handle future controls
    models_need_future = _models_require_future_controls(cfg.paths.models_dir, hs)

    # Always allow safe fallback: if a plan isn't provided, fill from current values.
    # If models *do* expect controls and the user didn't provide a plan, we note it.
    fill_missing_with_current = True
    auto_fill_note: Optional[str] = None
    if models_need_future and not (req.price_future or req.promo_future):
        auto_fill_note = (
            "Models expect future price/promo controls; no plan provided → "
            "used assume-no-change (copied current price/promo)."
        )

    # 4) Predict
    preds, diag_notes = _predict_for_horizons(
        df_scope,
        cfg,
        hs,
        price_future=req.price_future,
        promo_future=req.promo_future,
        unit_type=unit_type,
        decimal_places=decimal_places,
        fill_missing_with_current=fill_missing_with_current,
    )

    # 5) Paginate
    page_df, meta = paginate_df(preds, req.page, req.page_size)

    # 6) Response payload
    payload = [
        PredictionRow(
            store_id=str(r.store_id),
            item_id=str(r.item_id),
            base_date=r.base_date,
            target_date=r.target_date,
            horizon=int(r.horizon),
            sales=float(r.sales),
        )
        for r in page_df.itertuples(index=False)
    ]

    # Helpful note when empty or when we auto-filled
    note: Optional[str] = None
    if preds.empty:
        base = "No eligible rows to score for the requested scope/horizons."
        if diag_notes:
            base += " Details: " + "; ".join(diag_notes)
        if auto_fill_note:
            base += " " + auto_fill_note
        note = base
    elif auto_fill_note:
        note = auto_fill_note

    return ForecastResponse(
        predictions=payload,
        page=PageMeta(total=meta["total_rows"], page=meta["page"], page_size=meta["page_size"]),
        unit_type=unit_type,
        decimal_places=decimal_places,
        notes=note,
    )


def forecast_from_csv_bytes(cfg: AppConfig, csv_bytes: bytes, req: ForecastRequest) -> ForecastResponse:
    return forecast(cfg, req, csv_bytes=csv_bytes)
