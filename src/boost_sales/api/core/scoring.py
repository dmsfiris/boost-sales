# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from boost_sales.api.core.controls import add_horizon_controls, assume_no_change_fill
from boost_sales.config import AppConfig
from boost_sales.models.xgb import load_booster

_OUT_COLS = ["store_id", "item_id", "base_date", "target_date", "horizon", "sales"]


def _empty_out() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUT_COLS)


def score_prepared(dfh: pd.DataFrame, h: int, cfg: AppConfig) -> pd.DataFrame:
    """
    Score a DataFrame that already has horizon controls and (optionally) assume-no-change fills.
    Returns a standardized dataframe with columns:
      store_id, item_id, base_date, target_date, horizon, sales
    """
    c = cfg.columns
    booster, feats = load_booster(cfg.paths.models_dir, h)

    # Guard: required feature columns must exist
    if not set(feats).issubset(dfh.columns):
        return _empty_out()

    # Keep only rows with all required features present
    ok = dfh[feats].notna().all(axis=1)
    ready = dfh.loc[ok].copy()
    if ready.empty:
        return _empty_out()

    # Local import to avoid a hard dependency at import-time
    import xgboost as xgb  # type: ignore

    dmat = xgb.DMatrix(ready[feats], feature_names=feats)
    yhat = booster.predict(dmat)

    return pd.DataFrame(
        {
            "store_id": ready[c.store].astype("string"),
            "item_id": ready[c.item].astype("string"),
            "base_date": pd.to_datetime(ready[c.date]).dt.date,
            "target_date": (pd.to_datetime(ready[c.date]) + pd.to_timedelta(h, unit="D")).dt.date,
            "horizon": h,
            "sales": pd.Series(yhat, dtype="float64").values,
        }
    )


def score_many_horizons(
    df_base: pd.DataFrame,
    horizons: Iterable[int],
    cfg: AppConfig,
    *,
    assume_no_change: bool,
) -> List[pd.DataFrame]:
    """
    For each horizon: add controls, optional no-change fill, then score.
    Returns a list of per-horizon prediction dataframes (possibly empty for some horizons).
    Output column is 'sales'.
    """
    c = cfg.columns
    preds_all: List[pd.DataFrame] = []

    for h in horizons:
        dfh = add_horizon_controls(
            df_base.copy(),
            horizon=h,
            cfg=cfg,
            date_col=c.date,
            price_col=c.price,
            promo_col=c.promo,
        )

        if assume_no_change:
            dfh = assume_no_change_fill(dfh, horizon=h, cfg=cfg)

        out = score_prepared(dfh, h, cfg)
        if not out.empty:
            preds_all.append(out)

    return preds_all


# -----------------------------
# Optional post-processing helper
# -----------------------------
def format_sales_column(
    df: pd.DataFrame,
    cfg: AppConfig,
    *,
    unit_type: Optional[str] = None,  # "integer" | "float" | None -> cfg.output.unit_type
    decimal_places: Optional[int] = None,  # None -> cfg.output.decimal_places
) -> pd.DataFrame:
    """
    Return a copy of df with 'sales' formatted according to unit settings.

    - If unit_type == "integer": rounds to nearest integer and (for API safety) stores as float64.
    - Else (float): casts to float64 and rounds to `decimal_places` (default from cfg.output).

    Notes:
    - Expects a 'sales' column.
    - Side-effect free (returns a new dataframe).
    """
    out = df.copy()

    if "sales" not in out.columns:
        return out  # nothing to format

    # Resolve effective settings from cfg.output with request overrides
    try:
        eff_unit = (unit_type or cfg.output.unit_type).lower()
        eff_dp = decimal_places if decimal_places is not None else cfg.output.decimal_places
    except Exception:
        # If cfg.output isn't present yet, default to float with 2 decimals
        eff_unit = (unit_type or "float").lower()
        eff_dp = decimal_places if decimal_places is not None else 2

    out["sales"] = pd.to_numeric(out["sales"], errors="coerce")

    if eff_unit == "integer":
        # Keep schema compatibility (some APIs expect float)
        out["sales"] = np.rint(out["sales"]).astype("int64").astype("float64", copy=False)
    else:
        out["sales"] = out["sales"].astype("float64", copy=False).round(int(eff_dp))

    return out
