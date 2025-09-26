# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from boost_sales.config import AppConfig
from boost_sales.features.future_controls import (
    add_future_controls as _add_future_controls,
)


def add_horizon_controls(
    df: pd.DataFrame,
    *,
    horizon: int,
    cfg: AppConfig,
    date_col: Optional[str] = None,
    price_col: Optional[str] = None,
    promo_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build horizon-specific control columns using the canonical feature builder.

    Notes
    -----
    - This simply forwards AppConfig.future flags to the underlying implementation.
      If models were trained WITHOUT future controls (typical), keep:
          cfg.future.add_price_future = False
          cfg.future.add_promo_future = False
          cfg.future.add_price_ratio  = False
      to preserve training parity.
    - At inference, you can still inject `price_h{h}` / `promo_h{h}`; scoring will
      subset to the booster’s saved `feats`, so extra columns are harmless.
    """
    c, f = cfg.columns, cfg.future
    return _add_future_controls(
        df,
        horizon,
        date_col=(date_col or c.date),
        price_col=(price_col or c.price),
        promo_col=(promo_col or c.promo),
        denom_col=f.denom_col,
        add_price_future=f.add_price_future,
        add_promo_future=f.add_promo_future,
        add_price_ratio=f.add_price_ratio,
        safe_zero_denominator=f.safe_zero_denominator,
        # IMPORTANT: group-aware shifts so we don't leak across pairs
        group_cols=(c.store, c.item),
    )


def assume_no_change_fill(
    dfh: pd.DataFrame,
    *,
    horizon: int,
    cfg: AppConfig,
    price_col: Optional[str] = None,
    promo_col: Optional[str] = None,
    denom_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fill missing price_h{h}/promo_h{h} from current price/promo and (optionally)
    compute price_ratio_h{h} safely.

    Behavior
    --------
    - price_h{h}: copies current price when missing.
    - promo_h{h}: copies current promo when missing; values are coerced to float and
      clipped into [0, 1] so both binary (0/1) and intensity (e.g., 0.25, 0.5) are safe.
    - price_ratio_h{h}: recomputed when denom is present (respects safe_zero_denominator).

    Returns
    -------
    A NEW dataframe copy (no in-place mutation).

    When to use
    -----------
    - Keep for legacy flows or when models do NOT require explicit future controls.
      If models were trained expecting price_h*/promo_h*, prefer passing explicit plans.
    """
    c, f = cfg.columns, cfg.future

    price_col = price_col or c.price
    promo_col = promo_col or c.promo
    denom_col = denom_col or f.denom_col

    out = dfh.copy()

    ph = f"price_h{horizon}"
    pmh = f"promo_h{horizon}"
    prh = f"price_ratio_h{horizon}"

    # ---- price_h{h} ----
    if ph not in out.columns:
        out[ph] = np.nan
    out[ph] = pd.to_numeric(out[ph], errors="coerce").astype("float64", copy=False)

    need_ph = out[ph].isna()
    if need_ph.any() and price_col in out.columns:
        out.loc[need_ph, ph] = pd.to_numeric(out.loc[need_ph, price_col], errors="coerce").astype("float64")

    # ---- promo_h{h} ----
    if pmh not in out.columns:
        out[pmh] = np.nan
    out[pmh] = pd.to_numeric(out[pmh], errors="coerce").astype("float64", copy=False)

    need_pmh = out[pmh].isna()
    if need_pmh.any() and promo_col in out.columns:
        # Coerce current promo to float; support both binary and fractional intensities.
        cur_promo = pd.to_numeric(out.loc[need_pmh, promo_col], errors="coerce").astype("float64")
        out.loc[need_pmh, pmh] = cur_promo

    # Always clip promo intensity into [0,1]
    out[pmh] = out[pmh].clip(lower=0.0, upper=1.0)

    # ---- price_ratio_h{h} ----
    if denom_col in out.columns:
        if prh not in out.columns:
            out[prh] = np.nan
        out[prh] = pd.to_numeric(out[prh], errors="coerce").astype("float64", copy=False)

        num = pd.to_numeric(out[ph], errors="coerce").astype("float64")
        den = pd.to_numeric(out[denom_col], errors="coerce").astype("float64")

        mask = den.notna() & num.notna()
        if f.safe_zero_denominator:
            mask &= den.ne(0)

        out.loc[mask, prh] = (num[mask] / den[mask]).astype("float64")

    return out
