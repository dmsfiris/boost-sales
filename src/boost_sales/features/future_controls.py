# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def add_future_controls(
    df: pd.DataFrame,
    horizon: int,
    *,
    date_col: str,
    price_col: Optional[str] = None,
    promo_col: Optional[str] = None,
    denom_col: Optional[str] = None,
    add_price_future: bool = True,
    add_promo_future: bool = True,
    add_price_ratio: bool = True,
    safe_zero_denominator: bool = True,
    group_cols: Optional[Sequence[str]] = None,   # <- allow grouped shifts
) -> pd.DataFrame:
    """
    Build horizon-specific exogenous controls:

      - price_h{horizon} : future price shifted by -horizon (optionally per group)
      - promo_h{horizon} : future promo (flag/value) shifted by -horizon (optionally per group)
      - price_ratio_h{h} : price_h{h} / denom_col (e.g., rolling price) if requested

    Notes
    -----
    * Shifts can be done globally or per group (store/item) if group_cols provided.
    * Pure Pandas; no dependency on AppConfig, FastAPI, or Typer.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    ph = f"price_h{horizon}"
    pmh = f"promo_h{horizon}"
    prh = f"price_ratio_h{horizon}"

    def _shift(series: pd.Series) -> pd.Series:
        """Shift by -horizon globally or per group if all group_cols are present."""
        if group_cols:
            cols = [c for c in group_cols if c in out.columns]
            if len(cols) == len(group_cols):
                # IMPORTANT: for Series.groupby use a list of 1-D keys, not a DataFrame
                keys = [out[c] for c in cols]
                return series.groupby(keys, sort=False).shift(-horizon)
        return series.shift(-horizon)

    # price_h{h}
    if add_price_future and price_col and price_col in out.columns:
        base_price = pd.to_numeric(out[price_col], errors="coerce")
        out[ph] = _shift(base_price).astype("float64")

    # promo_h{h}
    if add_promo_future and promo_col and promo_col in out.columns:
        base_promo = pd.to_numeric(out[promo_col], errors="coerce")
        out[pmh] = _shift(base_promo).astype("float64")

    # price_ratio_h{h}
    if add_price_ratio and denom_col and denom_col in out.columns and ph in out.columns:
        num = pd.to_numeric(out[ph], errors="coerce").astype("float64")
        den = pd.to_numeric(out[denom_col], errors="coerce").astype("float64")
        safe_mask = den.notna() & num.notna()
        if safe_zero_denominator:
            safe_mask &= den.ne(0)
        out[prh] = np.nan
        out.loc[safe_mask, prh] = (num[safe_mask] / den[safe_mask]).astype("float64")

    return out
