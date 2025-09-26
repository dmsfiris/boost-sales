# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Optional

import pandas as pd


def add_holidays(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    country: str = "US",
    subdiv: Optional[str] = None,
    out_col: str = "is_hol",
    # ---- legacy-compat kwargs (safe no-ops) ----
    flag_name: Optional[str] = None,
    expand_alt_names: bool = False,  # kept for API compatibility; no effect here
) -> pd.DataFrame:
    """
    Add a binary holiday flag column.

    - If `holidays` package is unavailable or the region is unknown, fills zeros.
    - Uses `flag_name` if provided (legacy), otherwise `out_col`.
    - Returns a new dataframe (does not mutate `df`).
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: '{date_col}'")

    out = df.copy()

    # Ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    col = flag_name or out_col

    try:
        import holidays as hol
    except Exception:
        out[col] = 0
        return out

    try:
        holset = hol.country_holidays(country=country, subdiv=subdiv)
        hol_dates = set(holset.keys())
        out[col] = out[date_col].dt.date.isin(hol_dates).astype("int8")
    except Exception:
        out[col] = 0

    return out
