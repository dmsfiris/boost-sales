# SPDX-License-Identifier: MIT
from __future__ import annotations

import pandas as pd


def add_calendar(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    add_weekofyear: bool = False,
    add_weekofmonth: bool = False,
    add_quarter: bool = True,
    add_year: bool = True,
    add_month: bool = True,
    add_day: bool = True,
    add_dow: bool = True,
    add_is_weekend: bool = True,
) -> pd.DataFrame:
    """
    Add calendar features derived from `date_col` and return a new dataframe.

    Columns (if enabled):
      - year:int16, quarter:int8, month:int8, day:int8
      - dow:int8 (Mon=0..Sun=6), is_weekend:int8 (Sat/Sun=1 else 0)
      - weekofyear:int16 (ISO week number 1..53)
      - weekofmonth:int8  (1 + (day-1)//7 ∈ {1..5})
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: '{date_col}'")

    out = df.copy()

    # Ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    dt = out[date_col].dt

    if add_year and "year" not in out.columns:
        out["year"] = dt.year.astype("int16")

    if add_quarter and "quarter" not in out.columns:
        out["quarter"] = dt.quarter.astype("int8")

    if add_month and "month" not in out.columns:
        out["month"] = dt.month.astype("int8")

    if add_day and "day" not in out.columns:
        out["day"] = dt.day.astype("int8")

    # Compute DOW if either dow or is_weekend is requested
    if (add_dow or add_is_weekend) and "dow" not in out.columns:
        out["dow"] = dt.dayofweek.astype("int8")  # Mon=0..Sun=6

    if add_dow is False and "dow" in out.columns and not add_is_weekend:
        # If caller didn't ask for dow and it was pre-existing, we leave it;
        # if we created it only for is_weekend we still keep it (cheap & useful).
        pass

    if add_is_weekend and "is_weekend" not in out.columns:
        out["is_weekend"] = (out["dow"] >= 5).astype("int8")

    if add_weekofyear and "weekofyear" not in out.columns:
        # pandas 2.x returns UInt32; cast down for compactness
        out["weekofyear"] = dt.isocalendar().week.astype("int16")

    if add_weekofmonth and "weekofmonth" not in out.columns:
        # Simple & version-stable definition
        out["weekofmonth"] = (1 + (dt.day - 1) // 7).astype("int8")

    return out
