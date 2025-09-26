from __future__ import annotations

from typing import Tuple

import pandas as pd


def chronological_split(df: pd.DataFrame, cutoff: str, date_col: str = "date") -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.to_datetime(cutoff)
    return df[df[date_col] <= cutoff].copy(), df[df[date_col] > cutoff].copy()
