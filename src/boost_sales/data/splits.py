from __future__ import annotations
import pandas as pd
from typing import Tuple

def chronological_split(df: pd.DataFrame, cutoff: str, date_col: str = "date") -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.to_datetime(cutoff)
    return df[df[date_col] <= cutoff].copy(), df[df[date_col] > cutoff].copy()
