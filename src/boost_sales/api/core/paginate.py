# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd


def paginate_df(
    df: pd.DataFrame,
    page: Optional[int],
    page_size: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Return (df_slice, meta) where meta includes page, page_size, total_rows, total_pages.
    This stays non-Pydantic to keep core utils decoupled from API models.
    """
    # Defaults and bounds to match API schema (page_size <= 10_000)
    page = 1 if (page is None or page < 1) else int(page)
    if page_size is None or page_size <= 0:
        page_size = 100
    else:
        page_size = int(min(page_size, 10_000))

    total_rows = int(len(df))
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    # Clamp page into valid range
    if page > total_pages:
        page = total_pages

    start = (page - 1) * page_size
    end = start + page_size

    sliced = df.iloc[start:end].copy()
    meta: Dict[str, int] = {
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
    }
    return sliced, meta
