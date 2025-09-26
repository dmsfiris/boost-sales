# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _align_numeric(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align by index and coerce to numeric; drop rows where either is NaN."""
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    yt, yp = yt.align(yp, join="inner")
    mask = yt.notna() & yp.notna()
    return yt[mask], yp[mask]


def rmse(y_true: pd.Series, y_pred: pd.Series, sample_weight: Optional[pd.Series] = None) -> float:
    yt, yp = _align_numeric(y_true, y_pred)
    if yt.empty:
        return float("nan")
    err2 = (yt - yp) ** 2
    if sample_weight is not None:
        w = pd.to_numeric(sample_weight, errors="coerce").reindex(yt.index).fillna(0)
        denom = w.sum()
        if denom <= 0:
            return float("nan")
        return float(np.sqrt((err2 * w).sum() / denom))
    return float(np.sqrt(err2.mean()))


def mape(
    y_true: pd.Series,
    y_pred: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    eps: float = 1e-8,
) -> float:
    """
    Mean Absolute Percentage Error.
    Ignores rows where |y_true| <= eps to avoid div-by-zero explosions.
    """
    yt, yp = _align_numeric(y_true, y_pred)
    if yt.empty:
        return float("nan")
    denom = yt.abs().clip(lower=eps)
    ape = (yt - yp).abs() / denom
    if sample_weight is not None:
        w = pd.to_numeric(sample_weight, errors="coerce").reindex(ape.index).fillna(0)
        denom_w = w.sum()
        if denom_w <= 0:
            return float("nan")
        return float((ape * w).sum() / denom_w)
    return float(ape.mean())


def smape(
    y_true: pd.Series,
    y_pred: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    eps: float = 1e-8,
) -> float:
    """
    Symmetric MAPE: mean( 2*|y - yhat| / (|y| + |yhat|) ).
    """
    yt, yp = _align_numeric(y_true, y_pred)
    if yt.empty:
        return float("nan")
    denom = (yt.abs() + yp.abs()).clip(lower=eps)
    s = 2.0 * (yt - yp).abs() / denom
    if sample_weight is not None:
        w = pd.to_numeric(sample_weight, errors="coerce").reindex(s.index).fillna(0)
        denom_w = w.sum()
        if denom_w <= 0:
            return float("nan")
        return float((s * w).sum() / denom_w)
    return float(s.mean())
