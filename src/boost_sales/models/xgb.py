# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd
import xgboost as xgb


__all__ = [
    "train_one_horizon",
    "save_booster",
    "load_booster",
    "predict_one_horizon",
]


# -----------------------------
# Utilities
# -----------------------------
def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Return a copy with `date_col` ensured to be datetime (errors coerced)."""
    if date_col not in df.columns:
        raise KeyError(f"Missing date column: {date_col!r}")
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out


def _maybe_build_target(
    df: pd.DataFrame,
    *,
    horizon: int,
    base_target: str,
    target_col: str,
    date_col: str,
    store_col: str,
    item_col: str,
) -> pd.DataFrame:
    """
    If `target_col` doesn't exist, create it as a forward shift of `base_target`
    within (store,item) groups.
    """
    if target_col in df.columns:
        return df
    out = df.copy()
    out.sort_values([store_col, item_col, date_col], inplace=True, kind="stable")
    out[target_col] = (
        out.groupby([store_col, item_col], sort=False)[base_target]
        .shift(-int(horizon))
        .astype("float64")
    )
    return out


def _select_features(
    df: pd.DataFrame,
    *,
    date_col: str,
    store_col: str,
    item_col: str,
    base_target: str,
    target_col: str,
) -> List[str]:
    """Numeric feature columns excluding identifiers & targets."""
    exclude = {date_col, store_col, item_col, base_target, target_col}
    feats = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feats:
        raise ValueError("No numeric features found after exclusions.")
    return feats


def _train_valid_split_by_date(
    df: pd.DataFrame,
    date_col: str,
    valid_cutoff_date: Optional[str | pd.Timestamp],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Split by date: <= cutoff becomes train, > cutoff becomes valid.
    If no cutoff or no valid rows, return (df, None).
    """
    if valid_cutoff_date is None:
        return df, None
    cutoff = pd.to_datetime(valid_cutoff_date)
    train_df = df[df[date_col] <= cutoff].copy()
    valid_df = df[df[date_col] > cutoff].copy()
    if len(valid_df) == 0:
        return df, None
    return train_df, valid_df


def _prepare_dmatrix(
    df: pd.DataFrame,
    feats: Sequence[str],
    target_col: Optional[str] = None,
) -> xgb.DMatrix:
    """
    Build a DMatrix with consistent feature order. Cast to float32 for compactness.
    """
    X = df[feats].astype("float32", copy=False)
    if target_col is None:
        return xgb.DMatrix(X, feature_names=list(feats))
    y = df[target_col].astype("float32", copy=False)
    return xgb.DMatrix(X, label=y, feature_names=list(feats))


# -----------------------------
# Train / Save / Load
# -----------------------------
def train_one_horizon(
    df: pd.DataFrame,
    horizon: int,
    params: dict,
    *,
    date_col: str = "date",
    store_col: str = "store_id",
    item_col: str = "item_id",
    base_target: str = "sales",
    required_feature_notna: Optional[Sequence[str]] = None,
    valid_cutoff_date: Optional[str | pd.Timestamp] = None,
    early_stopping_rounds: Optional[int] = None,
    verbose_eval: int | bool = 0,
) -> Tuple[xgb.Booster, List[str]]:
    """
    Train a single XGB model for horizon `horizon` and return (booster, feature_order).
    """
    df = _ensure_datetime(df, date_col)

    # 1) Build target if needed
    target_col = f"{base_target}_h{horizon}"
    df = _maybe_build_target(
        df,
        horizon=horizon,
        base_target=base_target,
        target_col=target_col,
        date_col=date_col,
        store_col=store_col,
        item_col=item_col,
    )
    if target_col not in df.columns:
        raise KeyError(f"Expected target column '{target_col}' not found in dataframe.")

    # 2) Feature selection
    feats = _select_features(
        df,
        date_col=date_col,
        store_col=store_col,
        item_col=item_col,
        base_target=base_target,
        target_col=target_col,
    )

    # 3) Sort + clean
    df = df.sort_values([store_col, item_col, date_col], kind="stable").copy()

    if required_feature_notna:
        req = [c for c in required_feature_notna if c in df.columns]
        if req:
            df = df.dropna(subset=req)

    # Drop rows with NaNs in features or target
    df = df.dropna(subset=list(feats) + [target_col])
    if df.empty:
        raise ValueError("No training rows after dropping NaNs for features/target.")

    # 4) Train/valid split
    train_df, valid_df = _train_valid_split_by_date(df, date_col, valid_cutoff_date)

    # 5) DMatrix
    dtrain = _prepare_dmatrix(train_df, feats, target_col)
    evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]

    use_valid = valid_df is not None and len(valid_df) > 0
    if use_valid:
        dvalid = _prepare_dmatrix(valid_df, feats, target_col)
        evals.append((dvalid, "valid"))

    # 6) Params: accept sklearn-style and xgb-style keys
    params = dict(params) if params is not None else {}
    if "eta" not in params and "learning_rate" in params:
        params["eta"] = params.pop("learning_rate")
    num_boost_round = int(params.pop("n_estimators", 600))
    params.pop("verbose_eval", None)  # not an xgb core param

    # 7) Train
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if use_valid else None,
        verbose_eval=bool(verbose_eval),
    )
    return booster, list(feats)


def save_booster(
    booster: xgb.Booster,
    feature_order: Sequence[str],
    outdir_or_models_dir: str | Path,
    horizon: int,
) -> None:
    """Save model and feature order for a given horizon to models_dir."""
    outdir = Path(outdir_or_models_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"model_h{horizon}.xgb.json"
    feats_path = outdir / f"feature_order_h{horizon}.json"
    booster.save_model(str(model_path))
    with open(feats_path, "w", encoding="utf-8") as f:
        json.dump(list(feature_order), f, ensure_ascii=False)


def load_booster(
    models_dir: str | Path,
    horizon: int,
) -> Tuple[xgb.Booster, List[str]]:
    """Load model and feature order for the given horizon."""
    models_dir = Path(models_dir)
    model_path = models_dir / f"model_h{horizon}.xgb.json"
    feats_path = models_dir / f"feature_order_h{horizon}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"Feature order file not found: {feats_path}")
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    with open(feats_path, "r", encoding="utf-8") as f:
        feature_order = json.load(f)
    return booster, list(feature_order)


# -----------------------------
# Predict
# -----------------------------
def predict_one_horizon(
    df: pd.DataFrame,
    *,
    booster: xgb.Booster,
    feature_order: Sequence[str],
    date_col: str = "date",
    store_col: str = "store_id",
    item_col: str = "item_id",
    base_target: str = "sales",
    required_feature_notna: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Predict using the exact training feature order. This is STRICT:
      - All training features (except identifiers/date/target) must be present.
      - Rows with NaNs in required features are dropped; if none remain, raise.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    # Use the exact feature list the model was trained with (minus ids/targets).
    drop = {date_col, store_col, item_col, base_target}
    feats = [f for f in feature_order if f not in drop]

    # Presence check (fail fast if training features are missing)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required features for prediction: {missing[:6]}{'...' if len(missing) > 6 else ''}"
        )

    X = df.copy()

    # Enforce additional not-na constraints if provided
    if required_feature_notna:
        req = [c for c in required_feature_notna if c in X.columns]
        if req:
            X = X.dropna(subset=req)

    # Drop rows with NaNs in features
    X = X.dropna(subset=feats)
    if X.empty:
        raise ValueError("No rows left after dropping NaNs in required features.")

    dmat = _prepare_dmatrix(X, feats)
    return booster.predict(dmat)
