# src/model/train_per_group.py
"""
Train XGBoost models per group:
  - scope='pair'  : specific (store_id, item_id)
  - scope='item'  : all stores for a given item_id
  - scope='store' : all items for a given store_id

Artifacts are saved under:
  models/by_pair/{store}__{item}/
  models/by_item/{item}/
  models/by_store/{store}/

Each directory contains:
  - model_h{h}.xgb.json
  - feature_order_h{h}.json
  - categories.json       (only IDs present in this group)
  - holidays_meta.json
  - manifest.json         (metadata + simple validation metrics)

Public helpers (imported by serve_model.py):
  - add_features(df)
  - get_holidays()
  - weekly_cov(df_sub)
  - group_qualifies(df_sub, min_days, min_price_variety, max_weekly_cov)
  - train_group_models(df_full_features, group_filter, scope, HOL, holdout_days, warm_start)
"""

from pathlib import Path
import os
import json
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

# ----------------------------- Config -----------------------------

MODELS_DIR = Path("models")
H_LIST = [1, 2, 3, 4, 5, 6, 7]

# Holiday configuration via environment variables (LOCKED at training)
HOL_COUNTRY = os.getenv("HOL_COUNTRY", "GR")
HOL_SUBDIV  = os.getenv("HOL_SUBDIV", "")


# ---------------------- Holidays ----------------------

def get_holidays():
    """Return a holidays object for (HOL_COUNTRY, HOL_SUBDIV) or None if unavailable."""
    try:
        import holidays
        if HOL_SUBDIV:
            return holidays.country_holidays(HOL_COUNTRY, subdiv=HOL_SUBDIV)
        return holidays.country_holidays(HOL_COUNTRY)
    except Exception:
        return None


# ---------------------- Feature engineering (time t) ----------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute base time-t features per (store_id, item_id):
      - calendar: dow, month, is_weekend
      - lags: sales_lag_{1,7,14,28}
      - rolling stats: roll_mean_{7,28}, roll_std_7 (based on past)
      - price_roll_28: 28d rolling mean of price (based on past)
    """
    df = df.copy()
    df["store_id"] = df["store_id"].astype(str)
    df["item_id"]  = df["item_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "item_id", "date"])

    # calendar
    df["dow"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    g = df.groupby(["store_id", "item_id"], group_keys=False)

    # lags
    for L in (1, 7, 14, 28):
        df[f"sales_lag_{L}"] = g["sales"].shift(L)

    # rolling stats (use shift(1) to avoid current-day leakage)
    df["roll_mean_7"]  = g["sales"].shift(1).rolling(7).mean()
    df["roll_mean_28"] = g["sales"].shift(1).rolling(28).mean()
    df["roll_std_7"]   = g["sales"].shift(1).rolling(7).std()

    # price baseline
    df["price_roll_28"] = g["price"].shift(1).rolling(28).mean()

    return df


# ---------------------- Quality screens / stats ----------------------

def weekly_cov(df_sub: pd.DataFrame) -> float:
    """
    Coefficient of variation of weekly-summed sales.
    Accepts raw or feature-augmented df (needs 'date' and 'sales').
    """
    if df_sub.empty:
        return float("inf")
    tmp = df_sub.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    w = tmp.set_index("date")["sales"].resample("W-SUN").sum()
    m = w.mean()
    s = w.std()
    if m is None or m <= 0 or np.isnan(m):
        return float("inf")
    return float((s or 0) / m)


def group_qualifies(
    df_sub: pd.DataFrame,
    min_days: int = 200,
    min_price_variety: int = 10,
    max_weekly_cov: float = 2.0,
) -> bool:
    """
    Basic thresholds to avoid training on very sparse/noisy groups.
    """
    if df_sub.empty:
        return False
    n_days = int(df_sub.shape[0])
    variety = int(df_sub["price"].nunique()) if "price" in df_sub else 0
    cov = weekly_cov(df_sub)
    return (n_days >= min_days) and (variety >= min_price_variety) and (cov <= max_weekly_cov)


# ---------------------- Internal helpers ----------------------

def _filter_group(df_feat: pd.DataFrame, scope: str, group_filter: Dict[str, str]) -> pd.DataFrame:
    scope = scope.lower().strip()
    if scope == "pair":
        store_id = str(group_filter.get("store_id"))
        item_id  = str(group_filter.get("item_id"))
        sub = df_feat[(df_feat["store_id"] == store_id) & (df_feat["item_id"] == item_id)].copy()
    elif scope == "item":
        item_id = str(group_filter.get("item_id"))
        sub = df_feat[df_feat["item_id"] == item_id].copy()
    elif scope == "store":
        store_id = str(group_filter.get("store_id"))
        sub = df_feat[df_feat["store_id"] == store_id].copy()
    else:
        raise ValueError("scope must be one of: 'pair' | 'item' | 'store'")
    sub = sub.sort_values(["store_id", "item_id", "date"])
    return sub


def _models_dir_for(scope: str, group_filter: Dict[str, str]) -> Path:
    scope = scope.lower().strip()
    base = MODELS_DIR
    if scope == "pair":
        sid = str(group_filter["store_id"]).replace("/", "_")
        iid = str(group_filter["item_id"]).replace("/", "_")
        return base / "by_pair" / f"{sid}__{iid}"
    if scope == "item":
        iid = str(group_filter["item_id"]).replace("/", "_")
        return base / "by_item" / f"{iid}"
    if scope == "store":
        sid = str(group_filter["store_id"]).replace("/", "_")
        return base / "by_store" / f"{sid}"
    raise ValueError("invalid scope")


def _save_manifest(out_dir: Path, scope: str, group_filter: Dict[str, str], cols_by_h: Dict[int, List[str]],
                   val_metrics: Dict[int, Dict[str, float]], rows: int) -> None:
    manifest = {
        "scope": scope,
        "group_filter": group_filter,
        "horizons": H_LIST,
        "features_per_h": {str(h): cols_by_h[h] for h in H_LIST},
        "val_metrics": val_metrics,
        "rows_used": rows,
        "holidays_meta": {"country": HOL_COUNTRY, "subdiv": HOL_SUBDIV},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _one_hot_ids(df_ids: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode IDs present in the filtered group."""
    return pd.get_dummies(df_ids.astype(str), prefix=["store_id", "item_id"])


# ---------------------- Training per group ----------------------

def _build_matrix_for_h(df_sub: pd.DataFrame, h: int, HOL) -> Tuple[pd.DataFrame, pd.Series]:
    tmp = df_sub.copy()
    # target at t+h
    y = np.log1p(tmp["sales"].shift(-h))

    # future controls at t+h (no leakage)
    tmp[f"price_h{h}"] = tmp["price"].shift(-h)
    tmp[f"promo_h{h}"] = tmp["promo"].shift(-h)
    tmp[f"price_ratio_h{h}"] = tmp[f"price_h{h}"] / (tmp["price_roll_28"] + 1e-9)

    # holiday flag for target day
    if HOL is not None:
        tmp[f"is_hol_h{h}"] = (tmp["date"] + pd.to_timedelta(h, "D")).apply(lambda d: int(d in HOL))
    else:
        tmp[f"is_hol_h{h}"] = 0

    ids = _one_hot_ids(tmp[["store_id", "item_id"]])

    base_cols = [
        "dow", "month", "is_weekend",
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "roll_mean_7", "roll_mean_28", "roll_std_7",
        "price_roll_28",
    ]
    fut_cols = [f"price_h{h}", f"promo_h{h}", f"price_ratio_h{h}", f"is_hol_h{h}"]

    X = pd.concat([ids, tmp[base_cols + fut_cols]], axis=1)

    # valid samples: no NaNs in X or y
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]
    return X, y


def _train_one_h(X: pd.DataFrame, y: pd.Series, holdout_days: int = 60) -> Tuple[xgb.Booster, Dict[str, float]]:
    # time-based split: last N days as validation
    # we need the dates aligned to X index; assume caller keeps 'date' in side df if needed.
    # Here, we approximate with a chronological split on index order (already sorted by date).
    n = X.shape[0]
    split_idx = max(0, n - holdout_days)
    tr_idx = np.arange(n) < split_idx
    va_idx = np.arange(n) >= split_idx

    dtr = xgb.DMatrix(X.iloc[tr_idx], label=y.iloc[tr_idx])
    dva = xgb.DMatrix(X.iloc[va_idx], label=y.iloc[va_idx])

    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=8,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        seed=42,
    )

    num_boost_round = 800 if n >= 2000 else (500 if n >= 800 else 300)

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        evals=[(dtr, "train"), (dva, "val")],
        verbose_eval=False,
    )

    # simple validation metrics on sales scale
    if va_idx.sum() > 0:
        pred_val = np.expm1(bst.predict(dva))
        true_val = np.expm1(y.iloc[va_idx].values)
        rmse = float(np.sqrt(np.mean((pred_val - true_val) ** 2)))
        mae = float(np.mean(np.abs(pred_val - true_val)))
    else:
        rmse = float("nan"); mae = float("nan")

    return bst, {"val_rmse_sales": rmse, "val_mae_sales": mae, "n_train": int(tr_idx.sum()), "n_val": int(va_idx.sum())}


def train_group_models(
    df_full_features: pd.DataFrame,
    group_filter: Dict[str, str],
    scope: str,
    HOL=None,
    holdout_days: int = 60,
    warm_start: bool = True,  # kept for API compatibility; currently no-op due to feature-space mismatch
) -> Dict[str, Any]:
    """
    Train and save models for the given group scope.
    df_full_features: output of add_features() on the entire dataset.
    group_filter:
        - scope='pair'  : {"store_id": "...", "item_id": "..."}
        - scope='item'  : {"item_id": "..."}
        - scope='store' : {"store_id": "..."}
    Returns summary dict with artifact directory and validation metrics.
    """
    scope = scope.lower().strip()
    if scope not in {"pair", "item", "store"}:
        raise ValueError("scope must be one of: 'pair' | 'item' | 'store'")

    sub = _filter_group(df_full_features, scope, group_filter)
    if sub.empty:
        raise ValueError("no rows after filtering for group")

    # Persist categories for this group (IDs present)
    cats = {
        "store_ids": sorted(sub["store_id"].astype(str).unique().tolist()),
        "item_ids":  sorted(sub["item_id"].astype(str).unique().tolist()),
    }

    out_dir = _models_dir_for(scope, group_filter)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Holidays meta
    (out_dir / "holidays_meta.json").write_text(json.dumps({"country": HOL_COUNTRY, "subdiv": HOL_SUBDIV}))
    (out_dir / "categories.json").write_text(json.dumps(cats, indent=2))

    if HOL is None:
        print(f"[train-group] Holidays unavailable/disabled for {HOL_COUNTRY}"
              f"{('-' + HOL_SUBDIV) if HOL_SUBDIV else ''}")
    else:
        print(f"[train-group] Using holidays for {HOL_COUNTRY}"
              f"{('-' + HOL_SUBDIV) if HOL_SUBDIV else ''}")

    if warm_start:
        # Not practical to warm-start from global boosters because feature spaces differ (one-hot sets).
        print("[train-group] warm_start requested, but skipped (feature spaces differ from global).")

    cols_by_h: Dict[int, List[str]] = {}
    val_metrics: Dict[int, Dict[str, float]] = {}

    # Train per horizon
    for h in H_LIST:
        print(f"[train-group] scope={scope} {group_filter} h={h} …")
        X, y = _build_matrix_for_h(sub, h, HOL)

        if X.shape[0] < 50:
            print(f"[warn] h={h} has very few samples (n={X.shape[0]}). Model may underperform.")

        bst, metrics = _train_one_h(X, y, holdout_days=holdout_days)
        cols = list(X.columns)

        # Save artifacts
        model_path = out_dir / f"model_h{h}.xgb.json"
        feat_path  = out_dir / f"feature_order_h{h}.json"
        bst.save_model(str(model_path))
        feat_path.write_text(json.dumps(cols))

        cols_by_h[h] = cols
        val_metrics[h] = metrics

        print(f"[train-group] saved {model_path.name} | val_rmse={metrics['val_rmse_sales']:.4f} "
              f"| n_tr={metrics['n_train']} n_va={metrics['n_val']}")

    # Manifest with quick stats
    _save_manifest(out_dir, scope, group_filter, cols_by_h, val_metrics, rows=int(sub.shape[0]))

    return {
        "scope": scope,
        "group": group_filter,
        "models_dir": str(out_dir),
        "horizons": H_LIST,
        "val_metrics": val_metrics,
        "rows_used": int(sub.shape[0]),
        "categories": cats,
    }


# ---------------------- Optional CLI ----------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train per-group XGBoost models.")
    parser.add_argument("--csv", default="data/sales.csv", help="Path to sales.csv")
    parser.add_argument("--scope", choices=["pair", "item", "store"], required=True)
    parser.add_argument("--store_id")
    parser.add_argument("--item_id")
    parser.add_argument("--holdout_days", type=int, default=60)
    parser.add_argument("--min_days", type=int, default=200)
    parser.add_argument("--min_price_variety", type=int, default=10)
    parser.add_argument("--max_weekly_cov", type=float, default=2.0)
    args = parser.parse_args()

    df0 = pd.read_csv(args.csv)
    df0["store_id"] = df0["store_id"].astype(str)
    df0["item_id"] = df0["item_id"].astype(str)
    df0["date"] = pd.to_datetime(df0["date"])
    df0 = df0.sort_values(["store_id","item_id","date"])

    df_feat = add_features(df0)
    HOL = get_holidays()

    if args.scope == "pair":
        if not args.store_id or not args.item_id:
            raise SystemExit("--scope pair requires --store_id and --item_id")
        gf = {"store_id": args.store_id, "item_id": args.item_id}
        sub = _filter_group(df_feat, "pair", gf)
    elif args.scope == "item":
        if not args.item_id:
            raise SystemExit("--scope item requires --item_id")
        gf = {"item_id": args.item_id}
        sub = _filter_group(df_feat, "item", gf)
    else:
        if not args.store_id:
            raise SystemExit("--scope store requires --store_id")
        gf = {"store_id": args.store_id}
        sub = _filter_group(df_feat, "store", gf)

    if not group_qualifies(sub, args.min_days, args.min_price_variety, args.max_weekly_cov):
        print("[train-group] group did not meet thresholds; aborting.")
        raise SystemExit(2)

    res = train_group_models(
        df_full_features=df_feat,
        group_filter=gf,
        scope=args.scope,
        HOL=HOL,
        holdout_days=args.holdout_days,
        warm_start=False,
    )
    print(json.dumps(res, indent=2))
