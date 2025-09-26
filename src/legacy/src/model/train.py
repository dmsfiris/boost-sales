# src/model/train.py
"""
Train 7 separate XGBoost regression models to forecast sales at horizons H=1..7.
- Time-t features: calendar, lags, rolling stats, price_roll_28 (no future leakage)
- Future controls at t+h: price_h{h}, promo_h{h}, price_ratio_h{h}
- Holiday flag at t+h: is_hol_h{h} using country/subdivision set at TRAIN time

Saves to ./models:
  - model_h{h}.xgb.json         (XGBoost booster per horizon)
  - feature_order_h{h}.json     (column order per horizon)
  - categories.json             (store/item categories for serving one-hots)
  - holidays_meta.json          ({"country": "...", "subdiv": "..."})
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# ----------------------------- Config -----------------------------

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Horizons (days ahead)
H_LIST = [1, 2, 3, 4, 5, 6, 7]

# Holiday configuration via environment variables (LOCKED at training)
HOL_COUNTRY = os.getenv("HOL_COUNTRY", "GR")  # e.g. "US", "GB", "DE", "GR", ...
HOL_SUBDIV  = os.getenv("HOL_SUBDIV", "")     # e.g. "CA" (US-California), "" for none


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
      - rolling stats: roll_mean_{7,28}, roll_std_7  (all based on past)
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

    # price history baseline
    df["price_roll_28"] = g["price"].shift(1).rolling(28).mean()

    return df


# ----------------------------- Training per horizon -----------------------------

def train_one_h(df: pd.DataFrame, h: int, HOL) -> None:
    """
    Build design matrix for horizon h and train XGBoost:
      - Target: log1p(sales at t+h)
      - Future controls aligned to t+h: price_h{h}, promo_h{h}, price_ratio_h{h}
      - Holiday flag for target day: is_hol_h{h}
      - One-hot IDs (works here because IDs are small)
    """
    tmp = df.copy()

    # target at t+h
    y = np.log1p(tmp["sales"].shift(-h))

    # future controls (known/planned at t+h)
    tmp[f"price_h{h}"] = tmp["price"].shift(-h)
    tmp[f"promo_h{h}"] = tmp["promo"].shift(-h)
    tmp[f"price_ratio_h{h}"] = tmp[f"price_h{h}"] / (tmp["price_roll_28"] + 1e-9)

    # holiday flag for t+h
    if HOL is not None:
        tmp[f"is_hol_h{h}"] = (tmp["date"] + pd.to_timedelta(h, "D")).apply(lambda d: int(d in HOL))
    else:
        tmp[f"is_hol_h{h}"] = 0

    # one-hot IDs (cast to str to stay consistent)
    ids = pd.get_dummies(tmp[["store_id", "item_id"]].astype(str), prefix=["store_id", "item_id"])

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

    # time-based split: last 60 days as validation
    dates = pd.to_datetime(tmp.loc[X.index, "date"])
    split_day = dates.max() - pd.Timedelta(days=60)
    tr_idx = dates <= split_day
    va_idx = dates > split_day

    dtr = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
    dva = xgb.DMatrix(X[va_idx], label=y[va_idx])

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

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=800,
        evals=[(dtr, "train"), (dva, "val")],
        verbose_eval=False,
    )

    # save artifacts for this horizon
    bst.save_model(str(MODELS_DIR / f"model_h{h}.xgb.json"))
    (MODELS_DIR / f"feature_order_h{h}.json").write_text(json.dumps(list(X.columns)))


# --------------------------------- Entry point ---------------------------------

def main(csv_path: str = "data/sales.csv") -> None:
    # Load data
    df = pd.read_csv(csv_path)

    # Validate required columns
    required = {"date", "store_id", "item_id", "price", "promo", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    # Feature engineering
    df = add_features(df)

    # Persist categories for serving (one-hot reconstruction)
    cats = {
        "store_ids": sorted(df["store_id"].astype(str).unique().tolist()),
        "item_ids":  sorted(df["item_id"].astype(str).unique().tolist()),
    }
    (MODELS_DIR / "categories.json").write_text(json.dumps(cats))

    # Build holiday calendar (locked to training)
    HOL = get_holidays()
    if HOL is None:
        print(f"[train] Holidays unavailable/disabled for {HOL_COUNTRY}"
              f"{('-' + HOL_SUBDIV) if HOL_SUBDIV else ''}")
    else:
        print(f"[train] Using holidays for {HOL_COUNTRY}"
              f"{('-' + HOL_SUBDIV) if HOL_SUBDIV else ''}")

    # Persist holiday meta so serving matches training
    (MODELS_DIR / "holidays_meta.json").write_text(
        json.dumps({"country": HOL_COUNTRY, "subdiv": HOL_SUBDIV})
    )

    # Train per horizon
    for h in H_LIST:
        print(f"[train] Training horizon h={h} ...")
        train_one_h(df, h, HOL)

    print(f"[train] Done. Trained horizons: {H_LIST}")


if __name__ == "__main__":
    main()
