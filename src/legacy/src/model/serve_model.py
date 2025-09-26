# src/model/serve_model.py
"""
Sales forecast & price optimization with:
- cost_mode: "manual" | "replacement" | "wac" | "standard"
- quantity_mode: "int" | "float"   (default: int everywhere)
- /price_sweep optimizer and /predict endpoints
- /compare_cost_modes endpoint + UI comparison tables
- /autopilot endpoint
- Auto-reload for data/purchases.csv (mtime-based)
- Baseline (hold last historical price) + uplift vs baseline using selected quantity_mode

Behavior notes:
- /compare_cost_modes: runs BOTH revenue- and profit-optimized sweeps for each selected cost mode,
  computes weekly totals (units/revenue/profit) and uplifts vs the mode’s own baseline.
- Decision helper: per cost mode, picks the optimizer that maximizes TOTAL weekly profit;
  tie-break on TOTAL revenue, then TOTAL units.
- /autopilot: picks the best cost mode using the same rule and returns the 7-day price plan
  plus weekly stock (supports optional buffer %).

Requires:
- models/model_h{1..7}.xgb.json and models/feature_order_h{1..7}.json
- models/categories.json, models/holidays_meta.json
- data/sales.csv
- data/purchases.csv (for replacement/WAC/standard)
"""

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import xgboost as xgb
import json, os, csv
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, List, Dict, Any
import math
from decimal import InvalidOperation
import traceback

def _safe_float(x):
    try:
        f = float(x)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return None

def _safe_sum(a, b):
    a = _safe_float(a) or 0.0
    b = _safe_float(b) or 0.0
    return a + b

# ---------------------------- Config ----------------------------
H_LIST = [1, 2, 3, 4, 5, 6, 7]
LOG_PATH = "models/predictions_log.csv"
QMODE_DEFAULT = "int"  # "int" | "float"

# ---------------------- Money / validation ----------------------
def round_money(x: float) -> float:
    try:
        d = Decimal(str(x))
    except Exception:
        raise HTTPException(400, f"Invalid numeric value: {x}")
    if d < 0:
        raise HTTPException(400, f"Value must be non-negative: {x}")
    return float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def round_money_any(x: float) -> float:
    try:
        d = Decimal(str(x))
    except Exception:
        raise HTTPException(400, f"Invalid numeric value: {x}")
    return float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def round_money_list(v, n_expected: int):
    if isinstance(v, list):
        if len(v) != n_expected:
            raise HTTPException(400, f"Expected {n_expected} values, got {len(v)}")
        return [round_money(x) for x in v]
    return [round_money(v)] * n_expected

def normalize_cost_list(v, n_expected: int, name: str, allow_none: bool = True, default_val: float = 0.0) -> Optional[List[float]]:
    if v is None:
        return None if allow_none else [round_money(default_val)] * n_expected
    if isinstance(v, list):
        if len(v) != n_expected:
            raise HTTPException(400, f"{name} must be scalar or list of length {n_expected}")
        return [round_money(x) for x in v]
    return [round_money(v)] * n_expected

# ---------------------- Quantity rounding -----------------------
def validate_qmode(mode: Optional[str]) -> str:
    m = (mode or QMODE_DEFAULT).lower().strip()
    if m not in {"int", "float"}:
        raise HTTPException(400, "quantity_mode must be 'int' or 'float'")
    return m

def quantize_sales(qty: float, qmode: str) -> float:
    qty = max(0.0, float(qty))
    if qmode == "int":
        return float(Decimal(str(qty)).to_integral_value(rounding=ROUND_HALF_UP))
    return float(Decimal(str(qty)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---------------------- Load artifacts/data ---------------------
CATS = json.load(open("models/categories.json"))

# Holidays meta (from training)
try:
    HOL_META = json.load(open("models/holidays_meta.json"))
    HOL_COUNTRY = (HOL_META.get("country") or "").strip()
    HOL_SUBDIV  = (HOL_META.get("subdiv")  or "").strip()
except Exception:
    HOL_COUNTRY = os.getenv("HOL_COUNTRY", "GR")
    HOL_SUBDIV  = os.getenv("HOL_SUBDIV", "")

def get_holidays(country: str, subdiv: str):
    try:
        import holidays
        return holidays.country_holidays(country, subdiv=subdiv) if subdiv else holidays.country_holidays(country)
    except Exception:
        return None

HOL = get_holidays(HOL_COUNTRY, HOL_SUBDIV)
def is_holiday(d: pd.Timestamp) -> int:
    return int(HOL is not None and d in HOL)

# History
HIST = pd.read_csv("data/sales.csv")
HIST["date"] = pd.to_datetime(HIST["date"])
HIST = HIST.sort_values(["store_id", "item_id", "date"])
GLOBAL_MEAN = float(HIST["sales"].mean())

# ----------------- Auto-loader for purchases.csv ----------------
PURCHASES_PATH = "data/purchases.csv"
_PURCH = None
_PURCH_MTIME = None

def get_purchases_df() -> Optional[pd.DataFrame]:
    """Load/refresh purchases.csv when mtime changes. Returns cleaned DataFrame or None."""
    global _PURCH, _PURCH_MTIME
    try:
        mtime = os.path.getmtime(PURCHASES_PATH)
    except Exception:
        _PURCH = None
        _PURCH_MTIME = None
        return None
    if _PURCH is None or _PURCH_MTIME != mtime:
        try:
            df = pd.read_csv(PURCHASES_PATH)
        except Exception:
            _PURCH = None
            _PURCH_MTIME = mtime
            return None
        required = {"item_id", "date", "receipt_qty", "landed_unit_cost"}
        if not required.issubset(df.columns):
            _PURCH = None
            _PURCH_MTIME = mtime
            return None
        df["item_id"] = df["item_id"].astype(str).str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        for col in ("receipt_qty", "landed_unit_cost", "standard_cost"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        _PURCH = df
        _PURCH_MTIME = mtime
    return _PURCH

# ----------- Purchases-driven cost calculators (scalars) --------
def replacement_cost_scalar(item_id: str, asof: pd.Timestamp) -> Optional[float]:
    P = get_purchases_df()
    if P is None: return None
    iid = str(item_id).strip()
    sub = P[(P["item_id"] == iid) & (P["date"] <= asof)].sort_values("date")
    if sub.empty or "landed_unit_cost" not in sub.columns: return None
    val = sub["landed_unit_cost"].iloc[-1]
    return round(float(val), 2) if pd.notna(val) else None

def wac_cost_scalar(item_id: str, asof: pd.Timestamp, window_days: int = 90) -> Optional[float]:
    P = get_purchases_df()
    if P is None: return None
    if "receipt_qty" not in P.columns or "landed_unit_cost" not in P.columns: return None
    iid = str(item_id).strip()
    start = asof - pd.Timedelta(days=window_days - 1)
    sub = P[(P["item_id"] == iid) & (P["date"] >= start) & (P["date"] <= asof)].copy()
    if sub.empty: return None
    qty = sub["receipt_qty"].astype(float).clip(lower=0)
    if qty.sum() <= 0: return None
    wac = float((qty * sub["landed_unit_cost"].astype(float)).sum() / qty.sum())
    return round(wac, 2)

def standard_cost_scalar(item_id: str, asof: pd.Timestamp) -> Optional[float]:
    P = get_purchases_df()
    if P is None: return None
    iid = str(item_id).strip()
    sub = P[(P["item_id"] == iid) & (P["date"] <= asof)].sort_values("date")
    if "standard_cost" in sub.columns and sub["standard_cost"].notna().any():
        return round(float(sub["standard_cost"].dropna().iloc[-1]), 2)
    if "landed_unit_cost" in sub.columns and sub["landed_unit_cost"].notna().any():
        return round(float(sub["landed_unit_cost"].median()), 2)
    return None

def resolve_unit_costs(cost_mode: str, item_id: str, start_date: pd.Timestamp, wac_days: int) -> Optional[List[float]]:
    """Return a 7-length list or None. For replacement/WAC/standard compute one scalar and repeat."""
    asof = start_date - pd.Timedelta(days=1)
    mode = (cost_mode or "manual").lower()
    if mode == "manual":
        return None
    if mode == "replacement":
        c = replacement_cost_scalar(item_id, asof)
        return [c] * 7 if c is not None else None
    if mode == "wac":
        c = wac_cost_scalar(item_id, asof, wac_days)
        return [c] * 7 if c is not None else None
    if mode == "standard":
        c = standard_cost_scalar(item_id, asof)
        return [c] * 7 if c is not None else None
    return None

# --------------------------- Models -----------------------------
MODELS: Dict[int, xgb.Booster] = {}
FEATURES: Dict[int, List[str]] = {}
for h in H_LIST:
    bst = xgb.Booster(); bst.load_model(f"models/model_h{h}.xgb.json")
    MODELS[h] = bst
    with open(f"models/feature_order_h{h}.json") as f: FEATURES[h] = json.load(f)

# ----------------------------- App ------------------------------
app = FastAPI(title="SalesPrediction (auto-reload purchases, cost modes, compare, baseline uplift)")

# ---------------------------- Schemas ---------------------------
class NextDayReq(BaseModel):
    store_id: str
    item_id: str
    date: str
    price: float
    promo: int
    quantity_mode: Optional[str] = QMODE_DEFAULT

class Next7Req(BaseModel):
    store_id: str
    item_id: str
    start_date: str
    price: float | List[float]
    promo: int | List[int]
    quantity_mode: Optional[str] = QMODE_DEFAULT

class PriceSweepReq(BaseModel):
    store_id: str
    item_id: str
    start_date: str
    price_min: float
    price_max: float
    price_step: float = 0.10
    promo: int | List[int] = 0
    optimize_by: str = "revenue"       # "revenue" | "profit"
    cost_mode: str = "manual"          # "manual" | "replacement" | "wac" | "standard"
    wac_window_days: int = 90
    unit_cost: float | List[float] | None = None
    extra_unit_cost: float | List[float] | None = None
    extra_fixed_cost: float | List[float] | None = None
    quantity_mode: Optional[str] = QMODE_DEFAULT

# ---------------------- Feature preparation --------------------
def base_features_for(store_id: str, item_id: str, asof_date: pd.Timestamp):
    df = HIST[(HIST.store_id.astype(str) == str(store_id)) & (HIST.item_id.astype(str) == str(item_id))].copy()
    if df.empty: raise ValueError("Unknown store_id/item_id (not in training).")
    df = df[df["date"] <= asof_date].sort_values("date")
    if df.empty: raise ValueError("No history available up to the given date.")
    df["lag_1"]  = df["sales"].shift(1);  df["lag_7"]  = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14); df["lag_28"] = df["sales"].shift(28)
    df["roll_mean_7"]  = df["sales"].shift(1).rolling(7).mean()
    df["roll_mean_28"] = df["sales"].shift(1).rolling(28).mean()
    df["roll_std_7"]   = df["sales"].shift(1).rolling(7).std()
    df["price_roll_28"]= df["price"].shift(1).rolling(28).mean()
    last = df.iloc[-1]
    base = {
        "sales_lag_1":   float(last["lag_1"])   if pd.notna(last["lag_1"])   else 0.0,
        "sales_lag_7":   float(last["lag_7"])   if pd.notna(last["lag_7"])   else 0.0,
        "sales_lag_14":  float(last["lag_14"])  if pd.notna(last["lag_14"])  else 0.0,
        "sales_lag_28":  float(last["lag_28"])  if pd.notna(last["lag_28"])  else 0.0,
        "roll_mean_7":   float(last["roll_mean_7"])   if pd.notna(last["roll_mean_7"])   else 0.0,
        "roll_mean_28":  float(last["roll_mean_28"])  if pd.notna(last["roll_mean_28"])  else 0.0,
        "roll_std_7":    float(last["roll_std_7"])    if pd.notna(last["roll_std_7"])    else 0.0,
        "price_roll_28": float(last["price_roll_28"]) if pd.notna(last["price_roll_28"]) else 0.0,
    }
    for sid in CATS["store_ids"]:
        base[f"store_id_{sid}"] = 1 if str(sid) == str(store_id) else 0
    for iid in CATS["item_ids"]:
        base[f"item_id_{iid}"] = 1 if str(iid) == str(item_id) else 0
    info = {"hist_len": int(df.shape[0]),
            "ma7": float(df["sales"].tail(7).mean()) if df.shape[0] >= 1 else float("nan"),
            "ma28": float(df["sales"].tail(28).mean()) if df.shape[0] >= 1 else float("nan")}
    return base, info

def row_for_h(base: dict, h: int, target_date: pd.Timestamp, price_h: float, promo_h: int) -> dict:
    dow = int(target_date.weekday()); month = int(target_date.month); is_weekend = int(dow >= 5)
    price_h = round_money(price_h)
    row = base.copy(); row.update({
        "dow": dow, "month": month, "is_weekend": is_weekend,
        f"price_h{h}": float(price_h), f"promo_h{h}": int(promo_h),
        f"price_ratio_h{h}": float(price_h / (row["price_roll_28"] + 1e-9)) if row["price_roll_28"] > 0 else 1.0,
        f"is_hol_h{h}": is_holiday(target_date),
    })
    return {col: row.get(col, 0.0) for col in FEATURES[h]}

# ---------- Baseline helpers (respect quantity_mode) -----------
def last_hist_price(store_id: str, item_id: str, asof: pd.Timestamp) -> Optional[float]:
    df = HIST[
        (HIST.store_id.astype(str) == str(store_id)) &
        (HIST.item_id.astype(str) == str(item_id)) &
        (HIST["date"] <= asof)
    ].sort_values("date")
    if df.empty:
        return None
    return float(round(df["price"].iloc[-1], 2))

def _score_day_with_qmode(
    base: dict,
    info: dict,
    h: int,
    d: pd.Timestamp,
    price: float,
    promo: int,
    qmode: str,
    unit_cost: Optional[float],
    extra_unit_cost: float,
    extra_fixed_cost: float
):
    """Predict one day and compute sales (quantized per qmode), revenue, profit."""
    if info["hist_len"] < 14:
        pred = float(np.nan_to_num(info["ma7"], nan=info["ma28"])) if not np.isnan(info["ma7"]) else info["ma28"]
        if np.isnan(pred):
            pred = GLOBAL_MEAN
        fallback = True
    else:
        row = row_for_h(base, h, d, price, promo)
        X = pd.DataFrame([row])[FEATURES[h]]
        pred = float(np.expm1(MODELS[h].predict(xgb.DMatrix(X))[0]))
        fallback = False

    sales_q = quantize_sales(pred, qmode)
    revenue = round_money(max(0.0, price * sales_q))

    profit = None
    if unit_cost is not None:
        margin_unit = price - unit_cost - (extra_unit_cost or 0.0)
        profit = round_money_any(margin_unit * sales_q - (extra_fixed_cost or 0.0))

    return sales_q, revenue, profit, fallback

# --------------------------- Web UI -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html = """
<!doctype html><html><head><meta charset="utf-8" />
<title>Sales Prediction</title>
<style>
body{font-family:system-ui,Segoe UI,Arial;margin:40px;max-width:1400px}
input,button,select{font-size:16px;padding:6px;margin:4px 0;}
.card{padding:16px;border:1px solid #ddd;border-radius:12px;margin-bottom:18px}

/* FLEX rows inside forms */
.card form .row,
.card form .row3{
  display:flex; gap:12px; align-items:flex-end; flex-wrap:wrap;
}
.card form .row > div{ flex:1 1 280px; min-width:260px; }
.card form .row3 > div{ flex:1 1 220px; min-width:200px; }

.card form select,
.card form button{
  width: calc(100% + 6px);
}

.card form input:not([type="checkbox"]) {
  width: calc(100% - 10px);
}

/* JSON output boxes */
#out,#out2,#cmpOutm,#autoMemo{
  white-space:pre-wrap;background:#fafafa;border:1px solid #eee;
  padding:12px;border-radius:8px;overflow-x:auto;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  word-break:break-word;
}
#out,#out2,#autoMemo{max-height:200px;overflow-y:auto;}
#cmpGrid {margin-bottom: 20px;}

.badge{display:inline-block;padding:4px 8px;border-radius:6px;background:#eef;border:1px solid #ccd;margin-left:6px}
h2{margin:0 0 8px 0}
.flex-table{display:flex;flex-direction:column;border:1px solid #eee;border-radius:10px;overflow-x:auto}
.flex-row{display:flex;gap:8px}
.flex-row.header{background:#f3f6ff;font-weight:600;border-bottom:1px solid #e5e9ff}
.cell{flex:1;padding:8px 10px;min-width:120px}
.cell.small{min-width:80px}
.cell.right{text-align:right}
.row-alt{background:#fafbff}
.smallnote{font-size:12px;color:#666}
.stocknote,.benefitnote{font-weight:bold;color:#222}

@media (max-width: 900px){
  .card form .row > div, .card form .row3 > div{ flex-basis:100%; min-width:0; }
}

/* Decision helper styling */
.decision-wrap{border:1px solid #eee;border-radius:12px;padding:12px;background:#fcfcff}
.kpi-grid{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 6px}
.kpi{flex:1 1 180px;min-width:180px;border:1px solid #eee;border-radius:12px;padding:10px 12px;background:#fafafa}
.kpi .label{font-size:12px;color:#666;margin-bottom:2px}
.kpi .value{font-size:20px;font-weight:700;line-height:1.25}
.kpi .sub{font-size:12px;color:#555;margin-top:4px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.mode-chip{display:inline-block;padding:4px 10px;border-radius:999px;background:#eef;border:1px solid #ccd;font-weight:600}
.delta-pos{color:#0b7a28}
.delta-neg{color:#b00020}
.smallmono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#555}
.dec-list{display:flex;flex-direction:column;gap:12px;margin-top:8px}
.dec-card{border:1px solid #eee;border-radius:12px;padding:10px 12px;background:#fff}
.dec-card .title{display:flex;align-items:center;gap:8px;margin-bottom:6px}

/* Warning / alert styles */
.alert-warn{border:1px solid #f4c2c2;background:#fff6f7;color:#8a1c1c;border-radius:10px;padding:10px 12px;margin:10px 0 0;font-weight:600;}

</style>
</head><body>
<h1>Sales Prediction</h1>
<p>Holidays (locked to training): <span class="badge">__HOL__</span></p>

<div class="card">
  <h2>Predict 7 days</h2>
  <form id="f1">
    <div class="row">
      <div><label>Store ID</label><input id="store_id" value="S01" required></div>
      <div><label>Item ID</label><input id="item_id" value="I03" required></div>
    </div>
    <div class="row">
      <div><label>Start Date (YYYY-MM-DD)</label><input id="start_date" value="__START__" required></div>
      <div><label>Price (single or CSV of 7)</label><input id="price" required></div>
    </div>
    <div class="row">
      <div><label>Promo (single or CSV of 7; 0/1)</label><input id="promo" value="0" required></div>
      <div>
        <label>Quantity Mode</label>
        <select id="qmode1">
          <option value="int" selected>int (whole units)</option>
          <option value="float">float (2dp — weight/volume)</option>
        </select>
      </div>
    </div>
    <button type="submit">Predict 7 days</button>
  </form>
</div>
<h3>Response</h3>
<div id="out">—</div>

<!-- Forecast (Predict 7 days) + weekly stock need -->
<div class="card">
  <h2>Forecast (7 days)</h2>
  <div id="fcFlex" class="flex-table">
    <div class="flex-row header">
      <div class="cell">Date</div>
      <div class="cell right">Price</div>
      <div class="cell right">Promo</div>
      <div class="cell right">Predicted Sales</div>
      <div class="cell small">Fallback</div>
    </div>
  </div>
  <p id="fcNeed" class="stocknote">—</p>
</div>

<div class="card">
  <h2>Price Sweep (optimizer)</h2>
  <form id="f2">
    <div class="row">
      <div><label>Store ID</label><input id="s2" value="S01" required></div>
      <div><label>Item ID</label><input id="i2" value="I03" required></div>
    </div>
    <div class="row">
      <div><label>Start Date</label><input id="d2" value="__START__" required></div>
      <div><label>Promo (single or CSV of 7; 0/1)</label><input id="pm2" value="0" required></div>
    </div>
    <div class="row">
      <div><label>Price Min</label><input id="pmin" value="27" required></div>
      <div><label>Price Max</label><input id="pmax" value="36" required></div>
    </div>
    <div class="row">
      <div><label>Price Step (≥ 0.01)</label><input id="pstep" value="0.10" required></div>
      <div><label>Optimize By</label>
        <select id="optby">
          <option value="revenue" selected>revenue</option>
          <option value="profit">profit</option>
        </select>
      </div>
    </div>
    <div class="row">
      <div><label>Cost Mode</label>
        <select id="cmode">
          <option value="manual">manual (enter costs below)</option>
          <option value="replacement">replacement (latest receipt)</option>
          <option value="wac">wac (weighted avg)</option>
          <option value="standard" selected>standard (ERP)</option>
        </select>
      </div>
      <div><label>WAC Window (days)</label><input id="wacwin" value="90"></div>
    </div>
    <div class="row3">
      <div><label>Unit Cost (scalar or CSV x7)</label><input id="ucost" placeholder="required if manual & optimizing profit"></div>
      <div><label>Extra Unit Cost (per-unit; scalar or CSV x7)</label><input id="eucost" placeholder="e.g., payment fees"></div>
      <div><label>Extra Fixed Cost (daily; scalar or CSV x7)</label><input id="efcost" placeholder="e.g., daily ads"></div>
    </div>
    <div class="row">
      <div>
        <label>Quantity Mode</label>
        <select id="qmode2">
          <option value="int" selected>int (whole units)</option>
          <option value="float">float (2dp — weight/volume)</option>
        </select>
      </div>
    </div>
    <button type="submit">Run Sweep</button>
  </form>
</div>

<h3>Price Sweep Result</h3>
<div id="out2">—</div>

<div class="card">
  <h2>Recommended (per day)</h2>
  <div id="recFlex" class="flex-table">
    <div class="flex-row header">
      <div class="cell">Date</div>
      <div class="cell right">Best Price</div>
      <div class="cell right">Sales</div>
      <div class="cell right">Revenue</div>
      <div class="cell right">Profit</div>
      <div class="cell small">Fallback</div>
    </div>
  </div>
  <p id="recNeed" class="stocknote">—</p>
  <p id="recBenefit" class="benefitnote">—</p>
</div>

<div class="card">
  <h2>Compare Cost Modes</h2>
  <p class="smallnote">Runs both revenue- and profit-optimized sweeps per selected cost mode; computes totals and baseline uplifts.</p>
  <form id="f3">
    <div class="row">
      <div><label>Store ID</label><input id="cs_store" value="S01" required></div>
      <div><label>Item ID</label><input id="cs_item" value="I03" required></div>
    </div>
    <div class="row">
      <div><label>Start Date</label><input id="cs_date" value="__START__" required></div>
      <div><label>Promo (single or CSV of 7; 0/1)</label><input id="cs_promo" value="0" required></div>
    </div>
    <div class="row">
      <div><label>Price Min</label><input id="cs_pmin" value="27" required></div>
      <div><label>Price Max</label><input id="cs_pmax" value="36" required></div>
    </div>
    <div class="row">
      <div><label>Price Step</label><input id="cs_pstep" value="0.10" required></div>
      <div><label>WAC Window (days)</label><input id="cs_wacwin" value="90"></div>
    </div>
    <div class="row">
      <div>
        <label>Modes</label>
        <div>
          <label><input type="checkbox" class="mchk" value="replacement" checked> replacement</label>
          <label><input type="checkbox" class="mchk" value="wac" checked> wac</label>
          <label><input type="checkbox" class="mchk" value="standard" checked> standard</label>
          <label><input type="checkbox" class="mchk" value="manual"> manual</label>
        </div>
      </div>
    </div>
    <div class="row3">
      <div><label>Unit Cost (manual only)</label><input id="cs_ucost"></div>
      <div><label>Extra Unit Cost (per-unit)</label><input id="cs_eucost" value="0"></div>
      <div><label>Extra Fixed Cost (daily)</label><input id="cs_efcost" value="0"></div>
    </div>
    <div class="row">
      <div>
        <label>Quantity Mode</label>
        <select id="cs_qmode">
          <option value="int" selected>int (whole units)</option>
          <option value="float">float (2dp)</option>
        </select>
      </div>
    </div>
    <div class="row">
      <div>
        <label>Autopilot Buffer %</label>
        <input id="cs_buffer" value="0" placeholder="e.g., 10 for +10%">
      </div>
      <div><button type="button" id="btnAuto">Run Autopilot</button></div>
    </div>    
    <button type="submit">Compare</button>
  </form>

  <h3>Summary</h3>
  <div id="cmpOut">—</div>

  <h3>Per-day Best Price Grid</h3>
  <div id="cmpGrid" class="flex-table"></div>
  
  <div class="card">
    <h2>Decision helper</h2>
    <p class="smallnote">Chooses the best optimizer per cost mode that maximizes the total weekly profit; returns 7-day price plan + weekly stock (optional buffer)</p>
    <div id="decOut" class="decision-wrap">
      <div class="smallnote">Run a comparison to see a recommendation.</div>
    </div>
  </div>

  <div class="card">
    <h2>Autopilot</h2>
    <div id="autoOut" class="decision-wrap"><div class="smallnote">Click “Run Autopilot” to pick a plan & stock.</div></div>
    <h3 style="margin-top:10px">Autopilot Plan (7 days)</h3>
    <div id="autoPlan" class="flex-table"></div>
  </div>

  <h3>Autopilot Memo (JSON)</h3>
  <div id="autoMemo">—</div>

</div>

<script>
const clamp01 = v => (Number(v) > 0 ? 1 : 0);
// Normalize promo input to exactly 7 binary values (0/1).
const toPromo7 = (raw) => {
  const arr = parseArr(raw);
  if (!arr) return Array(7).fill(clamp01(raw));          // single value → repeat 7
  const clamped = arr.map(clamp01);
  if (clamped.length === 7) return clamped;              // CSV of 7 → ok
  if (clamped.length === 1) return Array(7).fill(clamped[0]); // 1 value → repeat 7
  throw new Error('Promo must be a single 0/1 or CSV of 7 (0/1).');
};


const fix2=(x)=> Number.parseFloat(x).toFixed(2);
const parseArr = (s)=> (s && s.includes(',')) ? s.split(',').map(x=>x.trim()).filter(Boolean) : null;
const fmtMoney=(x)=> (x==null? '—' : Number(x).toFixed(2));
const fmtDelta=(x)=> (x==null? '—' : ((Number(x)>=0? '+':'-') + Math.abs(Number(x)).toFixed(2)));

function fmtSales(x, qmode){
  if(x==null) return '—';
  return qmode==='int' ? String(Math.round(Number(x))) : Number.parseFloat(x).toFixed(2);
}

/* Forecast (Predict 7 days) + weekly stock need */
function renderForecastTable(data, qmode){
  const host=document.getElementById('fcFlex');
  host.querySelectorAll('.flex-row.data').forEach(el=>el.remove());
  const pNeed=document.getElementById('fcNeed');
  pNeed.textContent='—';
  if(!data || !data.forecasts || !data.forecasts.length){ return; }

  let totalUnits = 0;
  let totalRev   = 0;

  data.forecasts.forEach((r,i)=>{
    const row=document.createElement('div');
    row.className='flex-row data'+(i%2?' row-alt':'');

    const add=(cls,txt)=>{ const d=document.createElement('div'); d.className='cell '+cls; d.textContent=txt; row.appendChild(d); };
    add('', r.date || '');
    add('right', r.price!=null ? Number(r.price).toFixed(2) : '—');
    add('right', (r.promo!=null ? String(r.promo) : '—'));
    add('right', fmtSales(r.predicted_sales, qmode));
    add('small', r.fallback ? 'yes' : 'no');
    host.appendChild(row);

    const s = Number(r.predicted_sales || 0);
    totalUnits += s;
    if(r.price!=null) totalRev += Number(r.price) * s;
  });

  // Prefer backend-rounded totals if present
  if(data.total_expected_units!=null) totalUnits = Number(data.total_expected_units);
  if(data.total_expected_revenue!=null) totalRev = Number(data.total_expected_revenue);

  // Build the summary line
  let txt = 'Stock needed for the week: ' + fmtSales(totalUnits, qmode)
          + ' | Total revenue: ' + Number(totalRev).toFixed(2);

  // Show baseline (hold last price) only if different from entered prices
  const baseline = data.baseline || null;
  if(baseline && baseline.price!=null){
    const enteredPrices = data.forecasts.map(r => Number(r.price));
    const differs = enteredPrices.some(p => Number(p).toFixed(2) !== Number(baseline.price).toFixed(2));
    if(differs && baseline.total_revenue!=null){
      const delta = Number(totalRev) - Number(baseline.total_revenue);
      const sign = delta>=0 ? '+' : '−';
      txt += ' (baseline @ ' + Number(baseline.price).toFixed(2)
           + ': ' + Number(baseline.total_revenue).toFixed(2)
           + '; Δ ' + sign + Math.abs(delta).toFixed(2) + ')';
    }
  }

  pNeed.textContent = txt;
}

function renderFlex(recommended, qmode, uplift){
  const host=document.getElementById('recFlex');
  const pNeed=document.getElementById('recNeed');
  const pBen=document.getElementById('recBenefit');
  host.querySelectorAll('.flex-row.data').forEach(el=>el.remove());
  pNeed.textContent='—';
  pBen.textContent='—';
  if(!recommended || !recommended.length){ return; }
  let total=0;
  recommended.forEach((r,i)=>{
    const row=document.createElement('div');
    row.className='flex-row data'+(i%2?' row-alt':'');

    const t=(cls,txt)=>{ const d=document.createElement('div'); d.className='cell '+cls; d.textContent=txt; row.appendChild(d); };
    t('', r.date || '');
    t('right', (r.best_price!=null? Number(r.best_price).toFixed(2): '—'));
    t('right', fmtSales(r.expected_sales, qmode));
    t('right', (r.expected_revenue!=null? Number(r.expected_revenue).toFixed(2): '—'));
    t('right', (r.expected_profit!=null? Number(r.expected_profit).toFixed(2): '—'));
    t('small', r.fallback? 'yes':'no');
    host.appendChild(row);
    total += Number(r.expected_sales || 0);
  });
 
  // Moved the uplift block to here to calculate unitsStr
  if(uplift){
    const uUnits = uplift.units!=null ? uplift.units : null;
    const unitsStr = (uUnits==null)? '—' : ((Number(uUnits)>=0? '+':'-') + (qmode==='int'? Math.abs(Math.round(Number(uUnits))) : Math.abs(Number(uUnits)).toFixed(2)));
    pNeed.textContent = 'Stock needed for the week: ' + fmtSales(total, qmode) + (uUnits != null ? ' (' + unitsStr + ')' : '');
    
    // Now handle the benefit string
    const uRev = uplift.revenue!=null ? uplift.revenue : null;
    const uProf = uplift.profit!=null ? uplift.profit : null;
    const revStr = fmtDelta(uRev);
    const profStr = fmtDelta(uProf);

    // Calculate total benefit from revenue and profit uplifts
    let totalBenefit = null;
    if(uProf != null && uRev != null) {
      totalBenefit = Number(uProf) + Number(uRev);
    } else if (uProf != null) {
      totalBenefit = Number(uProf);
    } else if (uRev != null) {
      totalBenefit = Number(uRev);
    }
    const totalBenefitStr = fmtDelta(totalBenefit);

    pBen.textContent = 'Benefit vs baseline: Δ Revenue ' + revStr + (uProf==null? '' : (' | Δ Profit ' + profStr));
  } else {
    // This else is important to ensure pNeed is set correctly when there's no uplift
    pNeed.textContent = 'Stock needed for the week: ' + fmtSales(total, qmode);
  }
}

function renderSummaryTable(data){
  const host=document.getElementById('cmpOut');
  if(!data || !data.compare){ host.textContent='—'; return; }
  const rows = data.compare;
  const qmodeCmp = document.getElementById('cs_qmode')?.value || 'int';

  let html = '<div class="flex-table"><div class="flex-row header">'
           + '<div class="cell right">Mode (optimizer)</div>'
           + '<div class="cell right">Unit Cost</div>'
           + '<div class="cell right">Total Units</div>'
           + '<div class="cell right">Baseline Rev</div>'
           + '<div class="cell right">Total Rev</div>'
           + '<div class="cell right">Baseline Profit</div>'
           + '<div class="cell right">Total Profit</div>'
           + '</div>';

  rows.forEach((r,i)=>{
    if(r.error){
      html += `<div class="flex-row${(i%2?' row-alt':'')}">`
           + `<div class="cell right">${r.mode}</div>`
           + `<div class="cell right">—</div>`
           + `<div class="cell right">—</div>`
           + `<div class="cell right">—</div>`
           + `<div class="cell right">—</div>`
           + `<div class="cell right">—</div>`
           + `<div class="cell right">—</div>`
           + `</div>`;
      return;
    }

    const uc = Array.isArray(r.unit_cost) ? r.unit_cost[0] : r.unit_cost;
    const tr = (i%2)?' row-alt':'';
    const optimizer = (r.optimizer || '').toLowerCase();

    const totalUnits = (r.total_units!=null)
      ? (qmodeCmp==='int' ? Math.round(Number(r.total_units)) : Number(r.total_units).toFixed(2))
      : '—';

    const bRev = (r.baseline_total_revenue==null? '—' : Number(r.baseline_total_revenue).toFixed(2));
    const tRev = (r.total_revenue==null? '—' : Number(r.total_revenue).toFixed(2));
    const bProf= (r.baseline_total_profit==null? '—' : Number(r.baseline_total_profit).toFixed(2));
    const tProf= (r.total_profit==null? '—' : Number(r.total_profit).toFixed(2));

    const tRevCell  = (optimizer==='revenue' && tRev!=='—') ? `<b>${tRev}</b>` : tRev;
    const tProfCell = (optimizer==='profit'  && tProf!=='—') ? `<b>${tProf}</b>` : tProf;

    html += '<div class="flex-row'+tr+'">'
          + `<div class="cell right">${r.mode}</div>`
          + `<div class="cell right">${uc!=null? Number(uc).toFixed(2) : '—'}</div>`
          + `<div class="cell right">${totalUnits}</div>`
          + `<div class="cell right">${bRev}</div>`
          + `<div class="cell right">${tRevCell}</div>`
          + `<div class="cell right">${bProf}</div>`
          + `<div class="cell right">${tProfCell}</div>`
          + '</div>';
  });

  html += '</div>';
  host.innerHTML = html;
}

function renderCompareGrid(data){
  const grid=document.getElementById('cmpGrid');
  grid.innerHTML='';
  if(!data || !data.compare || !data.dates){ return; }
  const modes = data.compare.map(r=>r.mode);
  let header = document.createElement('div'); header.className='flex-row header';
  const addCell=(row,txt,cls='cell')=>{ const d=document.createElement('div'); d.className=cls; d.textContent=txt; row.appendChild(d); };
  addCell(header,'Date');
  modes.forEach(m=> addCell(header, m, 'cell right'));
  grid.appendChild(header);
  data.dates.forEach((dt,ri)=>{
    let row=document.createElement('div'); row.className='flex-row'+(ri%2?' row-alt':'');
    addCell(row, dt);
    data.compare.forEach(r=>{
      const bp = (r.best_prices && r.best_prices[ri]!=null) ? Number(r.best_prices[ri]).toFixed(2) : (r.error?'—':'—');
      addCell(row, bp, 'cell right');
    });
    grid.appendChild(row);
  });
}

function renderDecision(data){
  const host = document.getElementById('decOut');
  if(!host){ return; }

  // helpers
  const qmode = document.getElementById('cs_qmode')?.value || 'int';
  const fmtUnits = (x)=> x==null ? '—' : (qmode==='int' ? String(Math.round(Number(x))) : Number(x).toFixed(2));
  const fmtMoney = (x)=> x==null ? '—' : Number(x).toFixed(2);
  const deltaTag = (x)=>{
    if(x==null) return '—';
    const v = Number(x);
    const cls = v>=0 ? 'delta-pos' : 'delta-neg';
    const sign = v>=0 ? '+' : '−';
    return `<span class="${cls}">${sign}${Math.abs(v).toFixed(2)}</span>`;
  };

  if(!data || !Array.isArray(data.decisions) || !data.decisions.length){
    host.innerHTML = '<div class="smallnote">Run a comparison to see a recommendation.</div>';
    return;
  }

  // Map compare rows to lookup chosen totals
  const rows = Array.isArray(data.compare) ? data.compare : [];
  const keyOf = (cm, opt)=> `${cm}|${opt}`;
  const byKey = {};
  rows.forEach(r => byKey[keyOf(r.cost_mode, r.optimizer)] = r);

  let html = '';
  let anyPositiveChosen = false;
  const cards = [];

  data.decisions.forEach((d, idx)=>{
    if(d.error){
      cards.push(`<div class="kpi">
        <div class="label">Mode</div>
        <div class="value"><span class="mode-chip">${d.cost_mode}</span></div>
        <div class="sub">Error: ${d.error}</div>
      </div>`);
      return;
    }

    const chosen = (d.chosen_optimizer)
      ? byKey[keyOf(d.cost_mode, d.chosen_optimizer)] || {}
      : null;

    const tp = chosen ? chosen.total_profit  : null;
    const tr = chosen ? chosen.total_revenue : null;
    const tu = chosen ? chosen.total_units   : null;

    if(tp != null && Number(tp) > 0) anyPositiveChosen = true;

    // Card per cost mode
    let card = `
      <div style="margin-bottom:12px; padding:12px; border:1px solid #eee; border-radius:12px; background:#fcfcff">
        <div class="smallnote">Recommended</div>
        <div style="margin:2px 0 6px">
          <span class="mode-chip">${d.cost_mode}${d.chosen_optimizer ? ' · '+d.chosen_optimizer : ''}</span>
        </div>
        <div class="kpi-grid">
          <div class="kpi">
            <div class="label">Total Profit (score)</div>
            <div class="value">${fmtMoney(tp)}</div>
          </div>
          <div class="kpi">
            <div class="label">Total Revenue</div>
            <div class="value">${fmtMoney(tr)}</div>
          </div>
          <div class="kpi">
            <div class="label">Total Units (week)</div>
            <div class="value">${fmtUnits(tu)}</div>
          </div>
          <div class="kpi">
            <div class="label">Baseline (Profit / Revenue / Units)</div>
            <div class="sub">
              ${fmtMoney(chosen?.baseline_total_profit)} / ${fmtMoney(chosen?.baseline_total_revenue)} / ${fmtUnits(chosen?.baseline_total_units)}
            </div>
          </div>
        </div>
    `;

    // Per-mode warning if chosen total profit is negative
    if(tp != null && Number(tp) < 0){
      card += `<div class="alert-warn">
        Best plan for <b>${d.cost_mode}</b> still yields a negative weekly profit (${fmtMoney(tp)}).
        Consider widening the price range or revisiting your cost inputs.
      </div>`;
    }

    // Optional rationale/message from backend
    if(d.message){
      card += `<div class="smallnote" style="margin-top:6px">${d.message}</div>`;
    }

    card += `</div>`;
    cards.push(card);
  });

  // Top-level banner if no chosen plan across modes is profitable
  if(!anyPositiveChosen){
    html += `<div class="alert-warn">
      No profitable plan found in the tested price range across your selected cost modes.
      Try widening the price range (↑Price Max), increasing step coverage, or check unit/extra costs.
    </div>`;
  }

  html += cards.join('');
  host.innerHTML = html;
}


function renderAutopilot(data){
  const out = document.getElementById('autoOut');
  const planHost = document.getElementById('autoPlan');
  out.innerHTML = ''; planHost.innerHTML='';

  if(!data || !data.chosen){
    out.innerHTML = '<div class="smallnote">No result</div>';
    return;
  }
  const qmode = data.quantity_mode || (document.getElementById('cs_qmode')?.value || 'int');
  const fmtUnits = (x)=> x==null ? '—' : (qmode==='int' ? String(Math.round(Number(x))) : Number(x).toFixed(2));
  const chosen = data.chosen;
  const stock  = data.weekly_stock || {};
  const totals = data.totals || {};
  const uplift = data.uplift || {};

  const tag = (v)=> {
    if(v==null) return '—';
    const n = Number(v), cls = n>=0 ? 'delta-pos' : 'delta-neg', sign = n>=0 ? '+' : '−';
    return `<span class="${cls}">${sign}${Math.abs(n).toFixed(2)}</span>`;
  };

  out.innerHTML = `
    <div style="margin:2px 0 6px">
      <span class="mode-chip">${chosen.cost_mode}</span>
      <span class="mode-chip">${chosen.optimizer}</span>
    </div>
    <div class="kpi-grid">
      <div class="kpi"><div class="label">Total Profit (score)</div><div class="value">${chosen.score!=null? Number(chosen.score).toFixed(2):'—'}</div></div>
      <div class="kpi"><div class="label">Δ Profit vs Baseline</div><div class="value">${tag(chosen.delta_profit)}</div></div>
      <div class="kpi"><div class="label">Weekly Stock</div><div class="value">${fmtUnits(stock.needed)}</div><div class="sub">+ buffer ${((stock.buffer_pct||0)*100).toFixed(0)}% → ${fmtUnits(stock.needed_with_buffer)}</div></div>
      <div class="kpi"><div class="label">Total Revenue</div><div class="value">${totals.revenue!=null? Number(totals.revenue).toFixed(2):'—'}</div></div>
    </div>
    <div class="smallnote">${chosen.message || ''}</div>
  `;

  // plan table
  const header = document.createElement('div'); header.className='flex-row header';
  const add = (row, txt, cls='cell')=>{ const d=document.createElement('div'); d.className=cls; d.textContent=txt; row.appendChild(d); };
  add(header,'Date'); add(header,'Best Price','cell right'); add(header,'Sales','cell right'); add(header,'Revenue','cell right'); add(header,'Profit','cell right'); add(header,'Fallback','cell small');
  planHost.appendChild(header);

  (data.plan||[]).forEach((r,i)=>{
    const row=document.createElement('div'); row.className='flex-row'+(i%2?' row-alt':'');
    add(row, r.date||'');
    add(row, r.best_price!=null? Number(r.best_price).toFixed(2):'—','cell right');
    add(row, r.expected_sales!=null? (qmode==='int'? String(Math.round(Number(r.expected_sales))) : Number(r.expected_sales).toFixed(2)) :'—','cell right');
    add(row, r.expected_revenue!=null? Number(r.expected_revenue).toFixed(2):'—','cell right');
    add(row, r.expected_profit!=null? Number(r.expected_profit).toFixed(2):'—','cell right');
    add(row, r.fallback? 'yes':'no','cell small');
    planHost.appendChild(row);
  });
  
  const memoHost = document.getElementById('autoMemo');
  if(memoHost){
    memoHost.textContent = data.memo ? JSON.stringify(data.memo, null, 2) : '—';
  }
  
}

// --- AUTOPILOT button
document.getElementById('btnAuto')?.addEventListener('click', async ()=>{
  const out=document.getElementById('autoOut'); out.innerHTML='<div class="smallnote">Running…</div>';
  try{
    const promoRaw=document.getElementById('cs_promo').value.trim();
    const modes = Array.from(document.querySelectorAll('.mchk:checked')).map(x=>x.value);
    const body={
      store_id: document.getElementById('cs_store').value.trim(),
      item_id:  document.getElementById('cs_item').value.trim(),
      start_date: document.getElementById('cs_date').value.trim(),
      price_min: Number(fix2(document.getElementById('cs_pmin').value.trim())),
      price_max: Number(fix2(document.getElementById('cs_pmax').value.trim())),
      price_step: Number(fix2(document.getElementById('cs_pstep').value.trim())),
      promo: toPromo7(promoRaw),
      wac_window_days: parseInt(document.getElementById('cs_wacwin').value.trim(),10) || 90,
      quantity_mode: document.getElementById('cs_qmode').value,
      modes: modes,
      buffer_pct: (Number(document.getElementById('cs_buffer').value.trim()) || 0) / 100.0
    };
    const ucostRaw=document.getElementById('cs_ucost').value.trim();
    const eucostRaw=document.getElementById('cs_eucost').value.trim();
    const efcostRaw=document.getElementById('cs_efcost').value.trim();
    const uArr = parseArr(ucostRaw); const euArr=parseArr(eucostRaw); const efArr=parseArr(efcostRaw);
    if(ucostRaw) body.unit_cost = uArr ? uArr.map(Number) : Number(fix2(ucostRaw));
    if(eucostRaw) body.extra_unit_cost = euArr ? euArr.map(Number) : Number(fix2(eucostRaw));
    if(efcostRaw) body.extra_fixed_cost = efArr ? efArr.map(Number) : Number(fix2(efcostRaw));

    const data = await postJSON('/autopilot', body);
    renderAutopilot(data);
  }catch(err){
    out.innerHTML = `<div class="smallnote delta-neg">Error: ${err.message}</div>`;
    document.getElementById('autoPlan').innerHTML='';
    document.getElementById('autoMemo').textContent = '—';
  }
});

async function autofillFromStats(prefer){  // prefer: 'predict' | 'sweep' | 'compare' | undefined
  try{
    const pairs = {
      compare: ['cs_store','cs_item'],
      sweep:   ['s2','i2'],
      predict: ['store_id','item_id'],
    };
    const getVal = id => document.getElementById(id)?.value?.trim();

    const getPair = (keys)=>{
      if(!keys) return null;
      const s = getVal(keys[0]);
      const i = getVal(keys[1]);
      return (s && i) ? {store:s, item:i, keys} : null;
    };

    // 1) try the explicitly preferred pair
    let chosen = prefer ? getPair(pairs[prefer]) : null;
    // 2) otherwise: use the first pair that has BOTH values filled, in this order:
    if(!chosen){
      chosen = getPair(pairs.compare) || getPair(pairs.sweep) || getPair(pairs.predict);
    }
    if(!chosen) return;

    const r = await fetch(`/price_stats?store_id=${encodeURIComponent(chosen.store)}&item_id=${encodeURIComponent(chosen.item)}`);
    if(!r.ok) return;
    const s = await r.json();

    // Update the canonical sweep range (pmin/pmax/pstep)
    const pmin = Number(s.suggested_min).toFixed(2);
    const pmax = Number(s.suggested_max).toFixed(2);
    const step = Number(s.suggested_step ?? 0.10).toFixed(2);
    const pminEl = document.getElementById('pmin');
    const pmaxEl = document.getElementById('pmax');
    const pstepEl= document.getElementById('pstep');
    if(pminEl) pminEl.value = pmin;
    if(pmaxEl) pmaxEl.value = pmax;
    if(pstepEl)pstepEl.value = step;

    // Fill Predict price ONLY when:
    // - the Predict form was the source (prefer==='predict'), OR
    // - the price field is currently empty (so we don't stomp user input)
    const priceEl = document.getElementById('price');
    if(priceEl && (prefer === 'predict' || !priceEl.value)){
      priceEl.value = Number(s.last_price ?? s.p50).toFixed(2);
    }

    // Mirror range to Compare form fields
    syncRange();
  }catch(_){}
}

async function postJSON(url, body){
  const r = await fetch(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){
    const txt = await r.text();
    throw new Error(txt || (r.status+' '+r.statusText));
  }
  return r.json();
}

// ---------- Percentile autofill + sync for ranges ----------
function syncRange(){
  const pmin = document.getElementById('pmin')?.value;
  const pmax = document.getElementById('pmax')?.value;
  const pstep= document.getElementById('pstep')?.value;
  const csmin=document.getElementById('cs_pmin');
  const csmax=document.getElementById('cs_pmax');
  const csstep=document.getElementById('cs_pstep');
  if(csmin && pmin!=null) csmin.value = pmin;
  if(csmax && pmax!=null) csmax.value = pmax;
  if(csstep && pstep!=null) csstep.value = pstep;
}

/* ------------------------------------------------------------ */

document.getElementById('f1').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const out=document.getElementById('out'); out.textContent='Predicting...';
  try{
    const priceRaw=document.getElementById('price').value.trim();
    const promoRaw=document.getElementById('promo').value.trim();
    const priceArr=parseArr(priceRaw); 
    const qmode=document.getElementById('qmode1').value;
    const body={
      store_id: document.getElementById('store_id').value.trim(),
      item_id:  document.getElementById('item_id').value.trim(),
      start_date: document.getElementById('start_date').value.trim(),
      price: priceArr ? priceArr.map(fix2).map(Number) : Number(fix2(priceRaw)),
      promo: toPromo7(promoRaw),
      quantity_mode: qmode
    };
    const data = await postJSON('/predict_next_7days', body);
    out.textContent = JSON.stringify(data, null, 2);
    renderForecastTable(data, qmode);
  }catch(err){
    out.textContent = 'Error: '+ err.message;
    document.getElementById('fcFlex').querySelectorAll('.flex-row.data').forEach(el=>el.remove());
    document.getElementById('fcNeed').textContent='—';
  }
});

document.getElementById('f2').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const out=document.getElementById('out2'); out.textContent='Running sweep...';
  try{
    const promoRaw=document.getElementById('pm2').value.trim();
    const ucostRaw=document.getElementById('ucost').value.trim();
    const eucostRaw=document.getElementById('eucost').value.trim();
    const efcostRaw=document.getElementById('efcost').value.trim();
    const uArr=parseArr(ucostRaw);
    const euArr=parseArr(eucostRaw);
    const efArr=parseArr(efcostRaw);
    const qmode=document.getElementById('qmode2').value;
    const body={
      store_id: document.getElementById('s2').value.trim(),
      item_id:  document.getElementById('i2').value.trim(),
      start_date: document.getElementById('d2').value.trim(),
      price_min: Number(fix2(document.getElementById('pmin').value.trim())),
      price_max: Number(fix2(document.getElementById('pmax').value.trim())),
      price_step: Number(fix2(document.getElementById('pstep').value.trim())),
      promo: toPromo7(promoRaw),
      optimize_by: document.getElementById('optby').value.trim(),
      cost_mode: document.getElementById('cmode').value.trim(),
      wac_window_days: parseInt(document.getElementById('wacwin').value.trim(),10) || 90,
      unit_cost: uArr ? uArr.map(Number) : (ucostRaw? Number(fix2(ucostRaw)) : null),
      extra_unit_cost: euArr ? euArr.map(Number) : (eucostRaw? Number(fix2(eucostRaw)) : null),
      extra_fixed_cost: efArr ? efArr.map(Number) : (efcostRaw? Number(fix2(efcostRaw)) : null),
      quantity_mode: qmode
    };
    const data = await postJSON('/price_sweep', body);
    out.textContent = JSON.stringify(data, null, 2);
    renderFlex(data.recommended, qmode, data.uplift || null);
  }catch(err){
    out.textContent = 'Error: '+ err.message;
    document.getElementById('recFlex').querySelectorAll('.flex-row.data').forEach(el=>el.remove());
    document.getElementById('recNeed').textContent='—';
    document.getElementById('recBenefit').textContent='—';
  }
});

document.getElementById('f3').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const out=document.getElementById('cmpOut'); out.textContent='Comparing...';
  try{
    const promoRaw=document.getElementById('cs_promo').value.trim();
    const modes = Array.from(document.querySelectorAll('.mchk:checked')).map(x=>x.value);
    const body={
      store_id: document.getElementById('cs_store').value.trim(),
      item_id:  document.getElementById('cs_item').value.trim(),
      start_date: document.getElementById('cs_date').value.trim(),
      price_min: Number(fix2(document.getElementById('cs_pmin').value.trim())),
      price_max: Number(fix2(document.getElementById('cs_pmax').value.trim())),
      price_step: Number(fix2(document.getElementById('cs_pstep').value.trim())),
      promo: toPromo7(promoRaw),
      wac_window_days: parseInt(document.getElementById('cs_wacwin').value.trim(),10) || 90,
      quantity_mode: document.getElementById('cs_qmode').value,
      modes: modes
    };
    const ucostRaw=document.getElementById('cs_ucost').value.trim();
    const eucostRaw=document.getElementById('cs_eucost').value.trim();
    const efcostRaw=document.getElementById('cs_efcost').value.trim();
    const uArr = parseArr(ucostRaw); const euArr=parseArr(eucostRaw); const efArr=parseArr(efcostRaw);
    if(ucostRaw) body.unit_cost = uArr ? uArr.map(Number) : Number(fix2(ucostRaw));
    if(eucostRaw) body.extra_unit_cost = euArr ? euArr.map(Number) : Number(fix2(eucostRaw));
    if(efcostRaw) body.extra_fixed_cost = efArr ? efArr.map(Number) : Number(fix2(efcostRaw));
    const data = await postJSON('/compare_cost_modes', body);
    renderSummaryTable(data);
    renderCompareGrid(data);
    renderDecision(data);
  }catch(err){
    out.textContent = 'Error: '+ err.message;
    document.getElementById('cmpGrid').innerHTML='';
    document.getElementById('decOut').textContent = '—';
  }
});


// Listeners: call with the right context
;['store_id','item_id'].forEach(id=>{
  const el = document.getElementById(id);
  if(el) el.addEventListener('change', ()=>autofillFromStats('predict').then(syncRange));
});
;['s2','i2'].forEach(id=>{
  const el = document.getElementById(id);
  if(el) el.addEventListener('change', ()=>autofillFromStats('sweep').then(syncRange));
});
;['cs_store','cs_item'].forEach(id=>{
  const el = document.getElementById(id);
  if(el) el.addEventListener('change', ()=>autofillFromStats('compare').then(syncRange));
});

// On load: pick whichever pair is filled
autofillFromStats().then(syncRange);

</script>
</body></html>
"""
    # UI Start Date = next day after the last history date
    ui_start = (HIST["date"].max() + pd.Timedelta(days=1)).date().isoformat()
    hol_label = HOL_COUNTRY + (("-" + HOL_SUBDIV) if HOL_SUBDIV else "")
    return html.replace("__HOL__", hol_label).replace("__START__", ui_start)

# --------------------------- Endpoints --------------------------
class _NextDayCSVRow(csv.DictWriter): pass  # quiet linters

@app.post("/predict_next_day")
def predict_next_day(req: NextDayReq):
    qmode = validate_qmode(req.quantity_mode)
    price2 = round_money(req.price)
    start = pd.Timestamp(req.date) - pd.Timedelta(days=1)
    try:
        base, info = base_features_for(req.store_id, req.item_id, start)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if info["hist_len"] < 14:
        yhat = float(np.nan_to_num(info["ma7"], nan=info["ma28"])) if not np.isnan(info["ma7"]) else info["ma28"]
        if np.isnan(yhat): yhat = GLOBAL_MEAN
        sales_q = quantize_sales(yhat, qmode)
        result = {"store_id": req.store_id, "item_id": req.item_id, "date": req.date,
                  "price": price2, "promo": int(req.promo), "predicted_sales": sales_q,
                  "fallback": True, "holidays": HOL_COUNTRY + (("-" + HOL_SUBDIV) if HOL_SUBDIV else ""),
                  "quantity_mode": qmode}
    else:
        row = row_for_h(base, 1, pd.Timestamp(req.date), price2, req.promo)
        X = pd.DataFrame([row])[FEATURES[1]]
        yhat = float(np.expm1(MODELS[1].predict(xgb.DMatrix(X))[0]))
        sales_q = quantize_sales(yhat, qmode)
        result = {"store_id": req.store_id, "item_id": req.item_id, "date": req.date,
                  "price": price2, "promo": int(req.promo), "predicted_sales": sales_q,
                  "fallback": False, "holidays": HOL_COUNTRY + (("-" + HOL_SUBDIV) if HOL_SUBDIV else ""),
                  "quantity_mode": qmode}

    header = ["date", "store_id", "item_id", "price", "promo", "predicted_sales", "fallback"]
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow({k: result[k] for k in header})
    return result

@app.post("/predict_next_7days")
def predict_next_7days(req: Next7Req):
    qmode = validate_qmode(req.quantity_mode)
    prices = round_money_list(req.price, 7)
    promos = req.promo if isinstance(req.promo, list) else [req.promo] * len(H_LIST)
    try:
        promos = [int(x) for x in promos]
    except Exception:
        raise HTTPException(400, "promo must be 0/1 (single or list of 7).")
    if len(promos) != len(H_LIST):
        raise HTTPException(400, f"promo must be scalar or list of length {len(H_LIST)}")
    if any(x not in (0, 1) for x in promos):
        raise HTTPException(400, "promo values must be 0 or 1.")


    start_date = pd.Timestamp(req.start_date)
    asof = start_date - pd.Timedelta(days=1)
    try:
        base, info = base_features_for(req.store_id, req.item_id, asof)
    except ValueError as e:
        raise HTTPException(400, str(e))

    out = []
    if info["hist_len"] < 14:
        y0 = float(np.nan_to_num(info["ma7"], nan=info["ma28"])) if not np.isnan(info["ma7"]) else info["ma28"]
        if np.isnan(y0): y0 = GLOBAL_MEAN
        for idx, h in enumerate(H_LIST):
            d = start_date + timedelta(days=h - 1)
            sales_q = quantize_sales(y0, qmode)
            out.append({"date": str(d.date()), "predicted_sales": sales_q,
                        "price": prices[idx], "promo": promos[idx], "fallback": True})
    else:
        for idx, h in enumerate(H_LIST):
            d = start_date + timedelta(days=h - 1)
            row = row_for_h(base, h, d, prices[idx], promos[idx])
            X = pd.DataFrame([row])[FEATURES[h]]
            yhat = float(np.expm1(MODELS[h].predict(xgb.DMatrix(X))[0]))
            sales_q = quantize_sales(yhat, qmode)
            out.append({"date": str(d.date()), "predicted_sales": sales_q,
                        "price": prices[idx], "promo": promos[idx], "fallback": False})

    # ---- NEW: weekly totals for the entered price schedule ----
    tot_units = Decimal("0.00")
    tot_rev   = Decimal("0.00")
    for idx, r in enumerate(out):
        s = Decimal(str(r["predicted_sales"]))
        p = Decimal(str(prices[idx]))
        tot_units += s
        tot_rev   += Decimal(str(round_money_any(float(p * s))))

    # ---- NEW: compute baseline (hold last hist price) revenue ----
    baseline_price = last_hist_price(req.store_id, req.item_id, asof)
    baseline = None
    uplift = None
    if baseline_price is not None:
        b_units = Decimal("0.00")
        b_rev   = Decimal("0.00")
        for idx, h in enumerate(H_LIST):
            d = start_date + timedelta(days=h - 1)
            s_q, r_v, _p, _fb = _score_day_with_qmode(
                base, info, h, d, float(baseline_price), promos[idx], qmode,
                unit_cost=None, extra_unit_cost=0.0, extra_fixed_cost=0.0
            )
            b_units += Decimal(str(s_q))
            b_rev   += Decimal(str(r_v))

        baseline = {
            "price": float(round(baseline_price, 2)),
            "total_units": float(b_units.quantize(Decimal("0.01"))),
            "total_revenue": float(b_rev.quantize(Decimal("0.01"))),
        }
        uplift = {
            "units": float((tot_units - b_units).quantize(Decimal("0.01"))),
            "revenue": float((tot_rev - b_rev).quantize(Decimal("0.01")))
        }

    return {
        "store_id": req.store_id, "item_id": req.item_id, "start_date": req.start_date,
        "horizons_days": H_LIST, "forecasts": out,
        "holidays": HOL_COUNTRY + (("-" + HOL_SUBDIV) if HOL_SUBDIV else ""),
        "quantity_mode": qmode,
        # NEW: expose weekly totals + baseline/uplift for the UI
        "total_expected_units": float(tot_units.quantize(Decimal("0.01"))),
        "total_expected_revenue": float(tot_rev.quantize(Decimal("0.01"))),
        "baseline": baseline,
        "uplift": uplift,
    }           

# -------------------- Price Sweep (Elasticity) ------------------
def _price_grid(pmin: float, pmax: float, pstep: float) -> List[float]:
    dmin = Decimal(str(pmin)); dmax = Decimal(str(pmax)); dstep = Decimal(str(pstep))
    if dmin < 0 or dmax < dmin or dstep <= 0: raise HTTPException(400, "Invalid price range or step.")
    if dstep < Decimal("0.01"): raise HTTPException(400, "price_step must be at least 0.01")
    prices = []; x = dmin; limit = 400
    while x <= dmax + Decimal("0.0000001"):
        prices.append(round_money(float(x))); x += dstep
        if len(prices) > limit: raise HTTPException(400, f"Too many price points (> {limit}).")
    prices = sorted(set(prices))
    if not prices: prices = [round_money(float(dmin))]
    return prices

@app.post("/price_sweep")
def price_sweep(req: PriceSweepReq):
    qmode = validate_qmode(req.quantity_mode)

    promos = req.promo if isinstance(req.promo, list) else [req.promo] * len(H_LIST)
    try:
        promos = [int(x) for x in promos]
    except Exception:
        raise HTTPException(400, "promo must be 0/1 (single or list of 7).")
    if len(promos) != len(H_LIST):
        raise HTTPException(400, f"promo must be scalar or list of length {len(H_LIST)}")
    if any(x not in (0, 1) for x in promos):
        raise HTTPException(400, "promo values must be 0 or 1.")

    if req.optimize_by not in {"revenue", "profit"}:
        raise HTTPException(400, "optimize_by must be 'revenue' or 'profit'")

    start_date = pd.Timestamp(req.start_date)
    asof = start_date - pd.Timedelta(days=1)

    # Resolve unit costs by cost_mode (or manual)
    unit_costs = None
    if (req.cost_mode or "manual").lower() != "manual":
        unit_costs = resolve_unit_costs(req.cost_mode, req.item_id, start_date, req.wac_window_days)
        if req.optimize_by == "profit" and unit_costs is None:
            raise HTTPException(400, f"Cannot compute unit_cost via cost_mode='{req.cost_mode}'. "
                                     f"Provide purchases.csv coverage or switch to 'manual'.")
    else:
        unit_costs = normalize_cost_list(req.unit_cost, len(H_LIST), "unit_cost", allow_none=(req.optimize_by=="revenue"))

    if req.optimize_by == "profit" and unit_costs is None:
        raise HTTPException(400, "unit_cost required for profit optimization (provide manually or choose replacement/wac/standard).")

    extra_unit_costs = normalize_cost_list(req.extra_unit_cost, len(H_LIST), "extra_unit_cost", allow_none=True, default_val=0.0)
    extra_fixed_costs = normalize_cost_list(req.extra_fixed_cost, len(H_LIST), "extra_fixed_cost", allow_none=True, default_val=0.0)
    if extra_unit_costs is None: extra_unit_costs = [0.0] * len(H_LIST)
    if extra_fixed_costs is None: extra_fixed_costs = [0.0] * len(H_LIST)

    prices = _price_grid(req.price_min, req.price_max, req.price_step)

    try:
        base, info = base_features_for(req.store_id, req.item_id, asof)
    except ValueError as e:
        raise HTTPException(400, str(e))

    horizons_out: List[Dict[str, Any]] = []
    total_metric = Decimal("0.00"); total_rev = Decimal("0.00"); total_profit = Decimal("0.00")
    total_units = Decimal("0.00")

    def score_point(p: float, sales_raw: float, hidx: int):
        sales = quantize_sales(sales_raw, qmode)
        revenue = round_money(max(0.0, p * sales))
        profit_val = None
        if unit_costs is not None:
            eu = extra_unit_costs[hidx]; ef = extra_fixed_costs[hidx]
            margin_unit = p - unit_costs[hidx] - eu
            profit_val = round_money_any(margin_unit * sales - ef)
        return sales, revenue, profit_val

    for idx, h in enumerate(H_LIST):
        d = start_date + timedelta(days=h - 1)
        curve = []; best_metric = -1e309
        best = {"price": prices[0], "sales": 0.0, "revenue": 0.0, "profit": (0.0 if unit_costs is not None else None)}

        if info["hist_len"] < 14:
            y0 = float(np.nan_to_num(info["ma7"], nan=info["ma28"])) if not np.isnan(info["ma7"]) else info["ma28"]
            if np.isnan(y0): y0 = GLOBAL_MEAN
            for p in prices:
                sales_q, revenue, profit_val = score_point(p, y0, idx)
                curve.append({"price": p, "sales": sales_q, "revenue": revenue, "profit": profit_val})
                metric = (profit_val if req.optimize_by == "profit" else revenue)
                if metric > best_metric: best_metric = metric; best = {"price": p, "sales": sales_q, "revenue": revenue, "profit": profit_val}
            fallback_flag = True
        else:
            for p in prices:
                row = row_for_h(base, h, d, p, promos[idx])
                X = pd.DataFrame([row])[FEATURES[h]]
                pred = float(np.expm1(MODELS[h].predict(xgb.DMatrix(X))[0]))
                sales_q, revenue, profit_val = score_point(p, pred, idx)
                curve.append({"price": p, "sales": sales_q, "revenue": revenue, "profit": profit_val})
                metric = (profit_val if req.optimize_by == "profit" else revenue)
                if metric > best_metric: best_metric = metric; best = {"price": p, "sales": sales_q, "revenue": revenue, "profit": profit_val}
            fallback_flag = False

        horizons_out.append({"date": str(d.date()), "horizon": h, "curve": curve, "best": best, "fallback": fallback_flag})
        total_metric += Decimal(str(best["profit"] if req.optimize_by == "profit" else best["revenue"]))
        total_rev += Decimal(str(best["revenue"]))
        total_units += Decimal(str(best["sales"]))
        if unit_costs is not None: total_profit += Decimal(str(best["profit"] or 0.0))

    # ----- Baseline (hold last historical price constant) -----
    baseline_price = last_hist_price(req.store_id, req.item_id, asof)
    baseline = None
    uplift = None
    if baseline_price is not None:
        b_units = Decimal("0.00")
        b_rev   = Decimal("0.00")
        b_prof  = Decimal("0.00") if (unit_costs is not None) else None

        for idx, h in enumerate(H_LIST):
            d = start_date + timedelta(days=h - 1)
            ucost = unit_costs[idx] if unit_costs is not None else None
            euc   = extra_unit_costs[idx]
            efc   = extra_fixed_costs[idx]
            s_q, r_v, p_v, _ = _score_day_with_qmode(
                base, info, h, d, float(baseline_price), promos[idx], qmode, ucost, euc, efc
            )
            b_units += Decimal(str(s_q))
            b_rev   += Decimal(str(r_v))
            if b_prof is not None:
                b_prof += Decimal(str(p_v or 0.0))

        baseline = {
            "price": float(round(baseline_price, 2)),
            "total_units": float(b_units.quantize(Decimal("0.01"))),
            "total_revenue": float(b_rev.quantize(Decimal("0.01"))),
        }
        if b_prof is not None:
            baseline["total_profit"] = float(b_prof.quantize(Decimal("0.01")))

        uplift = {
            "units": float((total_units - b_units).quantize(Decimal("0.01"))),
            "revenue": float((total_rev - b_rev).quantize(Decimal("0.01")))
        }
        if unit_costs is not None:
            uplift["profit"] = float((total_profit - (b_prof or Decimal("0.00"))).quantize(Decimal("0.01")))

    resp = {
        "store_id": req.store_id, "item_id": req.item_id, "start_date": req.start_date,
        "horizons_days": H_LIST,
        "dates": [ (pd.Timestamp(req.start_date)+timedelta(days=h-1)).date().isoformat() for h in H_LIST ],
        "prices": prices, "results": horizons_out,
        "recommended": [
            {"date": h["date"], "horizon": h["horizon"], "best_price": h["best"]["price"],
             "expected_sales": h["best"]["sales"], "expected_revenue": h["best"]["revenue"],
             "expected_profit": h["best"]["profit"], "fallback": h["fallback"]}
            for h in horizons_out
        ],
        "optimize_by": req.optimize_by, "cost_mode": req.cost_mode, "wac_window_days": req.wac_window_days,
        "unit_cost": unit_costs, "extra_unit_cost": extra_unit_costs, "extra_fixed_cost": extra_fixed_costs,
        "quantity_mode": qmode,
        "metric_label": "profit" if req.optimize_by == "profit" else "revenue",
        "total_expected_metric": float(total_metric.quantize(Decimal("0.01"))),
        "total_expected_units": float(total_units.quantize(Decimal("0.01"))),
        "total_expected_revenue": float(total_rev.quantize(Decimal("0.01"))),
        "holidays": HOL_COUNTRY + (("-" + HOL_SUBDIV) if HOL_SUBDIV else ""),
        "baseline": baseline,
        "uplift": uplift,
    }
    if unit_costs is not None:
        resp["total_expected_profit"] = float(total_profit.quantize(Decimal("0.01")))
    return resp

# ------------------------ Comparison endpoint -------------------
@app.post("/compare_cost_modes")
def compare_cost_modes(payload: dict = Body(...)):
    """
    For each requested cost mode, run TWO plans:
      - revenue optimizer (optimize_by='revenue') with that cost mode (we also compute profit if costs resolve)
      - profit   optimizer (optimize_by='profit')   with that cost mode

    Each plan also gets its own baseline: last historical price (as-of start_date-1) held for the week.
    We compute uplifts: ΔUnits, ΔRevenue, ΔProfit (ΔProfit may be None if costs are unavailable).

    Decision helper (safe default):
      - For EACH cost mode, pick the plan that maximizes ΔProfit.
      - Ties broken by ΔRevenue, then ΔUnits.
      - If ΔProfit is None (no costs), that plan is skipped for scoring.
    """
    try:
        base = dict(payload)
        modes = base.pop("modes", ["replacement", "wac", "standard"])

        store_id       = str(base.get("store_id"))
        item_id        = str(base.get("item_id"))
        start_date_str = str(base.get("start_date"))
        try:
            start_date = pd.Timestamp(start_date_str)
        except Exception:
            raise HTTPException(400, "Invalid start_date")
        asof = start_date - pd.Timedelta(days=1)

        def _as_float(x, default=None):
            try: return float(x)
            except Exception: return default

        price_min  = _as_float(base.get("price_min"), 0.0)
        price_max  = _as_float(base.get("price_max"), 0.0)
        price_step = _as_float(base.get("price_step"), 0.10)
        promo      = base.get("promo", 0)
        wac_win    = int(base.get("wac_window_days", 90))
        qmode      = str(base.get("quantity_mode", QMODE_DEFAULT)).lower()

        unit_cost        = base.get("unit_cost")
        extra_unit_cost  = base.get("extra_unit_cost")
        extra_fixed_cost = base.get("extra_fixed_cost")

        promos = promo if isinstance(promo, list) else [promo] * len(H_LIST)
        if len(promos) != len(H_LIST):
            raise HTTPException(400, f"promo must be scalar or list of length {len(H_LIST)}")
        promos = [int(x) for x in promos]

        baseline_price = last_hist_price(store_id, item_id, asof)
        if baseline_price is None:
            raise HTTPException(400, "No baseline price available up to start_date-1.")

        def run_one(cost_mode: str, optimizer: str):
            """Run price_sweep for (cost_mode, optimizer) + baseline."""
            req = PriceSweepReq(
                store_id=store_id, item_id=item_id, start_date=start_date_str,
                price_min=price_min, price_max=price_max, price_step=price_step,
                promo=promos, optimize_by=optimizer, cost_mode=cost_mode,
                wac_window_days=wac_win, unit_cost=unit_cost,
                extra_unit_cost=extra_unit_cost, extra_fixed_cost=extra_fixed_cost,
                quantity_mode=qmode
            )
            res = price_sweep(req)

            base_req = PriceSweepReq(
                store_id=store_id, item_id=item_id, start_date=start_date_str,
                price_min=baseline_price, price_max=baseline_price, price_step=0.01,
                promo=promos, optimize_by=optimizer, cost_mode=cost_mode,
                wac_window_days=wac_win, unit_cost=unit_cost,
                extra_unit_cost=extra_unit_cost, extra_fixed_cost=extra_fixed_cost,
                quantity_mode=qmode
            )
            res_base = price_sweep(base_req)

            row = {
                "mode": f"{cost_mode} · {optimizer}",
                "cost_mode": cost_mode,
                "optimizer": optimizer,
                "unit_cost": res.get("unit_cost"),
                "total_units": _safe_float(res.get("total_expected_units")),
                "total_revenue": _safe_float(res.get("total_expected_revenue")),
                "total_profit":  _safe_float(res.get("total_expected_profit")),
                "total_metric":  _safe_float(res.get("total_expected_metric")),
                "best_prices":   [r.get("best_price") for r in (res.get("recommended") or [])],
                "baseline_total_units": _safe_float(res_base.get("total_expected_units")),
                "baseline_total_revenue": _safe_float(res_base.get("total_expected_revenue")),
                "baseline_total_profit":  _safe_float(res_base.get("total_expected_profit")),
            }

            def diff(a, b):
                if a is None or b is None: return None
                return float(Decimal(str(a - b)).quantize(Decimal("0.01")))
            row["uplift_units"]   = diff(row["total_units"],   row["baseline_total_units"])
            row["uplift_revenue"] = diff(row["total_revenue"], row["baseline_total_revenue"])
            row["uplift_profit"]  = diff(row["total_profit"],  row["baseline_total_profit"])
            return res, row

        compare_rows: List[Dict[str, Any]] = []
        dates = None
        for m in modes:
            # revenue-optimized
            try:
                res_rev, row_rev = run_one(m, "revenue")
                if dates is None: dates = [r["date"] for r in (res_rev.get("recommended") or [])]
                compare_rows.append(row_rev)
            except HTTPException as e:
                compare_rows.append({"mode": f"{m} · revenue", "cost_mode": m, "optimizer": "revenue", "error": e.detail})
            # profit-optimized
            try:
                res_prf, row_prf = run_one(m, "profit")
                if dates is None: dates = [r["date"] for r in (res_prf.get("recommended") or [])]
                compare_rows.append(row_prf)
            except HTTPException as e:
                compare_rows.append({"mode": f"{m} · profit", "cost_mode": m, "optimizer": "profit", "error": e.detail})

        # ---- Decision helper per cost mode
        # Score = TOTAL weekly profit; tie-break on TOTAL revenue, then TOTAL units
        decisions: List[Dict[str, Any]] = []

        def _tie_key(r: Dict[str, Any]):
            # Tuple for max(): (profit, revenue, units), treating None as very small
            pf = r.get("decision_score")
            rv = _safe_float(r.get("total_revenue"))
            un = _safe_float(r.get("total_units"))
            return (
                pf if pf is not None else -1e309,
                rv if rv is not None else -1e309,
                un if un is not None else -1e309,
            )

        for m in modes:
            # Candidates for this cost mode (no errors)
            cand = [r for r in compare_rows if (r.get("cost_mode") == m and not r.get("error"))]

            # Decision score = TOTAL weekly profit (not uplift)
            for r in cand:
                tp = r.get("total_profit")
                r["decision_score"] = (
                    float(Decimal(str(tp)).quantize(Decimal("0.01"))) if tp is not None else None
                )

            picked = None
            for r in cand:
                if r.get("decision_score") is None:
                    continue
                if picked is None or _tie_key(r) > _tie_key(picked):
                    picked = r

            if picked is None:
                decisions.append({
                    "cost_mode": m,
                    "chosen_optimizer": None,
                    "score": None,  # total profit
                    "total_revenue": None,
                    "total_units": None,
                    "baseline_stock": None,
                    "baseline_revenue": None,
                    "baseline_profit": None,
                    "delta_units": None,
                    "delta_revenue": None,
                    "delta_profit": None,
                    "message": f"No decision for '{m}': missing cost info → total profit unavailable."
                })
            else:
                decisions.append({
                    "cost_mode": m,
                    "chosen_optimizer": picked.get("optimizer"),
                    "score": picked.get("decision_score"),  # total weekly profit
                    "total_revenue": picked.get("total_revenue"),
                    "total_units": picked.get("total_units"),
                    "baseline_stock": picked.get("baseline_total_units"),
                    "baseline_revenue": picked.get("baseline_total_revenue"),
                    "baseline_profit": picked.get("baseline_total_profit"),
                    "delta_units": picked.get("uplift_units"),
                    "delta_revenue": picked.get("uplift_revenue"),
                    "delta_profit": picked.get("uplift_profit"),
                    "message": (
                        f"'{m}' → choose {picked.get('optimizer')} optimizer; "
                        f"maximizes TOTAL weekly profit = {picked.get('decision_score'):.2f}. "
                        f"Stock needed {picked.get('total_units')} "
                        f"(baseline {picked.get('baseline_total_units')}, Δ {picked.get('uplift_units')})."
                    )
                })

        return {"dates": dates, "compare": compare_rows, "decisions": decisions}


    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/compare_cost_modes crashed: {e}\n{traceback.format_exc()}")


# ------------------------ Autopilot endpoint --------------------
@app.post("/autopilot")
def autopilot(payload: dict = Body(...)):
    """
    One-shot: choose best plan and return daily prices + weekly stock.
    Rule: per cost mode, optimizer has already been chosen by TOTAL weekly profit in /compare_cost_modes
          (tie on TOTAL revenue, then TOTAL units).
    Here we pick the best cost mode by the same rule across decisions.
    Optional: `buffer_pct` (e.g., 0.10 for +10% safety stock).
    """
    base = dict(payload)
    modes = base.pop("modes", ["replacement", "wac", "standard"])
    buffer_pct = float(base.pop("buffer_pct", 0.0) or 0.0)

    # run comparison to get per-mode decisions
    cmp_res = compare_cost_modes({**base, "modes": modes})
    compare_rows = cmp_res.get("compare") or []
    decisions = cmp_res.get("decisions") or []

    def _sf(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    # pick best decision ACROSS cost modes:
    best = None
    for d in decisions:
        tp = _sf(d.get("score"))   # total profit from /compare_cost_modes
        tr = _sf(d.get("total_revenue"))
        tu = _sf(d.get("total_units"))
        if tp is None:
            continue
        if (
            best is None
            or tp > (_sf(best.get("score")) or -1e309)
            or (tp == (_sf(best.get("score")) or -1e309) and (tr or -1e309) > (_sf(best.get("total_revenue")) or -1e309))
            or (tp == (_sf(best.get("score")) or -1e309) and (tr or -1e309) == (_sf(best.get("total_revenue")) or -1e309)
                and (tu or -1e309) > (_sf(best.get("total_units")) or -1e309))
        ):
            best = d

    if best is None:
        raise HTTPException(400, "Autopilot could not choose: total profit unavailable (missing costs).")

    chosen_cost_mode = str(best["cost_mode"])
    chosen_optimizer = str(best.get("chosen_optimizer") or "profit")

    # final sweep for chosen pair to get the daily plan
    req = PriceSweepReq(
        store_id=str(base["store_id"]),
        item_id=str(base["item_id"]),
        start_date=str(base["start_date"]),
        price_min=float(base["price_min"]),
        price_max=float(base["price_max"]),
        price_step=float(base.get("price_step", 0.10) or 0.10),
        promo=base.get("promo", 0),
        optimize_by=chosen_optimizer,
        cost_mode=chosen_cost_mode,
        wac_window_days=int(base.get("wac_window_days", 90) or 90),
        unit_cost=base.get("unit_cost"),
        extra_unit_cost=base.get("extra_unit_cost"),
        extra_fixed_cost=base.get("extra_fixed_cost"),
        quantity_mode=str(base.get("quantity_mode") or QMODE_DEFAULT)
    )
    final = price_sweep(req)

    # weekly stock (+ buffer)
    qmode = final.get("quantity_mode") or QMODE_DEFAULT
    total_units = _sf(final.get("total_expected_units")) or 0.0
    stock_no_buf = total_units
    stock_with_buf = stock_no_buf * (1.0 + buffer_pct)
    if qmode == "int":
        stock_no_buf = float(Decimal(str(stock_no_buf)).to_integral_value(rounding=ROUND_HALF_UP))
        stock_with_buf = float(Decimal(str(stock_with_buf)).to_integral_value(rounding=ROUND_HALF_UP))
    else:
        stock_no_buf = float(Decimal(str(stock_no_buf)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        stock_with_buf = float(Decimal(str(stock_with_buf)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    # daily plan
    plan = [{
        "date": r["date"],
        "best_price": r["best_price"],
        "expected_sales": r["expected_sales"],
        "expected_revenue": r["expected_revenue"],
        "expected_profit": r["expected_profit"],
        "fallback": r["fallback"]
    } for r in (final.get("recommended") or [])]

    # ---- Build MEMO with explicit "expected_*" names ----
    # Baseline -> totals only + last price
    baseline_src = final.get("baseline") or {}
    baseline_totals = None
    if baseline_src:
        baseline_totals = {
            "units_week": baseline_src.get("total_units"),
            "revenue_week": baseline_src.get("total_revenue"),
            "profit_week": baseline_src.get("total_profit"),
        }

    # Uplift vs baseline
    uplift_src = final.get("uplift") or {}
    uplift_memo = None
    if uplift_src:
        uplift_memo = {
            "delta_units_week": uplift_src.get("units"),
            "delta_revenue_week": uplift_src.get("revenue"),
            "delta_profit_week": uplift_src.get("profit"),
        }

    # Expected weekly totals
    expected_weekly_totals = {
        "units_week": final.get("total_expected_units"),
        "revenue_week": final.get("total_expected_revenue"),
        "profit_week": final.get("total_expected_profit"),
    }

    # Expected daily plan with explicit names
    plan_expected = [{
        "date": r["date"],
        "price": r["best_price"],
        "expected_units": r["expected_sales"],
        "expected_revenue": r["expected_revenue"],
        "expected_profit": r["expected_profit"],
        "fallback": r["fallback"]
    } for r in (final.get("recommended") or [])]

    memo = {
        "selected_strategy": {
            "cost_mode": chosen_cost_mode,
            "optimizer": chosen_optimizer,
            "rule": "maximize total weekly profit; tie-break total revenue, then total units",
            "note": best.get("message"),
        },
        "expected_weekly_stock": {
            "units_needed": stock_no_buf,
            "buffer_pct": buffer_pct,
            "units_needed_with_buffer": stock_with_buf,
            "quantity_mode": qmode,
        },
        "expected_weekly_totals": expected_weekly_totals,
        "baseline": {
            "last_price": baseline_src.get("price"),
            "weekly_totals": baseline_totals
        } if baseline_totals is not None else None,
        "uplift_vs_baseline": uplift_memo,
        "expected_daily_plan": plan_expected,
        "dates": final.get("dates"),
    }

    # ---- Return both memo and existing fields (back-compat) ----
    return {
        "chosen": {
            "cost_mode": chosen_cost_mode,
            "optimizer": chosen_optimizer,
            "score": best.get("score"),    # TOTAL weekly profit
            "delta_revenue": best.get("delta_revenue"),
            "delta_profit": best.get("delta_profit"),
            "delta_units": best.get("delta_units"),
            "message": best.get("message"),
        },
        "weekly_stock": {
            "needed": stock_no_buf,
            "buffer_pct": buffer_pct,
            "needed_with_buffer": stock_with_buf
        },
        "totals": {
            "units": final.get("total_expected_units"),
            "revenue": final.get("total_expected_revenue"),
            "profit": final.get("total_expected_profit"),
        },
        "baseline": final.get("baseline"),
        "uplift": final.get("uplift"),
        "plan": plan,
        "dates": final.get("dates"),
        "quantity_mode": qmode,
        "memo": memo,
    }

# ------------------------ Price Stats Endpoint -------------------------------
@app.get("/price_stats")
def price_stats(store_id: str, item_id: str):
    """
    Percentile-based suggestions for sweep bounds and step,
    plus latest historical price to auto-fill Predict Price.
    """
    df = HIST[(HIST.store_id.astype(str) == str(store_id)) & (HIST.item_id.astype(str) == str(item_id))]
    if df.empty: raise HTTPException(404, "No history for that store_id/item_id")

    df = df.sort_values("date")
    n = int(len(df))
    last_price = float(round(df["price"].iloc[-1], 2))
    avg7  = float(round(df["price"].tail(7).mean(), 2))
    avg28 = float(round(df["price"].tail(28).mean(), 2))

    q = df["price"].quantile([0.05, 0.10, 0.50, 0.90, 0.95])
    p05 = float(round(q.loc[0.05], 2))
    p10 = float(round(q.loc[0.10], 2))
    p50 = float(round(q.loc[0.50], 2))
    p90 = float(round(q.loc[0.90], 2))
    p95 = float(round(q.loc[0.95], 2))

    # choose band
    if n >= 50:
        lo, hi = p10, p90
    elif n >= 20:
        lo, hi = p05, p95
    else:
        spread = max(0.5, (float(df["price"].max()) - float(df["price"].min())) * 0.5)
        lo, hi = p50 - spread/2, p50 + spread/2

    margin = 0.10
    raw_min = float(df["price"].min())
    raw_max = float(df["price"].max())
    suggested_min = float(round(max(raw_min, lo) - margin, 2))
    suggested_max = float(round(min(raw_max, hi) + margin, 2))

    step = 0.05 if suggested_max < 10 else (0.10 if suggested_max < 50 else 0.50)

    return {
        "count": n,
        "min": float(round(raw_min, 2)),
        "max": float(round(raw_max, 2)),
        "p05": p05, "p10": p10, "p50": p50, "p90": p90, "p95": p95,
        "suggested_min": suggested_min,
        "suggested_max": suggested_max,
        "suggested_step": step,
        "last_price": last_price,
        "avg7_price": avg7,
        "avg28_price": avg28
    }

# ------------------------ Debug purchases -----------------------
@app.get("/debug_cost_sources")
def debug_cost_sources(item_id: str, start_date: str, wac_window_days: int = 90):
    """Explain what the server sees for cost modes."""
    asof = pd.Timestamp(start_date) - pd.Timedelta(days=1)
    P = get_purchases_df()
    if P is None:
        return {"has_purchases": False, "reason": "missing/invalid data/purchases.csv"}
    iid = str(item_id).strip()
    sub = P[(P["item_id"] == iid) & (P["date"] <= asof)].sort_values("date")
    inwin = P[(P["item_id"] == iid) &
              (P["date"] >= asof - pd.Timedelta(days=wac_window_days - 1)) &
              (P["date"] <= asof)].sort_values("date")
    rep = replacement_cost_scalar(iid, asof)
    wac = wac_cost_scalar(iid, asof, wac_window_days)
    std = standard_cost_scalar(iid, asof)
    return {
        "as_of": str(asof.date()),
        "rows_total": int(P.shape[0]),
        "rows_for_item_le_asof": int(sub.shape[0]),
        "rows_for_item_in_window": int(inwin.shape[0]),
        "last_row_for_item": (sub.tail(1).to_dict(orient="records") if not sub.empty else None),
        "replacement_cost": rep,
        "wac_cost": wac,
        "standard_cost": std,
        "columns": list(P.columns),
        "mtime": os.path.getmtime(PURCHASES_PATH) if os.path.exists(PURCHASES_PATH) else None,
    }

# ---------------------- Optional GET quick test -----------------
@app.get("/predict_next_7days")
def predict_next_7days_get(store_id: str, item_id: str, start_date: str, price: float = 29.90, promo: int = 0, quantity_mode: str = QMODE_DEFAULT):
    body = Next7Req(store_id=store_id, item_id=item_id, start_date=start_date,
                    price=round_money(price), promo=promo, quantity_mode=quantity_mode)
    return predict_next_7days(body)
