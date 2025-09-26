# XGBoost Sales Forecast Web App and API

Production‑ready, end‑to‑end time‑series forecasting for retail‑style data.  
This project ships a FastAPI backend, a lightweight web UI, and a configurable XGBoost training pipeline that supports global and per‑group models, holiday features, and “what‑if” (price/promo) simulations.

<p align="center">
  <img alt="Sales Forecast UI" src="https://user-images.githubusercontent.com/placeholder/sales-forecast-ui.png" width="720">
</p>

> **Highlights**
>
> - 🧮 **Accurate**: strong lag/rolling features, holiday effects, future price/promo controls.
> - ⚡ **Fast**: XGBoost `hist` (CPU) & `gpu_hist` (GPU) support, global modeling for scale.
> - 🧰 **Practical**: train globally or per group; early‑stopping with validation windows.
> - 🌐 **Simple UI**: forecast scopes, paging, formatting, and plan simulations.
> - 🧾 **Typed API**: JSON & CSV endpoints; helpful errors; pagination; reproducible configs.
> - 🪄 **Smart defaults**: holiday region, validation window, and ES rounds auto‑suggest.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Quickstart (Dev)](#quickstart-dev)
- [Configuration](#configuration)
- [Training](#training)
- [Forecasting](#forecasting)
  - [Scopes](#scopes)
  - [What‑if Plans](#what-if-plans)
  - [Formatting](#formatting)
- [REST API](#rest-api)
  - [/forecast (JSON)](#forecast-json)
  - [/forecast/csv (multipart)](#forecastcsv-multipart)
  - [/train](#train)
  - [/train/suggest_es](#trainsuggest_es)
- [Data Schema](#data-schema)
- [Performance Tips](#performance-tips)
- [Deploy](#deploy)
- [Developing](#developing)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [License](#license)

---

## Architecture

```
sales_forecast/
├─ api/
│  ├─ server.py              # FastAPI app factory & routes (UI + REST)
│  ├─ schemas.py             # Pydantic models
│  ├─ service.py             # Forecast/train service glue
│  └─ core/
│     └─ horizons.py         # Horizon parsing helpers
├─ pipeline/
│  ├─ train.py               # train_global / train_per_group
│  └─ ...                    # feature engineering, model IO
├─ templates/                # Jinja2 HTML (forecast.html, training.html, base.html)
├─ static/
│  ├─ main.css
│  └─ main.js                # UI logic + friendly error handling
├─ config.py                 # AppConfig & training knobs
└─ data/
   └─ sales.csv              # (optional) demo CSV
```

- **FastAPI** serves both the UI and REST API.
- **Jinja2** templates for the Forecast and Training pages.
- **XGBoost** models per horizon (global or per‑group).
- **Holiday** features via country/subdivision.
- **Future controls**: price/promo plans for scenario testing.

## Features

- **Forecast UI** with multiple scopes: single target, latest per pair/store/item, last N days, since date, or exact date.
- **Training UI** with presets (Balanced/Fast/Quality/GPU), advanced knobs, validation windows, and **Auto‑suggest** for `early_stopping_rounds`.
- **CSV Upload** (optional) for both training and forecasting.
- **Models directory override** (UI + API) to switch model sets.
- **Friendly error messages** (Pydantic/FastAPI parsing, clear hints).

## Getting Started

### Requirements

- Python **3.10+**
- pip, venv (recommended)
- (Optional) CUDA‑enabled XGBoost for GPU

### Install

```bash
git clone https://github.com/your-org/sales-forecast-web.git
cd sales-forecast-web

# Option A: with a virtualenv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .          # or: pip install -r requirements.txt

# Option B: poetry (if you prefer)
# poetry install
```

### Quickstart (Dev)

```bash
# Environment
export SF_MODELS_DIR=./models
export SF_DATA_CSV=./data/sales.csv          # optional
export SF_HOL_COUNTRY=US                     # default: US
# export SF_HOL_SUBDIV=CA                    # optional (e.g., US-CA)

# Run the app (auto-reload during dev)
uvicorn sales_forecast.api.server:app --reload --port 8000
```

Open http://localhost:8000 to use the UI.  
Use **Training** to fit models, then **Forecast** to generate predictions.

## Configuration

Main config lives in **`sales_forecast/config.py`** (Pydantic models):

- **Paths**: CSV path & models directory.
- **Columns**: rename date/store/item/sales/price/promo if your dataset differs.
- **Calendar features**: year/quarter/month/day/dow/weekend (cheap & helpful).
- **Lag & rolling**: lags (1,7,14,28), rolling stats (7,28), price roll context.
- **Future controls**: price/promo futures and price ratio.
- **Output formatting**: integer vs float and decimal places.
- **Training knobs**: `n_estimators`, `max_depth`, `learning_rate`, `tree_method`, regularization, etc.
- **Validation**: `valid_tail_days` (e.g., 28) or `valid_cutoff_date` (exclusive split).
- **Early Stopping**: `early_stopping_rounds` (UI can auto‑suggest based on window size & estimators).

Environment overrides used by the app factory:
- `SF_MODELS_DIR` *(required)*
- `SF_DATA_CSV` *(optional)*
- `SF_HOL_COUNTRY` *(default `US`)*
- `SF_HOL_SUBDIV` *(optional)*
- `SF_HORIZONS` *(e.g., `1-7`)*

## Training

From the **Training** page you can:

- Choose **mode**: `global` (fastest) or `per_group` (by pair / item / store).
- Optional **wipe** of outputs.
- Set **horizons**, **holiday region**, **XGBoost** params.
- Configure **validation** via cutoff date or **last N days**.
- Click **Auto‑suggest** to derive a good `early_stopping_rounds` from `n_estimators` & your validation window.

> **Tip:** Prefer a reasonably large `n_estimators` with early stopping enabled. Let training stop when validation RMSE plateaus rather than guessing a small cap.

## Forecasting

Use the **Forecast** page to generate predictions and simulate plans.

- **Data source**: server CSV (default) or upload your own.
- **Models directory**: pick which trained models to load.
- **Paging**: control page size and navigation.
- **Units & decimals**: render as integers or floats.

### Scopes

- `single`: one `(store_id, item_id)` at its latest date.
- `latest_per_pair`: latest date for each pair across the dataset.
- `latest_per_store`: latest date for each item within a store.
- `latest_per_item`: latest date across stores for a given item.
- `last_n_days`: all rows within the last N days.
- `since_date`: all rows on/after a date.
- `at_date`: rows exactly at a date.

### What‑if Plans

- **Price**: set a scalar (e.g., `0.9` or `90%`) or a CSV (`12.5,12.7,12.9`).
- **Promo**: set a scalar (`0`, `0.5`, `1`, `50%`) or CSV per horizon.
- If both left blank, the model assumes **no change** (safe default).

### Formatting

- `unit_type`: `integer` or `float`.
- `decimal_places`: only applies to `float` output.

## REST API

All endpoints live under the same app as the UI.

### `/forecast` (JSON)

**Request** (`application/json`)

```jsonc
{
  "scope": "single",
  "store_id": "S01",
  "item_id": "I01",
  "horizons": "1-7",
  "use_server_csv": true,
  "price_future": "0.9",
  "promo_future": "0,0,0,0.5,0,0,0",
  "unit_type": "integer",
  "decimal_places": 0,
  "page": 1,
  "page_size": 100
}
```

**Optional body field** (top‑level):  
- `models_dir` *(string)* — override the models directory used for loading models.

**Response** (`application/json`)

```jsonc
{
  "predictions": [
    {
      "store_id": "S01",
      "item_id": "I01",
      "base_date": "2024-05-31",
      "target_date": "2024-06-01",
      "horizon": 1,
      "sales": 42
    }
  ],
  "page": { "page": 1, "page_size": 100, "total": 7 }
}
```

### `/forecast/csv` (multipart)

For uploaded CSVs (or to avoid JSON).

**Fields**
- `file` *(optional; required if `use_demo_csv=false`)*
- `scope`, `store_id`, `item_id`, `horizons`, `use_demo_csv`
- `price_future`, `promo_future`, `unit_type`, `decimal_places`
- `page`, `page_size`
- `models_dir` *(optional)*

**Response**: same schema as JSON variant.

### `/train`

**Fields** (multipart form)
- **Mode/Scope**: `mode` (`global`|`per_group`), `train_scope` (`pair`|`item`|`store`)
- **Data**: `use_demo_csv` (bool), `file` (csv)
- **Paths/Horizons/Holidays**: `models_dir`, `horizons`, `hol_country`, `hol_subdiv`, `wipe`
- **XGBoost**: `nthread`, `n_estimators`, `max_depth`, `learning_rate`, `tree_method`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, `max_bin`, `random_state`, `required_feature_notna`
- **Validation**: `valid_cutoff_date`, `valid_tail_days`, `early_stopping_rounds`, `verbose_eval`, `enforce_single_thread_env`

**Response**

```jsonc
{
  "ok": true,
  "trained": { "mode": "global", "groups": 1 },
  "models_dir": "models",
  "horizons": [1,2,3,4,5,6,7],
  "holidays": { "country": "US", "subdiv": null },
  "seconds": 12.34,
  "note": "Training finished."
}
```

### `/train/suggest_es`

Heuristic to recommend `early_stopping_rounds` given `n_estimators` and your validation window size.

**Fields**
- `n_estimators` *(int, required)*
- `valid_tail_days` *(int, optional)*
- `valid_cutoff_date` *(str, optional; assumes 28 days if provided without tail)*
- `cap_min` *(int, default 20)*
- `cap_max` *(int, default 120)*

**Response**

```jsonc
{
  "ok": true,
  "suggestion": 36,
  "percent": 0.12,
  "n_estimators": 300,
  "used_tail_days": 28,
  "assumed_default": false,
  "cap": { "min": 20, "max": 120 },
  "note": "Based on n_estimators and validation window ≈ 28 days."
}
```

## Data Schema

CSV must include these columns (rename via `config.py` if needed):

- `date` (YYYY‑MM‑DD or ISO date)
- `store_id` (string)
- `item_id` (string)
- `sales` (float/int)
- `price` (float)
- `promo` (0/1/0.5)

## Performance Tips

- Use **global** mode first; switch to per‑group for tricky segments only.
- Prefer `tree_method="hist"` on CPU. Use `gpu_hist` if available.
- Set a **large** `n_estimators` cap with **early stopping**.
- Validation: use a **recent window** (e.g., last 10–20% of dates).  
- Holidays: **enable** the appropriate country; optionally add subdivision (`US-CA`) for state holidays.
- Reproducibility: set `random_state`; enable “enforce single thread” + `nthread=1` only if you need bit‑for‑bit parity.

## Deploy

Any ASGI host works. Minimal examples:

### Uvicorn/Gunicorn

```bash
export SF_MODELS_DIR=/opt/models
export SF_DATA_CSV=/opt/data/sales.csv
uvicorn sales_forecast.api.server:app --host 0.0.0.0 --port 8000
# or: gunicorn -k uvicorn.workers.UvicornWorker sales_forecast.api.server:app
```

### Docker (sketch)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -U pip && pip install -e .
ENV SF_MODELS_DIR=/models SF_DATA_CSV=/data/sales.csv
EXPOSE 8000
CMD ["uvicorn", "sales_forecast.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Developing

- Code style: black/ruff (optional).
- Tests: add under `tests/` (PyTest).
- Static assets: edit `static/main.js` and `static/main.css`.
- Templates: `templates/forecast.html` & `templates/training.html`.

## Troubleshooting & FAQ

**Q: Why do I get “Field required” on /forecast?**  
A: Ensure you provided the required params for your **scope**. For `single`, both `store_id` and `item_id` are required. The UI performs quick validation; the API enforces it too.

**Q: Should I upload a CSV on the Forecast page?**  
A: Optional. If your server already has a configured CSV (demo or production), keep “Use server CSV” checked. Upload when you want to forecast against a different dataset ad‑hoc.

**Q: How do holidays help?**  
A: Holiday features often improve accuracy by capturing demand shifts (spikes and dips). Choose the matching country; optionally add a subdivision (e.g., `US-CA`) for state/province holidays. Leave subdivision blank to include **national** holidays only.

**Q: Why 36 for `early_stopping_rounds`?**  
A: With `valid_tail_days=28` we suggest ~12% of `n_estimators` (clamped). For `n_estimators=300`, that’s 36—balanced patience without over‑waiting.

**Q: Global vs per‑group training?**  
A: Global is faster and generalizes across entities; per‑group can capture idiosyncrasies but is slower and can overfit small groups.

## License

**MIT** — see `LICENSE` for details. Contributions welcome!
