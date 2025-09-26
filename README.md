# XGBoost Sales Forecast Web App and API

Production-ready time-series forecasting for retail-style data. **Who it’s for:** ML/data teams and planners who want a transparent, configurable baseline they can run locally or deploy as a service.

FastAPI backend + lightweight web UI + configurable XGBoost training (global or per‑group), with holiday features and **what‑if** price/promo simulations.

---

## Table of Contents

- [Architecture](#architecture)
- [Highlights](#highlights)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
  - [Defaults](#defaults)
- [Training](#training)
- [Forecasting](#forecasting)
  - [Scopes](#scopes)
  - [What‑if Plans](#what-if-plans)
  - [Formatting](#formatting)
- [REST API](#rest-api)
  - [/docs and /redoc](#docs-and-redoc)
  - [/forecast (JSON)](#forecast-json)
  - [/forecast/csv (multipart)](#forecastcsv-multipart)
  - [/train](#train)
  - [/train/suggest_es](#trainsuggest_es)
  - [Error responses](#error-responses)
- [Data Schema](#data-schema)
  - [Assumptions](#assumptions)
- [Performance Tips](#performance-tips)
- [Deploy](#deploy)
  - [Uvicorn/Gunicorn](#uvicorngunicorn)
  - [Docker](#docker)
  - [Docker Compose](#docker-compose)
  - [Security note](#security-note)
- [Developing](#developing)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [License](#license)

---

## Architecture

```
boost_sales/
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

> _Tip: replace this section’s image with a real screenshot of your UI when available._

---

## Highlights

- **Configurable**: global or per‑group models; lags/rolling windows; holiday effects; future price/promo controls.
- **Fast**: XGBoost `hist` (CPU) & `gpu_hist` (GPU) support; global modeling for scale.
- **Reproducible**: typed configs (Pydantic), versionable settings, deterministic seeds.
- **Usable**: simple web UI for explore/train/forecast, and a typed REST API.
- **Practical validation**: time‑based splits with validation windows and **auto‑suggested** `early_stopping_rounds` (derived from `n_estimators` and window size).
- **Smart defaults**: holiday region, validation windows, and early stopping suggestions.

---

## Quickstart

### Requirements
- Python **3.10+**
- pip, venv (recommended)
- (Optional) CUDA‑enabled XGBoost for GPU

### Install

```bash
git clone https://github.com/dmsfiris/boost-sales.git
cd boost-sales

# Virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .   # or: pip install -r requirements.txt
```

### Run the app (Dev)

```bash
# Environment
export SF_MODELS_DIR=./models
export SF_DATA_CSV=./data/sales.csv          # optional
export SF_HOL_COUNTRY=US                     # default: US
# export SF_HOL_SUBDIV=CA                    # optional (e.g., US-CA)

# Start
uvicorn boost_sales.api.server:app --reload --port 8000
```

Open the UI: `http://localhost:8000`  
Interactive API docs: `http://localhost:8000/docs` (Swagger UI), `http://localhost:8000/redoc`

### First forecast in one minute

Use the built‑in demo CSV and the default configuration:

```bash
# Train (from the Training page), then request a simple forecast via API:
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
        "scope":"single",
        "store_id":"S01",
        "item_id":"I01",
        "horizons":"1-7",
        "use_server_csv":true,
        "unit_type":"integer",
        "decimal_places":0
      }'
```

---

## Configuration

Main config lives in **`boost_sales/config.py`** (Pydantic models).

- **Paths**: CSV path & models directory.
- **Columns**: rename date/store/item/sales/price/promo if your dataset differs.
- **Calendar features**: year/quarter/month/day/dow/weekend.
- **Lag & rolling**: lags (1,7,14,28) and rolling stats (7,28).
- **Future controls**: price/promo futures and price ratio.
- **Output formatting**: integer vs float and decimal places.
- **Training knobs**: `n_estimators`, `max_depth`, `learning_rate`, `tree_method`, regularization, etc.
- **Validation**: `valid_tail_days` or `valid_cutoff_date` (time‑based split).
- **Early Stopping**: `early_stopping_rounds`; the UI can **auto‑suggest** a value based on your window and estimator cap.

### Defaults

| Setting                | Default         | Notes                               |
|-----------------------|-----------------|-------------------------------------|
| `SF_HOL_COUNTRY`      | `US`            | Country‑level holidays enabled       |
| `SF_HOL_SUBDIV`       | _(none)_        | Add e.g. `US-CA` for state holidays |
| `SF_MODELS_DIR`       | _(required)_    | Where models are saved/loaded        |
| `SF_DATA_CSV`         | _(optional)_    | Server‑side CSV path                 |
| Horizons              | `1-7`           | 7 daily horizons                     |
| Unit type             | `integer`       | Rounding applied                     |
| Validation window     | `~10–20%` tail  | Choose recent tail or cutoff date    |

Environment overrides are read on startup.

---

## Training

From the **Training** page you can:

- Choose **mode**: `global` (fastest) or `per_group` (by pair/item/store).
- Wipe outputs (optional).
- Set **horizons**, **holiday region**, and **XGBoost** params.
- Choose validation via cutoff date or **last N days**.
- Use **Auto‑suggest** to derive `early_stopping_rounds` from `n_estimators` & your validation window.

> **Tip:** Prefer a reasonably large `n_estimators` with early stopping. Let training stop when validation RMSE plateaus rather than guessing a small cap.

---

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

---

## REST API

All endpoints live under the same app as the UI.

### /docs and /redoc

- Swagger UI: `GET /docs`  
- ReDoc: `GET /redoc`

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

**Optional body field**:  
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

### Error responses

```json
{
  "detail": [
    {"loc":["body","store_id"],"msg":"Field required","type":"value_error.missing"}
  ]
}
```

---

## Data Schema

CSV must include these columns (rename via `config.py` if needed):

- `date` (YYYY‑MM‑DD or ISO date)
- `store_id` (string)
- `item_id` (string)
- `sales` (float/int)
- `price` (float)
- `promo` (0/1/0.5)

### Assumptions

- **Cadence**: daily rows per `(store_id, item_id)`; non‑daily data should be resampled to daily.
- **Timezone**: dates treated as naive local dates; convert to consistent local or UTC before ingest.
- **Missing data**: rows with missing `sales` are excluded from training; `price/promo` missing values are imputed with hold‑forward where appropriate.
- **Outliers**: spikes/dips are not automatically clipped; handle in your preprocessing if needed.

---

## Performance Tips

- Start with **global** mode; switch to per‑group for tricky segments only.
- Prefer `tree_method="hist"` on CPU. Use `gpu_hist` if available (e.g., `max_bin=256`).
- Set a **large** `n_estimators` with **early stopping**.
- Validation: use a **recent window** (e.g., last 10–20% of dates).  
- Holidays: enable the appropriate country; optionally add subdivision (e.g., `US-CA`) for state holidays.
- Reproducibility: set `random_state`; enforce single thread (`nthread=1`) only if you need bit‑for‑bit parity.

---

## Deploy

Any ASGI host works. Minimal examples:

### Uvicorn/Gunicorn

```bash
export SF_MODELS_DIR=/opt/models
export SF_DATA_CSV=/opt/data/sales.csv
uvicorn boost_sales.api.server:app --host 0.0.0.0 --port 8000
# or: gunicorn -k uvicorn.workers.UvicornWorker boost_sales.api.server:app
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -U pip && pip install -e .
ENV SF_MODELS_DIR=/models SF_DATA_CSV=/data/sales.csv
EXPOSE 8000
CMD ["uvicorn", "boost_sales.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
services:
  sales-forecast:
    build: .
    ports: ["8000:8000"]
    environment:
      SF_MODELS_DIR: /models
      SF_DATA_CSV: /data/sales.csv
      SF_HOL_COUNTRY: US
    volumes:
      - ./models:/models
      - ./data:/data
```

### Security note

Run behind a reverse proxy / gateway with authentication if exposed outside a trusted network. Configure CORS appropriately for the UI domain.

---

## Developing

- Code style: black/ruff (optional).
- Tests: add under `tests/` (PyTest).
- Static assets: edit `static/main.js` and `static/main.css`.
- Templates: `templates/forecast.html` & `templates/training.html`.

---

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

---

## License

**MIT** — see `LICENSE` for details. Contributions welcome!
