# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import pandas as pd
import typer

from boost_sales.config import AppConfig
from boost_sales.pipeline.train import train_global, train_per_group
from boost_sales.data.io import load_sales_csv
from boost_sales.models.xgb import load_booster

# ---- unified core (single source of truth) ----
from boost_sales.api.core.horizons import parse_horizons_opt
from boost_sales.api.core.controls import (
    add_horizon_controls,      # builds price_h{h}, promo_h{h}, price_ratio_h{h}
    assume_no_change_fill,     # fills *_h{h} safely from current values when future plan unknown
)
from boost_sales.api.core.selectors import select_rows
from boost_sales.api.core.features import prepare_features as core_prepare_features

app = typer.Typer(help="Sales Forecast CLI", add_completion=False)

# ----------------------------
# Shared helpers
# ----------------------------
def _schema_kwargs(cfg: AppConfig) -> dict:
    """Expand cfg.columns into explicit keyword args for the CSV loader."""
    c = cfg.columns
    return dict(
        date_col=c.date,
        store_col=c.store,
        item_col=c.item,
        sales_col=c.sales,
        price_col=c.price,
        promo_col=c.promo,
    )

# ----------------------------
# Shared feature prep (parity with training)
# ----------------------------
def _prepare_features(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Use the single source of truth for feature engineering."""
    return core_prepare_features(df, cfg)

def _format_sales(yhat, cfg: AppConfig) -> pd.Series:
    """
    Format raw predictions according to unit settings:
      - unit_type == "integer" -> round(0) and cast to int64
      - unit_type == "float"   -> round(cfg.output.decimal_places) as float64
    """
    s = pd.Series(yhat, dtype="float64")
    unit_type = str(cfg.output.unit_type)
    if unit_type == "integer":
        return s.round(0).astype("int64")
    decimals = int(cfg.output.decimal_places)
    return s.round(decimals).astype("float64")

# ----------------------------
# generate-data  (lazy import to avoid hard dep)
# ----------------------------
@app.command("generate-data")
def cmd_generate_data(
    # mode
    mode: str = typer.Option(
        "synthetic",
        help="Generation mode: 'synthetic', 'from-flat', or 'from-tx'.",
    ),
    out_csv: Path = typer.Option(..., help="Where to write the generated CSV."),
    # schema overrides (optional)
    store_col: Optional[str] = typer.Option(None, help="Override store column name."),
    item_col: Optional[str] = typer.Option(None, help="Override item column name."),
    sales_col: Optional[str] = typer.Option(None, help="Override sales column name."),
    price_col: Optional[str] = typer.Option(None, help="Override price column name."),
    promo_col: Optional[str] = typer.Option(None, help="Override promo column name."),
    date_col: Optional[str] = typer.Option(None, help="Override date column name."),
    unit_price_col: Optional[str] = typer.Option(None, help="Override unit price column."),
    qty_col: Optional[str] = typer.Option(None, help="Override quantity column."),
    promo_col_name: Optional[str] = typer.Option(
        None, help="When promo_strategy='column', the source promo column name."
    ),
    # synthetic args
    n_stores: int = typer.Option(10, help="(synthetic) Number of stores."),
    n_items: int = typer.Option(50, help="(synthetic) Number of items."),
    start: str = typer.Option("2024-01-01", help="(synthetic) Start date YYYY-MM-DD."),
    periods: int = typer.Option(180, help="(synthetic) Number of daily periods."),
    seed: int = typer.Option(123, help="(synthetic) RNG seed."),
    # from-flat / from-tx args
    flat_csv: Optional[Path] = typer.Option(
        None, help="(from-flat) Path to a flat CSV to aggregate."
    ),
    tx_csv: Optional[Path] = typer.Option(
        None, help="(from-tx) Path to a transactions CSV to aggregate."
    ),
    sales_as: str = typer.Option(
        "sum",
        help="(from-flat/tx) Aggregate measure for 'sales' (e.g., 'sum' or 'mean').",
    ),
    price_strategy: str = typer.Option(
        "weighted_avg",
        help="(from-flat/tx) How to derive price (e.g., 'weighted_avg', 'mean').",
    ),
    price_round_to: int = typer.Option(
        2, help="(from-flat/tx) Round derived price to this many decimals."
    ),
    promo_strategy: str = typer.Option(
        "column",
        help="(from-flat/tx) How to derive promo (e.g., 'column', 'rolling').",
    ),
    promo_roll_window: int = typer.Option(
        28, help="(from-flat/tx) Rolling window for promo derivation."
    ),
    promo_drop_threshold: float = typer.Option(
        0.10, help="(from-flat/tx) Drop rows with promo share below this fraction."
    ),
    group_extra: List[str] = typer.Option(
        [],
        "--group-extra",
        help="(from-flat/tx) Extra grouping columns (repeatable).",
    ),
):
    """
    Generate a sales dataset and write it as CSV.
    """
    try:
        from boost_sales.data.generate import build_dataset, write_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "boost_sales.data.generate not available. "
            "If you don't need synthetic/aggregation utilities, skip this command."
        ) from e

    cfg = AppConfig()

    df = build_dataset(
        mode=mode,
        cfg=cfg,
        # schema overrides
        store_col=store_col,
        item_col=item_col,
        sales_col=sales_col,
        price_col=price_col,
        promo_col=promo_col,
        date_col=date_col,
        unit_price_col=unit_price_col,
        qty_col=qty_col,
        promo_col_name=promo_col_name,
        # synthetic args
        n_stores=n_stores,
        n_items=n_items,
        start=start,
        periods=periods,
        seed=seed,
        # from-flat / from-tx args
        flat_csv=flat_csv,
        tx_csv=tx_csv,
        sales_as=sales_as,
        price_strategy=price_strategy,
        price_round_to=price_round_to,
        promo_strategy=promo_strategy,
        promo_roll_window=promo_roll_window,
        promo_drop_threshold=promo_drop_threshold,
        group_extra=group_extra,
    )

    write_dataset(df, out_csv)
    typer.echo(f"✅ Wrote dataset: {out_csv}")

# ----------------------------
# train (global)
# ----------------------------
@app.command("train")
def cmd_train(
    data_csv: Path = typer.Option(..., help="Path to sales CSV."),
    models_dir: Path = typer.Option(..., help="Directory to write models."),
    hol_country: str = typer.Option("US", help="Holidays country code, e.g. 'US'."),
    hol_subdiv: Optional[str] = typer.Option(
        None, help="Holidays subdivision (e.g., 'CA' for California)."
    ),
    horizons: Optional[str] = typer.Option(
        None,
        help="List or range of horizons, e.g. '1-7' or '1,2,3,4'. "
             "If omitted, uses config default.",
    ),
    # ---- training speed/quality knobs ----
    nthread: Optional[int] = typer.Option(None, help="XGBoost nthread parameter (None = use many threads)."),
    verbose_eval: int = typer.Option(0, help="XGBoost training verbosity (0/1)."),
    enforce_single_thread_env: bool = typer.Option(
        False, help="Set single-thread env vars (OMP, MKL) for reproducibility."
    ),
    valid_cutoff_date: Optional[str] = typer.Option(
        None, help="YYYY-MM-DD. If provided, use as time-based validation split."
    ),
    valid_tail_days: Optional[int] = typer.Option(
        None, help="If set (e.g., 28), use last N days as validation when no cutoff date is given."
    ),
    early_stopping_rounds: Optional[int] = typer.Option(
        None, help="Enable early stopping if a validation split is present (e.g., 50)."
    ),
):
    """
    Train one global model per horizon and save in models_dir.
    """
    cfg = AppConfig()
    # Paths
    cfg.paths.data_csv = data_csv
    cfg.paths.models_dir = models_dir
    # Holidays
    cfg.train.hol_country = hol_country
    cfg.train.hol_subdiv = hol_subdiv
    # Training knobs
    cfg.train.nthread = nthread
    cfg.train.verbose_eval = verbose_eval
    cfg.train.enforce_single_thread_env = enforce_single_thread_env
    cfg.train.valid_cutoff_date = valid_cutoff_date
    cfg.train.valid_tail_days = valid_tail_days
    cfg.train.early_stopping_rounds = early_stopping_rounds
    # Optional override horizons
    cfg.train.horizons = parse_horizons_opt(horizons, cfg.train.horizons)

    train_global(cfg)
    typer.echo(f"✅ Trained horizons {cfg.train.horizons} → {models_dir}")

# ----------------------------
# train-per-group
# ----------------------------
@app.command("train-per-group")
def cmd_train_per_group(
    scope: str = typer.Argument(
        ..., help="Grouping scope: 'pair' (store+item), 'item', or 'store'."
    ),
    data_csv: Path = typer.Option(..., help="Path to sales CSV."),
    models_dir: Path = typer.Option(..., help="Root models directory."),
    hol_country: str = typer.Option("US", help="Holidays country code."),
    hol_subdiv: Optional[str] = typer.Option(None, help="Holidays subdivision."),
    horizons: Optional[str] = typer.Option(
        None,
        help="List or range of horizons, e.g. '1-7' or '1,2,3,4'. "
             "If omitted, uses config default.",
    ),
    # ---- training speed/quality knobs ----
    nthread: Optional[int] = typer.Option(None, help="XGBoost nthread parameter (None = use many threads)."),
    verbose_eval: int = typer.Option(0, help="XGBoost training verbosity (0/1)."),
    enforce_single_thread_env: bool = typer.Option(
        False, help="Set single-thread env vars (OMP, MKL) for reproducibility."
    ),
    valid_cutoff_date: Optional[str] = typer.Option(
        None, help="YYYY-MM-DD. If provided, use as time-based validation split."
    ),
    valid_tail_days: Optional[int] = typer.Option(
        None, help="If set (e.g., 28), use last N days as validation when no cutoff date is given."
    ),
    early_stopping_rounds: Optional[int] = typer.Option(
        None, help="Enable early stopping if a validation split is present (e.g., 50)."
    ),
):
    """
    Train models per group (by pair/item/store). Artifacts are nested under models_dir.
    """
    if scope not in {"pair", "item", "store"}:
        raise typer.BadParameter("scope must be one of: pair, item, store")

    cfg = AppConfig()
    cfg.paths.data_csv = data_csv
    cfg.paths.models_dir = models_dir
    cfg.train.hol_country = hol_country
    cfg.train.hol_subdiv = hol_subdiv
    cfg.train.nthread = nthread
    cfg.train.verbose_eval = verbose_eval
    cfg.train.enforce_single_thread_env = enforce_single_thread_env
    cfg.train.valid_cutoff_date = valid_cutoff_date
    cfg.train.valid_tail_days = valid_tail_days
    cfg.train.early_stopping_rounds = early_stopping_rounds
    cfg.train.horizons = parse_horizons_opt(horizons, cfg.train.horizons)

    train_per_group(cfg, scope)
    typer.echo(f"✅ Trained per-{scope} horizons {cfg.train.horizons} under {models_dir}")

# ----------------------------
# forecast (scope-aware)
# ----------------------------
@app.command("forecast")
def cmd_forecast(
    data_csv: Path = typer.Option(..., help="Path to sales CSV used for features."),
    models_dir: Path = typer.Option(..., help="Directory with trained models."),
    out_csv: Path = typer.Option(Path("preds.csv"), help="Where to write predictions."),
    # scope + params
    scope: str = typer.Option(
        "single",
        help=(
            "Scope: single | latest_per_pair | latest_per_store | latest_per_item | last_n_days | "
            "since_date | at_date"
        ),
    ),
    store_id: Optional[str] = typer.Option(
        None, help="Store id (required for single/latest_per_store; optional filter for others)."
    ),
    item_id: Optional[str] = typer.Option(
        None, help="Item id (required for single/latest_per_item; optional filter for others)."
    ),
    n_days: Optional[int] = typer.Option(None, help="For scope=last_n_days."),
    since_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD for scope=since_date."),
    at_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD for scope=at_date."),
    # horizons + features
    hol_country: str = typer.Option("US", help="Holidays country code to use."),
    hol_subdiv: Optional[str] = typer.Option(None, help="Holidays subdivision."),
    horizons: Optional[str] = typer.Option(None, help="Horizons (e.g. '1-7' or '1,2,4')."),
    assume_no_change: bool = typer.Option(
        True, help="Assume future price/promo equals latest observed."
    ),
    # output formatting overrides (optional)
    unit_type: Optional[str] = typer.Option(None, help="Format output as 'integer' or 'float' (default from config)."),
    decimal_places: Optional[int] = typer.Option(None, help="When float, number of decimals (default from config)."),
    # legacy compatibility
    latest_only: bool = typer.Option(
        False, help="[Deprecated] If true with scope=single, behaves like scope=latest_per_pair."
    ),
):
    """
    Load saved boosters and generate predictions for the requested horizons and scope.
    """
    cfg = AppConfig()
    cfg.paths.data_csv = data_csv
    cfg.paths.models_dir = models_dir
    cfg.train.hol_country = hol_country
    cfg.train.hol_subdiv = hol_subdiv

    # Optional output formatting overrides
    if unit_type:
        cfg.output.unit_type = unit_type  # type: ignore[assignment]
    if decimal_places is not None:
        cfg.output.decimal_places = int(decimal_places)

    hs = parse_horizons_opt(horizons, cfg.train.horizons)

    # resolve scope (legacy flag -> latest_per_pair)
    scope_resolved = "latest_per_pair" if latest_only and scope == "single" else scope

    # Load + prepare features (same as training)  **FIX: schema-aware loader**
    df = load_sales_csv(cfg.paths.data_csv, parse_dates=True, **_schema_kwargs(cfg))
    df = _prepare_features(df, cfg)
    c = cfg.columns

    # Parse dates
    sd = pd.to_datetime(since_date).date() if since_date else None
    ad = pd.to_datetime(at_date).date() if at_date else None

    # Select rows for the chosen scope
    sub = select_rows(
        df,
        scope_resolved,  # type: ignore
        c,
        store_id=store_id,
        item_id=item_id,
        n_days=n_days,
        since_date=sd,
        at_date=ad,
    )

    all_preds: list[pd.DataFrame] = []
    for h in hs:
        # Horizon controls
        dfh = add_horizon_controls(
            sub,
            horizon=h,
            cfg=cfg,
            date_col=c.date,
            price_col=c.price,
            promo_col=c.promo,
        )
        if assume_no_change:
            dfh = assume_no_change_fill(dfh, horizon=h, cfg=cfg)

        # Load model + features
        booster, feats = load_booster(models_dir, h)

        # Filter rows that have all required features
        if not set(feats).issubset(dfh.columns):
            continue
        ok = dfh[feats].notna().all(axis=1)
        ready = dfh.loc[ok].copy()
        if ready.empty:
            continue

        import xgboost as xgb  # local import
        dmat = xgb.DMatrix(ready[feats], feature_names=feats)
        yhat = booster.predict(dmat)

        # Format predictions per unit settings and emit as `sales`
        sales_series = _format_sales(yhat, cfg)

        out = pd.DataFrame(
            {
                "store_id": ready[c.store].astype("string"),
                "item_id": ready[c.item].astype("string"),
                "base_date": pd.to_datetime(ready[c.date]).dt.date,
                "target_date": (
                    pd.to_datetime(ready[c.date]) + pd.to_timedelta(h, unit="D")
                ).dt.date,
                "horizon": h,
                "sales": sales_series.values,
            }
        )
        all_preds.append(out)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not all_preds:
        pd.DataFrame(
            columns=["store_id", "item_id", "base_date", "target_date", "horizon", "sales"]
        ).to_csv(out_csv, index=False)
        typer.echo("⚠️ No eligible rows to predict. Empty file written.")
        return

    preds = (
        pd.concat(all_preds, ignore_index=True)
        .sort_values(["store_id", "item_id", "target_date", "horizon"])
    )
    preds.to_csv(out_csv, index=False)
    typer.echo(f"✅ Wrote predictions ({len(preds):,} rows) → {out_csv}")

# ----------------------------
# serve-web (API + HTML)
# ----------------------------
@app.command("serve-web")
def cmd_serve_web(
    models_dir: Path = typer.Option(..., help="Directory with trained models."),
    data_csv: Optional[Path] = typer.Option(
        None, help="(Optional) Path to sales CSV for feature parity / demo."
    ),
    hol_country: str = typer.Option("US", help="Holidays country code to use."),
    hol_subdiv: Optional[str] = typer.Option(None, help="Holidays subdivision."),
    horizons: Optional[str] = typer.Option(
        None,
        help="List or range of horizons the service should expose (e.g. '1-7'). "
             "If omitted, uses config default.",
    ),
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(8000, help="Port to bind."),
    reload: bool = typer.Option(False, help="Reload on code changes (dev only)."),
):
    """
    Start the FastAPI web service (Swagger + web UI).

    We start uvicorn with an import string ('boost_sales.api.server:app') in both
    reload and non-reload modes. The server builds the ASGI app from environment vars.
    """
    # Resolve horizons now for env export
    cfg = AppConfig()
    hs = parse_horizons_opt(horizons, cfg.train.horizons)

    # Optional demo CSV default (if not provided)
    if data_csv is None:
        demo = Path("data/sales.csv")
        if demo.exists():
            data_csv = demo

    try:
        import uvicorn  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Web dependencies missing. Install with:  pip install '.[webui]'"
        ) from e

    # Export config to env for the ASGI app builder in boost_sales.api.server
    os.environ["SF_MODELS_DIR"] = str(models_dir)
    if data_csv is not None:
        os.environ["SF_DATA_CSV"] = str(data_csv)
    else:
        os.environ.pop("SF_DATA_CSV", None)
    os.environ["SF_HOL_COUNTRY"] = hol_country
    if hol_subdiv:
        os.environ["SF_HOL_SUBDIV"] = hol_subdiv
    else:
        os.environ.pop("SF_HOL_SUBDIV", None)
    os.environ["SF_HORIZONS"] = ",".join(map(str, hs))

    import uvicorn
    uvicorn.run(
        "boost_sales.api.server:app",
        host=host,
        port=port,
        reload=reload,
        factory=False,
    )

if __name__ == "__main__":
    app()
