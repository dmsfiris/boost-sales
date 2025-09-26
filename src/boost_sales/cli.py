# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, List, Optional

import pandas as pd
import typer

from boost_sales.api.core.controls import (
    add_horizon_controls,  # builds price_h{h}, promo_h{h}, price_ratio_h{h}
    assume_no_change_fill,  # fills *_h{h} safely from current values when future plan unknown
)
from boost_sales.api.core.features import prepare_features as core_prepare_features
from boost_sales.api.core.horizons import parse_horizons_opt  # single source of truth
from boost_sales.api.core.selectors import select_rows
from boost_sales.config import AppConfig
from boost_sales.data.io import load_sales_csv
from boost_sales.models.xgb import load_booster
from boost_sales.pipeline.train import train_global, train_per_group

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
    mode: Annotated[str, typer.Option(help="Generation mode: 'synthetic', 'from-flat', or 'from-tx'.")] = "synthetic",
    out_csv: Annotated[Path, typer.Option(help="Where to write the generated CSV.")] = ...,
    # schema overrides (optional)
    store_col: Annotated[Optional[str], typer.Option(help="Override store column name.")] = None,
    item_col: Annotated[Optional[str], typer.Option(help="Override item column name.")] = None,
    sales_col: Annotated[Optional[str], typer.Option(help="Override sales column name.")] = None,
    price_col: Annotated[Optional[str], typer.Option(help="Override price column name.")] = None,
    promo_col: Annotated[Optional[str], typer.Option(help="Override promo column name.")] = None,
    date_col: Annotated[Optional[str], typer.Option(help="Override date column name.")] = None,
    unit_price_col: Annotated[Optional[str], typer.Option(help="Override unit price column.")] = None,
    qty_col: Annotated[Optional[str], typer.Option(help="Override quantity column.")] = None,
    promo_col_name: Annotated[
        Optional[str], typer.Option(help="When promo_strategy='column', source promo column.")
    ] = None,
    # synthetic args
    n_stores: Annotated[int, typer.Option(help="(synthetic) Number of stores.")] = 10,
    n_items: Annotated[int, typer.Option(help="(synthetic) Number of items.")] = 50,
    start: Annotated[str, typer.Option(help="(synthetic) Start date YYYY-MM-DD.")] = "2024-01-01",
    periods: Annotated[int, typer.Option(help="(synthetic) Number of daily periods.")] = 180,
    seed: Annotated[int, typer.Option(help="(synthetic) RNG seed.")] = 123,
    # from-flat / from-tx args
    flat_csv: Annotated[Optional[Path], typer.Option(help="(from-flat) Path to a flat CSV to aggregate.")] = None,
    tx_csv: Annotated[Optional[Path], typer.Option(help="(from-tx) Path to a transactions CSV to aggregate.")] = None,
    sales_as: Annotated[
        str, typer.Option(help="(from-flat/tx) Aggregate for 'sales' (e.g., 'sum' or 'mean').")
    ] = "sum",
    price_strategy: Annotated[
        str, typer.Option(help="(from-flat/tx) Price derivation (e.g., 'weighted_avg', 'mean').")
    ] = "weighted_avg",
    price_round_to: Annotated[int, typer.Option(help="(from-flat/tx) Round derived price to this many decimals.")] = 2,
    promo_strategy: Annotated[
        str, typer.Option(help="(from-flat/tx) Promo derivation (e.g., 'column', 'rolling').")
    ] = "column",
    promo_roll_window: Annotated[int, typer.Option(help="(from-flat/tx) Rolling window for promo derivation.")] = 28,
    promo_drop_threshold: Annotated[
        float, typer.Option(help="(from-flat/tx) Drop rows with promo share below this fraction.")
    ] = 0.10,
    group_extra: Annotated[
        Optional[List[str]],
        typer.Option("--group-extra", help="(from-flat/tx) Extra grouping columns (repeatable)."),
    ] = None,
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
        group_extra=group_extra or [],
    )

    write_dataset(df, out_csv)
    typer.echo(f"✅ Wrote dataset: {out_csv}")


# ----------------------------
# train (global)
# ----------------------------
@app.command("train")
def cmd_train(
    data_csv: Annotated[Path, typer.Option(help="Path to sales CSV.")] = ...,
    models_dir: Annotated[Path, typer.Option(help="Directory to write models.")] = ...,
    hol_country: Annotated[str, typer.Option(help="Holidays country code, e.g. 'US'.")] = "US",
    hol_subdiv: Annotated[Optional[str], typer.Option(help="Holidays subdivision (e.g., 'CA' for California).")] = None,
    horizons: Annotated[
        Optional[str],
        typer.Option(help="Horizons like '1-7' or '1,2,3,4'. If omitted, uses config default."),
    ] = None,
    # ---- training speed/quality knobs ----
    nthread: Annotated[Optional[int], typer.Option(help="XGBoost nthread (None = use many threads).")] = None,
    verbose_eval: Annotated[int, typer.Option(help="XGBoost training verbosity (0/1).")] = 0,
    enforce_single_thread_env: Annotated[
        bool, typer.Option(help="Set single-thread env vars (OMP, MKL) for reproducibility.")
    ] = False,
    valid_cutoff_date: Annotated[Optional[str], typer.Option(help="YYYY-MM-DD. Time-based validation split.")] = None,
    valid_tail_days: Annotated[
        Optional[int], typer.Option(help="If set (e.g., 28), use last N days as validation.")
    ] = None,
    early_stopping_rounds: Annotated[
        Optional[int], typer.Option(help="Enable early stopping if a validation split is present.")
    ] = None,
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
    scope: Annotated[str, typer.Argument(help="Grouping scope: 'pair' (store+item), 'item', or 'store'.")] = ...,
    data_csv: Annotated[Path, typer.Option(help="Path to sales CSV.")] = ...,
    models_dir: Annotated[Path, typer.Option(help="Root models directory.")] = ...,
    hol_country: Annotated[str, typer.Option(help="Holidays country code.")] = "US",
    hol_subdiv: Annotated[Optional[str], typer.Option(help="Holidays subdivision.")] = None,
    horizons: Annotated[
        Optional[str],
        typer.Option(help="Horizons like '1-7' or '1,2,3,4'. If omitted, uses config default."),
    ] = None,
    # ---- training speed/quality knobs ----
    nthread: Annotated[Optional[int], typer.Option(help="XGBoost nthread (None = use many threads).")] = None,
    verbose_eval: Annotated[int, typer.Option(help="XGBoost training verbosity (0/1).")] = 0,
    enforce_single_thread_env: Annotated[
        bool, typer.Option(help="Set single-thread env vars (OMP, MKL) for reproducibility.")
    ] = False,
    valid_cutoff_date: Annotated[Optional[str], typer.Option(help="YYYY-MM-DD. Time-based validation split.")] = None,
    valid_tail_days: Annotated[
        Optional[int], typer.Option(help="If set (e.g., 28), use last N days as validation.")
    ] = None,
    early_stopping_rounds: Annotated[
        Optional[int], typer.Option(help="Enable early stopping if a validation split is present.")
    ] = None,
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
    data_csv: Annotated[Path, typer.Option(help="Path to sales CSV used for features.")] = ...,
    models_dir: Annotated[Path, typer.Option(help="Directory with trained models.")] = ...,
    out_csv: Annotated[Optional[Path], typer.Option(help="Where to write predictions.")] = None,
    # scope + params
    scope: Annotated[
        str,
        typer.Option(
            help=(
                "Scope: single | latest_per_pair | latest_per_store | latest_per_item | "
                "last_n_days | since_date | at_date"
            )
        ),
    ] = "single",
    store_id: Annotated[Optional[str], typer.Option(help="Store id (required for single/latest_per_store).")] = None,
    item_id: Annotated[Optional[str], typer.Option(help="Item id (required for single/latest_per_item).")] = None,
    n_days: Annotated[Optional[int], typer.Option(help="For scope=last_n_days.")] = None,
    since_date: Annotated[Optional[str], typer.Option(help="YYYY-MM-DD for scope=since_date.")] = None,
    at_date: Annotated[Optional[str], typer.Option(help="YYYY-MM-DD for scope=at_date.")] = None,
    # horizons + features
    hol_country: Annotated[str, typer.Option(help="Holidays country code to use.")] = "US",
    hol_subdiv: Annotated[Optional[str], typer.Option(help="Holidays subdivision.")] = None,
    horizons: Annotated[Optional[str], typer.Option(help="Horizons (e.g. '1-7' or '1,2,4').")] = None,
    assume_no_change: Annotated[bool, typer.Option(help="Assume future price/promo equals latest observed.")] = True,
    # output formatting overrides (optional)
    unit_type: Annotated[
        Optional[str], typer.Option(help="Format output as 'integer' or 'float' (default from config).")
    ] = None,
    decimal_places: Annotated[
        Optional[int], typer.Option(help="When float, number of decimals (default from config).")
    ] = None,
    # legacy compatibility
    latest_only: Annotated[
        bool,
        typer.Option(help="[Deprecated] With scope=single, behaves like scope=latest_per_pair."),
    ] = False,
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

    # Load + prepare features (same as training)
    df = load_sales_csv(cfg.paths.data_csv, parse_dates=True, **_schema_kwargs(cfg))
    df = _prepare_features(df, cfg)
    c = cfg.columns

    # Parse dates
    sd = pd.to_datetime(since_date).date() if since_date else None
    ad = pd.to_datetime(at_date).date() if at_date else None

    # Select rows for the chosen scope
    sub = select_rows(
        df,
        scope_resolved,  # type: ignore[arg-type]
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
                "target_date": (pd.to_datetime(ready[c.date]) + pd.to_timedelta(h, unit="D")).dt.date,
                "horizon": h,
                "sales": sales_series.values,
            }
        )
        all_preds.append(out)

    # Output path
    out_csv = out_csv or Path("preds.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not all_preds:
        pd.DataFrame(columns=["store_id", "item_id", "base_date", "target_date", "horizon", "sales"]).to_csv(
            out_csv, index=False
        )
        typer.echo("⚠️ No eligible rows to predict. Empty file written.")
        return

    preds = pd.concat(all_preds, ignore_index=True).sort_values(["store_id", "item_id", "target_date", "horizon"])
    preds.to_csv(out_csv, index=False)
    typer.echo(f"✅ Wrote predictions ({len(preds):,} rows) → {out_csv}")


# ----------------------------
# serve-web (API + HTML)
# ----------------------------
@app.command("serve-web")
def cmd_serve_web(
    host: Annotated[str, typer.Option(help="Host to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload on code changes (dev only).")] = False,
):
    """
    Start the FastAPI web service (Swagger + web UI).

    Configuration is resolved from environment variables (with safe defaults):
      - SF_MODELS_DIR   (default: ./models, auto-created)
      - SF_DATA_CSV     (optional; auto-detected as ./data/sales.csv if it exists)
      - SF_HOL_COUNTRY  (default: US)
      - SF_HOL_SUBDIV   (optional)
      - SF_HORIZONS     (default: from AppConfig if unset)
    """
    try:
        import uvicorn  # noqa: F401
    except Exception as e:
        raise RuntimeError("Web dependencies missing. Install with:  pip install '.[webui]'") from e

    cfg = AppConfig()

    # Resolve MODELS_DIR: env -> ./models
    models_dir = Path(os.getenv("SF_MODELS_DIR") or "./models")
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SF_MODELS_DIR"] = str(models_dir)

    # Resolve DATA_CSV: keep existing env if set; else auto-detect ./data/sales.csv; else unset
    data_env = os.getenv("SF_DATA_CSV")
    if data_env:
        os.environ["SF_DATA_CSV"] = data_env  # keep existing
    else:
        demo_csv = Path("data/sales.csv")
        if demo_csv.exists():
            os.environ["SF_DATA_CSV"] = str(demo_csv)
        else:
            os.environ.pop("SF_DATA_CSV", None)

    # Holidays: env -> default
    os.environ["SF_HOL_COUNTRY"] = os.getenv("SF_HOL_COUNTRY") or "US"
    hol_subdiv = os.getenv("SF_HOL_SUBDIV")
    if hol_subdiv:
        os.environ["SF_HOL_SUBDIV"] = hol_subdiv
    else:
        os.environ.pop("SF_HOL_SUBDIV", None)

    # Horizons: use env if provided; else from AppConfig default list
    if not os.getenv("SF_HORIZONS"):
        hs = cfg.train.horizons  # e.g., [1,2,3,4,5,6,7]
        os.environ["SF_HORIZONS"] = ",".join(map(str, hs))

    # Run the ASGI app defined as an import string
    uvicorn.run(
        "boost_sales.api.server:app",
        host=host,
        port=port,
        reload=reload,
        factory=False,
    )

if __name__ == "__main__":
    app()
