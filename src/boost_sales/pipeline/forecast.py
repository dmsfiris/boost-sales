# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from pathlib import Path
import re
import typing as T
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from boost_sales.config import AppConfig
from boost_sales.data.io import load_sales_csv
from boost_sales.features.future_controls import add_future_controls  # no fallback
from boost_sales.features.holidays import add_holidays
from boost_sales.features.lag_features import add_lags_and_rollups
from boost_sales.features.time_features import add_calendar
from boost_sales.models.xgb import load_booster
from boost_sales.pipeline.train import train_global, train_per_group

app = typer.Typer(help="Sales Forecast CLI", add_completion=False)


# ----------------------------
# Helpers
# ----------------------------
def _parse_horizons_opt(val: T.Optional[str], default: T.Sequence[int]) -> T.List[int]:
    if not val:
        return list(default)
    tokens = re.split(r"[,\s]+", val.strip())
    out: T.List[int] = []
    for tok in tokens:
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a_i, b_i = int(a), int(b)
            if a_i <= 0 or b_i <= 0:
                raise typer.BadParameter("Horizon values must be positive integers.")
            step = 1 if a_i <= b_i else -1
            out.extend(range(a_i, b_i + step, step))
        else:
            n = int(tok)
            if n <= 0:
                raise typer.BadParameter("Horizon values must be positive integers.")
            out.append(n)
    return sorted(set(out))


def _prepare_features(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Build horizon-agnostic features (parity with training)."""
    c = cfg.cols

    df = add_calendar(
        df,
        date_col=c.date,
        add_weekofyear=cfg.calendar.add_weekofyear,
        add_weekofmonth=cfg.calendar.add_weekofmonth,
        add_quarter=cfg.calendar.add_quarter,
        add_year=cfg.calendar.add_year,
        add_month=cfg.calendar.add_month,
        add_day=cfg.calendar.add_day,
        add_dow=cfg.calendar.add_dow,
        add_is_weekend=cfg.calendar.add_is_weekend,
    )

    df = add_lags_and_rollups(
        df,
        group_cols=(c.store, c.item),
        date_col=c.date,
        target_col=c.sales,
        lag_steps=cfg.lags.lag_steps,
        roll_windows=cfg.lags.roll_windows,
        roll_use_target=cfg.lags.roll_use_target,
        roll_min_periods=cfg.lags.roll_min_periods,
        include_std=cfg.lags.include_std,
        include_price_roll=cfg.lags.include_price_roll,
        price_col=cfg.lags.price_col,
    )

    df = add_holidays(
        df,
        date_col=c.date,
        country=cfg.train.hol_country,
        subdiv=cfg.train.hol_subdiv,
        out_col="is_hol",
    )
    return df


def _import_webapp():
    """Return ("factory", create_app) or ("instance", app) from boost_sales.api.server."""
    try:
        from boost_sales.api.server import create_app  # type: ignore

        return "factory", create_app
    except Exception:
        pass
    try:
        from boost_sales.api.server import app as fastapi_app  # type: ignore

        return "instance", fastapi_app
    except Exception as e:
        raise RuntimeError("boost_sales.api.server must export either `create_app(...)` or `app`.") from e


# ----------------------------
# train (global)
# ----------------------------
@app.command("train")
def cmd_train(
    data_csv: Annotated[Path, typer.Option(help="Path to sales CSV.")] = ...,
    models_dir: Annotated[Path, typer.Option(help="Directory to write models.")] = ...,
    hol_country: Annotated[str, typer.Option(help="Holidays country code, e.g. 'US'.")] = "US",
    hol_subdiv: Annotated[T.Optional[str], typer.Option(help="Holidays subdivision.")] = None,
    horizons: Annotated[T.Optional[str], typer.Option(help="Horizons like '1-7' or '1,2,3'.")] = None,
    nthread: Annotated[int, typer.Option(help="XGBoost nthread parameter.")] = 1,
    verbose_eval: Annotated[int, typer.Option(help="XGBoost training verbosity (0/1).")] = 0,
    enforce_single_thread_env: Annotated[
        bool, typer.Option(help="Set single-thread env vars (OMP, MKL) for reproducibility.")
    ] = False,
):
    cfg = AppConfig()
    cfg.paths.data_csv = data_csv
    cfg.paths.models_dir = models_dir
    cfg.train.hol_country = hol_country
    cfg.train.hol_subdiv = hol_subdiv
    cfg.train.nthread = nthread
    cfg.train.verbose_eval = verbose_eval
    cfg.train.enforce_single_thread_env = enforce_single_thread_env
    cfg.train.horizons = _parse_horizons_opt(horizons, cfg.train.horizons)

    train_global(cfg)
    typer.echo(f"✅ Trained horizons {cfg.train.horizons} → {models_dir}")


# ----------------------------
# train-per-group
# ----------------------------
@app.command("train-per-group")
def cmd_train_per_group(
    scope: Annotated[str, typer.Argument(help="Scope: 'pair' (store+item), 'item', or 'store'.")] = ...,
    data_csv: Annotated[Path, typer.Option(help="Path to sales CSV.")] = ...,
    models_dir: Annotated[Path, typer.Option(help="Root models directory.")] = ...,
    hol_country: Annotated[str, typer.Option(help="Holidays country code.")] = "US",
    hol_subdiv: Annotated[T.Optional[str], typer.Option(help="Holidays subdivision.")] = None,
    horizons: Annotated[T.Optional[str], typer.Option(help="Horizons like '1-7' or '1,2,3'.")] = None,
    nthread: Annotated[int, typer.Option(help="XGBoost nthread parameter.")] = 1,
    verbose_eval: Annotated[int, typer.Option(help="XGBoost training verbosity (0/1).")] = 0,
    enforce_single_thread_env: Annotated[bool, typer.Option(help="Set single-thread env vars to 1.")] = False,
):
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
    cfg.train.horizons = _parse_horizons_opt(horizons, cfg.train.horizons)

    train_per_group(cfg, scope)
    typer.echo(f"✅ Trained per-{scope} horizons {cfg.train.horizons} under {models_dir}")


# ----------------------------
# forecast
# ----------------------------
@app.command("forecast")
def cmd_forecast(
    data_csv: Annotated[Path, typer.Option(help="Path to sales CSV used for features.")] = ...,
    models_dir: Annotated[Path, typer.Option(help="Directory with trained models.")] = ...,
    out_csv: Annotated[T.Optional[Path], typer.Option(help="Where to write predictions.")] = None,
    latest_only: Annotated[
        bool, typer.Option(help="Keep only the latest base date per (store,item) before scoring.")
    ] = False,
    hol_country: Annotated[str, typer.Option(help="Holidays country code to use.")] = "US",
    hol_subdiv: Annotated[T.Optional[str], typer.Option(help="Holidays subdivision.")] = None,
    horizons: Annotated[T.Optional[str], typer.Option(help="Horizons to produce (e.g. '1-7').")] = None,
    assume_no_change: Annotated[
        bool,
        typer.Option(help="Fill price/promo_h{h} from current values; compute price_ratio_h{h} safely."),
    ] = False,
):
    cfg = AppConfig()
    cfg.paths.data_csv = data_csv
    cfg.paths.models_dir = models_dir
    cfg.train.hol_country = hol_country
    cfg.train.hol_subdiv = hol_subdiv
    hs = _parse_horizons_opt(horizons, cfg.train.horizons)

    # 1) Load & build horizon-agnostic features
    c = cfg.cols
    df = load_sales_csv(
        cfg.paths.data_csv,
        date_col=c.date,
        store_col=c.store,
        item_col=c.item,
        sales_col=c.sales,
        price_col=c.price,
        promo_col=c.promo,
        parse_dates=True,
    )
    df = _prepare_features(df, cfg)

    # 2) Horizon-specific controls + score
    all_preds: list[pd.DataFrame] = []
    for h in hs:
        dfh = add_future_controls(
            df,
            h,
            date_col=c.date,
            price_col=c.price,
            promo_col=c.promo,
            denom_col=cfg.future.denom_col,
            add_price_future=cfg.future.add_price_future,
            add_promo_future=cfg.future.add_promo_future,
            add_price_ratio=cfg.future.add_price_ratio,
            safe_zero_denominator=cfg.future.safe_zero_denominator,
        )

        if assume_no_change:
            ph = f"price_h{h}"
            pmh = f"promo_h{h}"
            prh = f"price_ratio_h{h}"

            if ph not in dfh.columns:
                dfh[ph] = np.nan
            dfh[ph] = dfh[ph].astype("float64", copy=False)
            need_ph = dfh[ph].isna()
            if need_ph.any():
                dfh.loc[need_ph, ph] = dfh.loc[need_ph, c.price].astype("float64")

            if pmh not in dfh.columns:
                dfh[pmh] = np.nan
            dfh[pmh] = dfh[pmh].astype("float64", copy=False)
            need_pmh = dfh[pmh].isna()
            if need_pmh.any():
                dfh.loc[need_pmh, pmh] = dfh.loc[need_pmh, c.promo].astype("float64")

            denom_name = cfg.future.denom_col
            if denom_name in dfh.columns:
                if prh not in dfh.columns:
                    dfh[prh] = np.nan
                dfh[prh] = dfh[prh].astype("float64", copy=False)
                num = dfh[ph].astype("float64")
                denom = dfh[denom_name].astype("float64")
                mask = denom.notna() & num.notna()
                if cfg.future.safe_zero_denominator:
                    mask &= denom.ne(0)
                dfh.loc[mask, prh] = np.divide(num[mask], denom[mask]).astype("float64")

        if latest_only:
            idx = dfh.groupby([c.store, c.item])[c.date].transform("max").eq(dfh[c.date])
            dfh = dfh[idx].copy()

        booster, feats = load_booster(models_dir, h)
        if not set(feats).issubset(dfh.columns):
            continue

        ok = dfh[feats].notna().all(axis=1)
        ready = dfh.loc[ok].copy()
        if ready.empty:
            continue

        import xgboost as xgb  # local import

        dmat = xgb.DMatrix(ready[feats], feature_names=feats)
        yhat = booster.predict(dmat)

        out = pd.DataFrame(
            {
                "store_id": ready[c.store].astype("string"),
                "item_id": ready[c.item].astype("string"),
                "base_date": pd.to_datetime(ready[c.date]).dt.date,
                "target_date": (pd.to_datetime(ready[c.date]) + pd.to_timedelta(h, unit="D")).dt.date,
                "horizon": h,
                "pred": yhat,
            }
        )
        all_preds.append(out)

    # Output path (avoid callable default)
    out_csv = out_csv or Path("preds.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not all_preds:
        pd.DataFrame(columns=["store_id", "item_id", "base_date", "target_date", "horizon", "pred"]).to_csv(
            out_csv, index=False
        )
        typer.echo("⚠️ No eligible rows to predict (missing features?). Empty file written.")
        return

    preds = pd.concat(all_preds, ignore_index=True).sort_values(["store_id", "item_id", "target_date", "horizon"])
    preds.to_csv(out_csv, index=False)
    typer.echo(f"✅ Wrote predictions ({len(preds):,} rows) → {out_csv}")


# ----------------------------
# serve-web (API + HTML)
# ----------------------------
@app.command("serve-web")
def cmd_serve_web(
    models_dir: Annotated[Path, typer.Option(help="Directory with trained models.")] = ...,
    data_csv: Annotated[T.Optional[Path], typer.Option(help="Optional demo CSV path.")] = None,
    hol_country: Annotated[str, typer.Option(help="Holidays country code to use.")] = "US",
    hol_subdiv: Annotated[T.Optional[str], typer.Option(help="Holidays subdivision.")] = None,
    horizons: Annotated[T.Optional[str], typer.Option(help="Horizons exposed by the service.")] = None,
    assume_no_change: Annotated[bool, typer.Option(help="Default assume-no-change behavior.")] = True,
    host: Annotated[str, typer.Option(help="Host to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload on code changes (dev only).")] = False,
):
    try:
        import uvicorn  # noqa: F401
    except Exception as e:
        raise RuntimeError("Web dependencies missing. Install with:  pip install '.[webui]'") from e

    cfg = AppConfig()
    hs = _parse_horizons_opt(horizons, cfg.train.horizons)

    if reload:
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
        os.environ["SF_ASSUME_NO_CHANGE"] = "1" if assume_no_change else "0"

        import uvicorn

        uvicorn.run("boost_sales.api.server:app", host=host, port=port, reload=True, factory=False)
        return

    kind, obj = _import_webapp()
    if kind == "factory":
        app_fastapi = obj(
            models_dir=models_dir,
            data_csv=data_csv,
            hol_country=hol_country,
            hol_subdiv=hol_subdiv,
            horizons=hs,
            assume_no_change=assume_no_change,
        )
    else:
        app_fastapi = obj
        app_fastapi.state.models_dir = models_dir
        app_fastapi.state.data_csv = data_csv
        app_fastapi.state.hol_country = hol_country
        app_fastapi.state.hol_subdiv = hol_subdiv
        app_fastapi.state.horizons = hs
        app_fastapi.state.assume_no_change = assume_no_change

    import uvicorn

    uvicorn.run(app_fastapi, host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
