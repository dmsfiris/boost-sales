# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from boost_sales.config import AppConfig
from .schemas import ForecastRequest, ForecastResponse
from .service import forecast, forecast_from_csv_bytes

from boost_sales.api.core.horizons import parse_horizons_opt


class APIServer:
    """Thin wrapper that builds a configured FastAPI app."""

    def __init__(
        self,
        *,
        models_dir: Path,
        data_csv: Optional[Path] = None,
        hol_country: str = "US",
        hol_subdiv: Optional[str] = None,
        horizons: Optional[list[int]] = None,  # already parsed
        templates_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = AppConfig()
        self.cfg.paths.models_dir = Path(models_dir)
        if data_csv is not None:
            self.cfg.paths.data_csv = Path(data_csv)
        self.cfg.train.hol_country = hol_country
        self.cfg.train.hol_subdiv = hol_subdiv
        if horizons:
            self.cfg.train.horizons = list(horizons)  # type: ignore[assignment]

        # ---------- Templates (package-level) ----------
        # server.py is sales_forecast/api/server.py; templates at sales_forecast/templates
        if templates_dir is None:
            templates_dir = Path(__file__).resolve().parents[1] / "templates"
        self.templates = Jinja2Templates(directory=str(templates_dir))

        self.app = FastAPI(
            title="Sales Forecast API",
            version="0.1.0",
            description="Legacy-compatible forecasting endpoints with richer scopes.",
        )

        # Expose a couple defaults to templates (e.g., training/forecast pages)
        self.app.state.models_dir = str(self.cfg.paths.models_dir)

        # ---------- Static (optional, package-level only) ----------
        static_dir = Path(__file__).resolve().parents[1] / "static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        self._mount_routes()

    def _mount_routes(self) -> None:
        app = self.app

        @app.get("/health")
        def health() -> dict:
            return {"status": "ok"}

        # ---------------- UI pages ----------------

        @app.get("/", response_class=HTMLResponse)
        def ui_home(request: Request) -> HTMLResponse:
            """
            Controller home: links to Forecast and Training pages.
            """
            return self.templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "year": datetime.now().year,
                },
            )

        @app.get("/forecast", response_class=HTMLResponse)
        def ui_forecast(request: Request) -> HTMLResponse:
            """
            Forecast page (uses your existing partials/forms + results).
            """
            dc = getattr(self.cfg.paths, "data_csv", None)
            has_demo_csv = bool(dc and Path(dc).exists())
            return self.templates.TemplateResponse(
                "forecast.html",
                {
                    "request": request,
                    "default_horizons": ",".join(map(str, self.cfg.train.horizons)),
                    "hol_country": self.cfg.train.hol_country,
                    "hol_subdiv": self.cfg.train.hol_subdiv or "",
                    "has_demo_csv": has_demo_csv,
                    "models_dir": str(self.cfg.paths.models_dir),  # default for the form
                    "year": datetime.now().year,
                },
            )

        @app.get("/training", response_class=HTMLResponse)
        def ui_training(request: Request) -> HTMLResponse:
            """
            Training page (shows the new training form).
            """
            return self.templates.TemplateResponse(
                "training.html",
                {
                    "request": request,
                    "default_horizons": ",".join(map(str, self.cfg.train.horizons)),
                    "hol_country": self.cfg.train.hol_country,
                    "hol_subdiv": self.cfg.train.hol_subdiv or "",
                    "models_dir": str(self.cfg.paths.models_dir),
                    "year": datetime.now().year,
                },
            )

        # ---------------- Forecast API endpoints ----------------
        # Support an optional models_dir override for both JSON and multipart flows.

        @app.post("/forecast", response_model=ForecastResponse)
        def post_forecast(
            req: ForecastRequest,
            models_dir: Optional[str] = Body(
                None,
                description="Optional override for models directory used to load trained models.",
            ),
        ) -> ForecastResponse:
            try:
                # clone base config, then apply models_dir override if provided
                cfg = AppConfig(**self.cfg.model_dump())
                if models_dir:
                    cfg.paths.models_dir = Path(models_dir)
                return forecast(cfg, req)
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve)) from ve
            except Exception as e:  # pragma: no cover
                raise HTTPException(status_code=500, detail=f"Forecast failed: {e}") from e

        @app.post("/forecast/csv", response_model=ForecastResponse)
        async def post_forecast_csv(
            file: Optional[UploadFile] = File(None),  # optional to allow demo CSV
            scope: str = Form("single"),
            store_id: Optional[str] = Form(None),
            item_id: Optional[str] = Form(None),
            horizons: Optional[str] = Form(None),
            use_demo_csv: bool = Form(False),
            # new plan + formatting fields to match schema
            price_future: Optional[str] = Form(None),
            promo_future: Optional[str] = Form(None),
            unit_type: Optional[str] = Form(None),        # "integer" | "float"
            decimal_places: Optional[int] = Form(None),
            page: int = Form(1),
            page_size: int = Form(100),
            # models override
            models_dir: Optional[str] = Form(
                None, description="Optional override for models directory."
            ),
        ) -> ForecastResponse:
            try:
                # clone base config, then apply models_dir override if provided
                cfg = AppConfig(**self.cfg.model_dump())
                if models_dir:
                    cfg.paths.models_dir = Path(models_dir)

                req = ForecastRequest(
                    scope=scope,
                    store_id=store_id,
                    item_id=item_id,
                    horizons=horizons,
                    use_server_csv=use_demo_csv,
                    price_future=price_future,
                    promo_future=promo_future,
                    unit_type=unit_type,               # pydantic will validate Literal
                    decimal_places=decimal_places,
                    page=page,
                    page_size=page_size,
                )

                if use_demo_csv:
                    return forecast(cfg, req)

                if file is None:
                    raise ValueError("Upload a CSV file or set 'use_demo_csv=true'.")
                csv_bytes = await file.read()
                if not csv_bytes:
                    raise ValueError("Uploaded CSV is empty.")
                return forecast_from_csv_bytes(cfg, csv_bytes, req)

            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve)) from ve
            except Exception as e:  # pragma: no cover
                raise HTTPException(status_code=500, detail=f"Forecast failed: {e}") from e

        # ---------------- Training: auto-suggest for early_stopping_rounds ----------------

        @app.post("/train/suggest_es")
        async def suggest_early_stopping_rounds(
            n_estimators: int = Form(..., description="Total trees planned"),
            valid_tail_days: Optional[int] = Form(
                None, description="Validation window length (last N days)."
            ),
            valid_cutoff_date: Optional[str] = Form(
                None, description="Explicit cutoff date; used to assume a default tail if needed."
            ),
            cap_min: int = Form(20, description="Lower clamp for suggestion."),
            cap_max: int = Form(120, description="Upper clamp for suggestion."),
        ):
            """
            Suggest a patience (early_stopping_rounds) based on n_estimators and validation window size.

            Heuristic:
              - Shorter validation windows are noisier → higher % of n_estimators (more patience).
              - Longer windows are calmer → lower %.
              - If only a cutoff date is provided (no N), assume 28 days as a sensible default.
            """
            if n_estimators <= 0:
                raise HTTPException(status_code=400, detail="n_estimators must be > 0.")

            # Effective validation length in days
            tail_days = valid_tail_days
            if (tail_days is None or tail_days <= 0) and (valid_cutoff_date or "").strip():
                tail_days = 28  # sensible default when only a cutoff is supplied

            def pct_for_days(d: Optional[int]) -> float:
                if not d or d <= 0:
                    return 0.12
                if d <= 7:
                    return 0.18
                if d <= 14:
                    return 0.15
                if d <= 28:
                    return 0.12
                if d <= 56:
                    return 0.10
                return 0.08

            pct = pct_for_days(tail_days)
            suggestion = int(round(n_estimators * pct))
            suggestion = max(cap_min, min(cap_max, suggestion))

            note = "Based on n_estimators and "
            if tail_days:
                note += f"validation window ≈ {tail_days} days."
            elif (valid_cutoff_date or "").strip():
                note += "a cutoff date (assumed 28-day window)."
            else:
                note += "no validation split (neutral %. Consider adding a split for early stopping to work)."

            return JSONResponse(
                {
                    "ok": True,
                    "suggestion": suggestion,
                    "percent": pct,
                    "n_estimators": n_estimators,
                    "used_tail_days": tail_days,
                    "assumed_default": bool((valid_cutoff_date or "").strip() and not valid_tail_days),
                    "cap": {"min": cap_min, "max": cap_max},
                    "note": note,
                }
            )

        # ---------------- Training API endpoint ----------------

        @app.post("/train")
        async def post_train(
            # mode & scope
            mode: str = Form("global"),                 # "global" | "per_group"
            train_scope: Optional[str] = Form(None),    # "pair" | "item" | "store" (when per_group)

            # data
            use_demo_csv: bool = Form(True),
            file: Optional[UploadFile] = File(None),

            # paths / horizons / holidays
            models_dir: str = Form("models"),
            horizons: Optional[str] = Form(None),
            hol_country: str = Form("US"),
            hol_subdiv: Optional[str] = Form(None),
            wipe: bool = Form(False),

            # xgb knobs (optional)
            nthread: Optional[int] = Form(None),
            n_estimators: Optional[int] = Form(None),
            max_depth: Optional[int] = Form(None),
            learning_rate: Optional[float] = Form(None),
            tree_method: Optional[str] = Form(None),
            subsample: Optional[float] = Form(None),
            colsample_bytree: Optional[float] = Form(None),
            min_child_weight: Optional[float] = Form(None),
            gamma: Optional[float] = Form(None),
            reg_alpha: Optional[float] = Form(None),
            reg_lambda: Optional[float] = Form(None),
            max_bin: Optional[int] = Form(None),
            random_state: Optional[int] = Form(None),
            required_feature_notna: Optional[str] = Form(None),  # comma-separated

            # validation
            valid_cutoff_date: Optional[str] = Form(None),
            valid_tail_days: Optional[int] = Form(None),         # honor window (last N days)
            early_stopping_rounds: Optional[int] = Form(None),
            verbose_eval: Optional[int] = Form(0),
            enforce_single_thread_env: bool = Form(False),
        ):
            """
            Synchronous training trigger used by the Training page.
            Covers global & per-group modes, optional wipe, full XGB knobs,
            server/demo vs uploaded CSV, and validation settings.
            """
            from time import perf_counter
            from boost_sales.pipeline.train import train_global, train_per_group
            from boost_sales.data.io import load_sales_csv
            import tempfile

            cfg = AppConfig(**self.cfg.model_dump())

            # Paths
            cfg.paths.models_dir = Path(models_dir)

            # Resolve data source
            tmp_path: Optional[Path] = None
            try:
                if use_demo_csv:
                    dc = getattr(self.cfg.paths, "data_csv", None)
                    if not dc or not Path(dc).exists():
                        raise HTTPException(status_code=400, detail="Server CSV not configured/available.")
                    cfg.paths.data_csv = Path(dc)
                else:
                    if file is None:
                        raise HTTPException(status_code=400, detail="Upload a CSV or enable 'Use server CSV'.")
                    contents = await file.read()
                    if not contents:
                        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        tmp.write(contents)
                        tmp_path = Path(tmp.name)
                    # quick validation
                    _ = load_sales_csv(
                        tmp_path,
                        parse_dates=True,
                        date_col=cfg.columns.date,
                        store_col=cfg.columns.store,
                        item_col=cfg.columns.item,
                        sales_col=cfg.columns.sales,
                        price_col=cfg.columns.price,
                        promo_col=cfg.columns.promo,
                    )
                    cfg.paths.data_csv = tmp_path

                # Holidays
                cfg.train.hol_country = hol_country
                cfg.train.hol_subdiv = hol_subdiv

                # Horizons
                cfg.train.horizons = parse_horizons_opt(horizons, cfg.train.horizons)

                # Apply XGB knobs if provided
                def set_if(name, val):
                    if val is not None and val != "":
                        setattr(cfg.train, name, val)

                for name, val in [
                    ("nthread", nthread),
                    ("n_estimators", n_estimators),
                    ("max_depth", max_depth),
                    ("learning_rate", learning_rate),
                    ("tree_method", tree_method),
                    ("subsample", subsample),
                    ("colsample_bytree", colsample_bytree),
                    ("min_child_weight", min_child_weight),
                    ("gamma", gamma),
                    ("reg_alpha", reg_alpha),
                    ("reg_lambda", reg_lambda),
                    ("max_bin", max_bin),
                    ("random_state", random_state),
                    ("valid_cutoff_date", valid_cutoff_date),
                    ("valid_tail_days", valid_tail_days),
                    ("early_stopping_rounds", early_stopping_rounds),
                    ("verbose_eval", verbose_eval),
                    ("enforce_single_thread_env", enforce_single_thread_env),
                ]:
                    set_if(name, val)

                # required_feature_notna parsing
                if required_feature_notna:
                    rf = [s.strip() for s in required_feature_notna.split(",") if s.strip()]
                    cfg.train.required_feature_notna = rf

                # Optional wipe
                mdir = Path(models_dir)
                if wipe:
                    if mode == "global":
                        shutil.rmtree(mdir, ignore_errors=True)
                    else:
                        sub = {"pair": "by_pair", "item": "by_item", "store": "by_store"}.get(train_scope or "", "")
                        if not sub:
                            raise HTTPException(status_code=400, detail="train_scope must be one of: pair, item, store.")
                        shutil.rmtree(mdir / sub, ignore_errors=True)

                # Train
                t0 = perf_counter()
                if mode == "global":
                    train_global(cfg)
                    trained = {"mode": "global", "groups": 1}
                elif mode == "per_group":
                    scope = train_scope or "pair"
                    if scope not in {"pair", "item", "store"}:
                        raise HTTPException(status_code=400, detail="train_scope must be one of: pair, item, store.")
                    train_per_group(cfg, scope)
                    trained = {"mode": "per_group", "scope": scope}
                else:
                    raise HTTPException(status_code=400, detail="mode must be one of: global, per_group")
                dt = perf_counter() - t0

                return JSONResponse(
                    {
                        "ok": True,
                        "trained": trained,
                        "models_dir": str(cfg.paths.models_dir),
                        "horizons": cfg.train.horizons,
                        "holidays": {"country": cfg.train.hol_country, "subdiv": cfg.train.hol_subdiv},
                        "seconds": round(dt, 2),
                        "note": "Training finished.",
                    }
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Training failed: {e}")
            finally:
                # cleanup temp upload if used
                try:
                    if tmp_path and tmp_path.exists():
                        os.unlink(tmp_path)
                except Exception:
                    pass


# -------- factory from ENV for uvicorn import-string & reload --------
def create_app(
    *,
    models_dir: Path,
    data_csv: Optional[Path] = None,
    hol_country: str = "US",
    hol_subdiv: Optional[str] = None,
    horizons: Optional[list[int]] = None,
) -> FastAPI:
    server = APIServer(
        models_dir=models_dir,
        data_csv=data_csv,
        hol_country=hol_country,
        hol_subdiv=hol_subdiv,
        horizons=horizons,
    )
    return server.app


def create_app_from_env() -> FastAPI:
    """
    Build the FastAPI app from environment variables set by the CLI:
      SF_MODELS_DIR (required)
      SF_DATA_CSV   (optional)
      SF_HOL_COUNTRY (optional, default 'US')
      SF_HOL_SUBDIV  (optional)
      SF_HORIZONS    (optional, e.g. '1-7')
    """
    models_dir = os.environ.get("SF_MODELS_DIR")
    if not models_dir:
        raise RuntimeError("Missing SF_MODELS_DIR env var (set by CLI serve-web).")
    data_csv = os.environ.get("SF_DATA_CSV")
    hol_country = os.environ.get("SF_HOL_COUNTRY", "US")
    hol_subdiv = os.environ.get("SF_HOL_SUBDIV") or None
    horizons_raw = os.environ.get("SF_HORIZONS") or None

    cfg_default = AppConfig()
    horizons = parse_horizons_opt(horizons_raw, cfg_default.train.horizons)

    server = APIServer(
        models_dir=Path(models_dir),
        data_csv=Path(data_csv) if data_csv else None,
        hol_country=hol_country,
        hol_subdiv=hol_subdiv,
        horizons=horizons,
    )
    return server.app


# Module-level ASGI app for: uvicorn boost_sales.api.server:app --reload
try:
    app = create_app_from_env()
except Exception as e:
    _err_msg = f"Server misconfigured: {e}"
    app = FastAPI(title="Sales Forecast API (error)")

    @app.get("/", response_class=HTMLResponse)
    def _startup_error(request: Request):
        html = f"""
        <html><body>
        <h3>Server misconfigured</h3>
        <p>{_err_msg}</p>
        <p>Set <code>SF_MODELS_DIR</code> and optionally <code>SF_DATA_CSV</code>, then restart.</p>
        </body></html>
        """
        return HTMLResponse(html)
