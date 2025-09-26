# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, field_validator

# =========================
# Paths & column names
# =========================


class Paths(BaseModel):
    """Centralized filesystem locations."""

    data_csv: Path = Field(default=Path("data/sales.csv"))
    models_dir: Path = Field(default=Path("models"))


class Columns(BaseModel):
    """Rename these if your legacy used different column names."""

    date: str = "date"
    store: str = "store_id"
    item: str = "item_id"
    sales: str = "sales"
    price: str = "price"
    promo: str = "promo"

    @field_validator("date", "store", "item", "sales", "price", "promo")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Column names must be non-empty.")
        return v


# =========================
# Feature config
# =========================


class CalendarConfig(BaseModel):
    # Quality: calendar features are cheap and often helpful
    add_year: bool = True
    add_quarter: bool = True
    add_month: bool = True
    add_day: bool = True
    add_dow: bool = True
    add_is_weekend: bool = True
    # Turn these on only if you know your legacy used them (some leakage risk if used carelessly)
    add_weekofyear: bool = False
    add_weekofmonth: bool = False


class LagRollConfig(BaseModel):
    # Auto-synced via AppConfig.model_post_init
    group_cols: Sequence[str] = ("store_id", "item_id")

    # Quality vs speed:
    # - lag 1/7/14/28 gives short/weekly/biweekly/monthly memory (good quality)
    # - removing 28 is faster but can hurt weekly-cycle stability for longer horizons
    lag_steps: Sequence[int] = (1, 7, 14, 28)

    # Rolling windows:
    # - 7 & 28 are common; 28 is also used as the price ratio denominator below
    roll_windows: Sequence[int] = (7, 28)

    roll_use_target: bool = True  # keep: strong signal, cheap
    roll_min_periods: Optional[int] = None  # set to exact legacy if you must match outputs

    # Std adds features but costs extra compute; enable only if you want a (small) lift
    include_std: bool = False  # True = (slightly) better, slower

    include_price_roll: bool = True  # keep: needed for price ratio context
    price_col: str = "price"


class FutureControlsConfig(BaseModel):
    # If you trained with price/promo futures, serving must provide them (or you derive/assume).
    # If you WON’T provide futures at inference, set these 3 to False and retrain.
    denom_col: str = "price_roll_28"
    add_price_future: bool = True
    add_promo_future: bool = True
    add_price_ratio: bool = True
    safe_zero_denominator: bool = True


# =========================
# Output formatting
# =========================


class OutputUnitConfig(BaseModel):
    """
    Controls how forecasted values are formatted in outputs (post-processing only).
    - unit_type="integer": round to nearest int (e.g., unit counts).
    - unit_type="float": round to decimal_places (e.g., weight).
    """

    unit_type: Literal["integer", "float"] = "float"
    decimal_places: int = 2

    @field_validator("decimal_places")
    @classmethod
    def _clamp_decimals(cls, v: int) -> int:
        # keep it sane; expand if you really need more precision
        try:
            iv = int(v)
        except Exception:
            iv = 2
        return max(0, min(6, iv))


# =========================
# Training / modeling config
# =========================


class TrainingConfig(BaseModel):
    # Horizons to train; override via CLI if needed
    horizons: List[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])

    # Holiday region
    hol_country: str = "US"
    hol_subdiv: Optional[str] = None

    # Reproducibility & threading
    random_state: int = 42
    nthread: Optional[int] = None  # None = use all cores; CLI can override
    enforce_single_thread_env: bool = False  # True only if you need bit-for-bit parity

    # XGBoost knobs (quality ↔ speed)
    # - n_estimators: higher with early_stopping is often best (set big cap; let ES stop early)
    # - tree_method="hist": large speedup with minimal quality loss
    n_estimators: int = 2000  # cap; ES will usually stop < this
    max_depth: int = 6  # 6–8 typical; deeper = slower/overfit risk
    learning_rate: float = 0.05  # 0.03–0.1; lower = steadier but needs more trees
    subsample: Optional[float] = 0.8  # 0.7–0.9 helps generalization
    colsample_bytree: Optional[float] = 0.8  # 0.7–0.9 helps generalization
    min_child_weight: Optional[float] = None
    gamma: Optional[float] = None
    reg_alpha: Optional[float] = None
    reg_lambda: Optional[float] = None
    tree_method: Optional[str] = "hist"
    max_bin: Optional[int] = None  # set 512 for a tiny lift, slightly slower

    # Validation / early stopping
    # Use exactly ONE of these:
    #   - valid_cutoff_date: "YYYY-MM-DD" (explicit time-based split)
    #   - valid_tail_days: N (automatic: last N days used as validation)
    valid_cutoff_date: Optional[str] = None
    valid_tail_days: Optional[int] = 28  # set None to disable auto-cutoff

    early_stopping_rounds: Optional[int] = 36
    verbose_eval: Union[bool, int] = 0

    # Optional: require specific features to be not-null before training (parity with legacy)
    required_feature_notna: List[str] = Field(default_factory=list)

    # ---- light normalizations / guards ----
    @field_validator("hol_country")
    @classmethod
    def _norm_country(cls, v: str) -> str:
        v = (v or "").strip().upper()
        if not v:
            return "US"
        return v

    @field_validator("hol_subdiv")
    @classmethod
    def _norm_subdiv(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        return s.upper() if s else None

    @field_validator("nthread")
    @classmethod
    def _valid_nthread(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 1:
            raise ValueError("nthread must be >= 1 if provided.")
        return v

    @field_validator("valid_tail_days")
    @classmethod
    def _valid_tail_days(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 1:
            raise ValueError("valid_tail_days must be >= 1 if provided.")
        return v

    @field_validator("early_stopping_rounds")
    @classmethod
    def _valid_es(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 1:
            # treat non-positive as "disabled"
            return None
        return v


class AppConfig(BaseModel):
    """Top-level config passed through the pipeline."""

    paths: Paths = Field(default_factory=Paths)
    cols: Columns = Field(default_factory=Columns)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    lags: LagRollConfig = Field(default_factory=LagRollConfig)
    future: FutureControlsConfig = Field(default_factory=FutureControlsConfig)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputUnitConfig = Field(default_factory=OutputUnitConfig)

    # Back-compat alias (some modules reference cfg.columns instead of cfg.cols)
    @property
    def columns(self) -> Columns:
        return self.cols

    @field_validator("paths", mode="after")
    @classmethod
    def _ensure_dirs(cls, v: Paths) -> Paths:
        v.models_dir.mkdir(parents=True, exist_ok=True)
        v.data_csv.parent.mkdir(parents=True, exist_ok=True)
        return v

    def model_post_init(self, __context: Any) -> None:
        """
        Keep lags.group_cols and price_col synced with Columns when defaults are used.
        """
        default_pair: Tuple[str, str] = ("store_id", "item_id")
        try:
            if tuple(self.lags.group_cols) == default_pair:
                self.lags.group_cols = (self.cols.store, self.cols.item)
            if self.lags.price_col == "price":
                self.lags.price_col = self.cols.price
        except Exception:
            # don't fail construction if someone sets unusual values
            pass


# =========================
# Helpers
# =========================


def load_default() -> AppConfig:
    """Return a fresh config with defaults."""
    return AppConfig()


def xgb_params_from(cfg: AppConfig) -> dict:
    """
    Build an XGBoost param dict from config, including only keys with values.
    """
    p = cfg.train
    out = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": p.learning_rate,
        "max_depth": p.max_depth,
        "seed": p.random_state,
    }
    for k in (
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "tree_method",
        "max_bin",
        "nthread",
    ):
        val = getattr(p, k)
        if val is not None:
            out[k] = val
    return out
