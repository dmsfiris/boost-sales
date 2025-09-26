# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Optional

import pandas as pd

from boost_sales.config import AppConfig
from boost_sales.features.time_features import add_calendar
from boost_sales.features.lag_features import add_lags_and_rollups
from boost_sales.features.holidays import add_holidays


def prepare_features(
    df: pd.DataFrame,
    cfg: AppConfig,
    *,
    hol_country: Optional[str] = None,
    hol_subdiv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Horizon-agnostic features used at both train and inference.
    Mirrors training pipeline for parity.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing at least the configured date/store/item/sales columns.
    cfg : AppConfig
        Global configuration (column names, feature flags, etc.).
    hol_country, hol_subdiv : optional
        If provided, override the holidays region; otherwise use cfg.train settings.
    """
    c = cfg.columns

    # Ensure date column is datetime (hardening if caller didn't parse dates)
    if not pd.api.types.is_datetime64_any_dtype(df[c.date]):
        df = df.copy()
        df[c.date] = pd.to_datetime(df[c.date], errors="coerce")

    # Calendar features
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

    # Lags / rolling features (optionally includes price rolls)
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

    # Holiday flag (default to cfg.train region)
    hc = hol_country if hol_country is not None else cfg.train.hol_country
    hs = hol_subdiv if hol_subdiv is not None else cfg.train.hol_subdiv
    df = add_holidays(
        df,
        date_col=c.date,
        country=hc,
        subdiv=hs,
        out_col="is_hol",
    )

    return df
