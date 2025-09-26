# SPDX-License-Identifier: MIT
"""
boost_sales.features
-----------------------

Public feature-engineering API re-exports.

Prefer importing from this package (stable surface) rather than individual
modules, e.g.:

    from boost_sales.features import add_calendar, add_lags_and_rollups, add_holidays, add_future_controls
"""

from __future__ import annotations

from .holidays import add_holidays
from .lag_features import add_lags_and_rollups

# Lightweight, explicit re-exports
from .time_features import add_calendar

__all__ = [
    "add_calendar",
    "add_lags_and_rollups",
    "add_holidays",
]
