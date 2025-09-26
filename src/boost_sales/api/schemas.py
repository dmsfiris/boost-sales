# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# -----------------------------
# Scopes
# -----------------------------
ForecastScope = Literal[
    "single",
    "latest_per_pair",
    "latest_per_store",
    "latest_per_item",
    "last_n_days",
    "since_date",
    "at_date",
]

# Output unit type (controls formatting of predictions)
UnitType = Literal["integer", "float"]


# -----------------------------
# Request / Response models
# -----------------------------
class ForecastRequest(BaseModel):
    """
    Request payload for forecasting endpoints.

    Notes
    -----
    - `horizons` is a human-friendly string (e.g., "1-7" or "1,2,4"), parsed in service/cli.
    - Data source can be the server's configured CSV (`use_server_csv=true`) or an uploaded CSV.
    - Scope indicates which rows are selected before expanding horizons.
    - Future plans:
        * price_future: single scalar multiplier for all horizons (e.g. "1.03"),
                        OR a CSV list of absolute prices per horizon (e.g. "12.5,12.7,12.9").
        * promo_future: promo *intensity* in [0,1].
                        Either a single scalar applied to all horizons (e.g. "0.5" for 50% off),
                        OR a CSV list per horizon (e.g. "0,0,0.5,0,0,0,0").
                        0 means no promo; 1 means full promotion.
    - `unit_type`/`decimal_places` optionally override server defaults for output formatting.
    """

    # Prefer rejecting unexpected properties to surface client mistakes early (Pydantic v2 style)
    model_config = ConfigDict(extra="forbid")

    # ---- What to predict (scope selection) ----
    scope: ForecastScope = Field(
        "single",
        description=(
            "Forecast scope. "
            "single | latest_per_pair | latest_per_store | latest_per_item | "
            "last_n_days | since_date | at_date"
        ),
    )

    # Target identifiers (used by certain scopes)
    store_id: Optional[str] = Field(
        None, description="Store identifier. Required for scope=single and latest_per_store."
    )
    item_id: Optional[str] = Field(None, description="Item identifier. Required for scope=single and latest_per_item.")

    # Horizons as a string (e.g., '1-7' or '1,2,4'); parsing happens in service/cli
    horizons: Optional[str] = Field(
        None,
        description="Horizons to score, e.g. '1-7' or '1,2,4'. If omitted, the service default is used.",
    )

    # Dataset source flags
    use_server_csv: bool = Field(
        False,
        description="If true, use the server's demo CSV (when configured) instead of an uploaded file.",
    )

    # ---- Future plans (scalar or CSV) ----
    price_future: Optional[str] = Field(
        None,
        description=(
            "Future price plan: either a single scalar multiplier for all horizons (e.g. '1.03'), "
            "or a comma-separated list of absolute prices per horizon (e.g. '12.5,12.7,12.9')."
        ),
    )
    promo_future: Optional[str] = Field(
        None,
        description=(
            "Future promo plan: a promo intensity in [0,1]. "
            "Either a single scalar applied to all horizons (e.g. '0.5' for 50% off), "
            "or a comma-separated list per horizon (e.g. '0,0,0.5,0,0,0,0'). "
            "Use 0 for no promo and 1 for full promotion."
        ),
    )

    # Scope parameters (only some are required depending on scope)
    n_days: Optional[int] = Field(
        None,
        ge=1,
        description="For scope=last_n_days, the trailing number of days to include.",
    )
    since_date: Optional[date] = Field(None, description="For scope=since_date, include rows on/after this date.")
    at_date: Optional[date] = Field(None, description="For scope=at_date, include rows exactly on this date.")

    # Output formatting overrides (optional; falls back to server config if omitted)
    unit_type: Optional[UnitType] = Field(None, description="Override output unit type: 'integer' or 'float'.")
    decimal_places: Optional[int] = Field(
        None,
        ge=0,
        le=6,
        description="When unit_type='float', round to this many decimals (default from server config).",
    )

    # Paging (applies to all scopes; server can paginate results)
    page: int = Field(1, ge=1, description="1-based page index.")
    page_size: int = Field(10, ge=1, le=10000, description="Rows per page (max 10,000).")

    # ---- Cross-field validation for scope & unit requirements ----
    @model_validator(mode="after")
    def _validate(self) -> "ForecastRequest":
        s = self.scope

        if s == "single":
            if not self.store_id or not self.item_id:
                raise ValueError("scope=single requires both 'store_id' and 'item_id'.")
        elif s == "latest_per_store":
            if not self.store_id:
                raise ValueError("scope=latest_per_store requires 'store_id'.")
        elif s == "latest_per_item":
            if not self.item_id:
                raise ValueError("scope=latest_per_item requires 'item_id'.")
        elif s == "last_n_days":
            if not self.n_days or self.n_days < 1:
                raise ValueError("scope=last_n_days requires a positive 'n_days'.")
        elif s == "since_date":
            if not self.since_date:
                raise ValueError("scope=since_date requires 'since_date'.")
        elif s == "at_date":
            if not self.at_date:
                raise ValueError("scope=at_date requires 'at_date'.")
        elif s == "latest_per_pair":
            pass
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported scope '{s}'.")

        # Unit rules: if decimal_places provided, unit_type must be 'float'
        if self.decimal_places is not None and self.unit_type == "integer":
            raise ValueError("decimal_places cannot be used with unit_type='integer'.")
        return self


class PredictionRow(BaseModel):
    store_id: str
    item_id: str
    base_date: date
    target_date: date
    horizon: int
    sales: float


class PageMeta(BaseModel):
    total: int
    page: int
    page_size: int


class ForecastResponse(BaseModel):
    predictions: List[PredictionRow]
    page: PageMeta
    unit_type: UnitType = Field(description="Unit type used for sales values.")
    decimal_places: int = Field(description="Decimal places used when unit_type='float'.")
    notes: Optional[str] = None
