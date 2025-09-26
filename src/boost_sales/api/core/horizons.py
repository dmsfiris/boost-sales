# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Optional, Sequence, List
import re


_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_INT_RE = re.compile(r"^\s*(\d+)\s*$")


def parse_horizons_opt(
    val: Optional[str],
    default: Sequence[int],
    *,
    min_horizon: int = 1,
    max_horizon: Optional[int] = None,
) -> List[int]:
    """
    Parse a human-friendly horizons string into a sorted, de-duplicated list of ints.

    Accepted forms
    --------------
    - Comma or space separated: "1,2,3", "1 2 3", "1, 2  3"
    - Ranges (inclusive): "1-3"  → [1,2,3]
    - Mixed: "1,2  4-7"
    - Descending ranges: "7-3"   → [7,6,5,4,3] (still de-duped & sorted in the end)
    """
    if min_horizon < 1:
        raise ValueError("min_horizon must be >= 1.")

    def _normalize(seq: Sequence[int]) -> List[int]:
        out = sorted({int(x) for x in seq})
        for n in out:
            _validate_bounds(n, min_horizon, max_horizon)
        return out

    # If not provided, just normalize the default
    if not val:
        return _normalize(default)

    tokens = [t for t in re.split(r"[,\s]+", val.strip()) if t]
    if not tokens:
        return _normalize(default)

    out: list[int] = []

    for tok in tokens:
        # Range like "a-b"
        m = _RANGE_RE.match(tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            _validate_bounds(a, min_horizon, max_horizon, label=f"left bound '{a}' of range '{tok}'")
            _validate_bounds(b, min_horizon, max_horizon, label=f"right bound '{b}' of range '{tok}'")
            step = 1 if a <= b else -1
            out.extend(range(a, b + step, step))
            continue

        # Single integer like "7"
        m = _INT_RE.match(tok)
        if m:
            n = int(m.group(1))
            _validate_bounds(n, min_horizon, max_horizon, label=f"token '{tok}'")
            out.append(n)
            continue

        # Anything else is malformed
        raise ValueError(
            f"Invalid horizon token: {tok!r}. Use integers and ranges like '1,2,4-7'."
        )

    return sorted(set(out))


def _validate_bounds(
    n: int,
    min_horizon: int,
    max_horizon: Optional[int],
    *,
    label: str | None = None,
) -> None:
    what = label or f"value '{n}'"
    if n < min_horizon:
        raise ValueError(f"Horizon {what} is < {min_horizon}.")
    if max_horizon is not None and n > max_horizon:
        raise ValueError(f"Horizon {what} is > {max_horizon}.")
