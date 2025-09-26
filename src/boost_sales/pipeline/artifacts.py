# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional


def _slug(part: str) -> str:
    """
    Safe-ish slug for filesystem paths: keep alnum, dash, underscore, dot; replace others with '_'.
    Collapse runs of '_' and strip edges.
    """
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(part))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("._") or "NA"


def outdir_for(scope: str, models_dir: Path, store: Optional[str] = None, item: Optional[str] = None) -> Path:
    """
    Compute the output directory for artifacts based on grouping scope.

    Layout (unchanged):
      global:   {models_dir}
      pair:     {models_dir}/by_pair/{store}__{item}
      item:     {models_dir}/by_item/{item}
      store:    {models_dir}/by_store/{store}
    """
    models_dir = Path(models_dir)

    if scope == "pair":
        if store is None or item is None:
            raise ValueError("scope='pair' requires both store and item.")
        return models_dir / "by_pair" / f"{_slug(store)}__{_slug(item)}"

    if scope == "item":
        if item is None:
            raise ValueError("scope='item' requires item.")
        return models_dir / "by_item" / _slug(item)

    if scope == "store":
        if store is None:
            raise ValueError("scope='store' requires store.")
        return models_dir / "by_store" / _slug(store)

    # global/unknown -> root
    return models_dir


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Write text atomically to avoid partial files (write to tmp, then replace).
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_categories(outdir: Path, store_ids: Iterable[object], item_ids: Iterable[object]) -> None:
    """
    Persist basic category info present in the group; useful for serving/debug.
    Writes {outdir}/categories.json with:
      {"stores": [...], "items": [...]}

    Deduplicates and stringifies ids; sorts for stability.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "stores": sorted({str(s) for s in store_ids}),
        "items": sorted({str(i) for i in item_ids}),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    _atomic_write_text(outdir / "categories.json", text)
