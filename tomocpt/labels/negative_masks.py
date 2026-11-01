"""Negative masks (Gate 7): segmentations of structures that are never a class (e.g. membranes).

``prepData.negative_masks = "<name>:<mask dir>,..."``. MRC masks are matched to tomograms by
name (as ``prepData.filament_masks``); ``prepare_data`` records them in the tracking CSV
(column ``negative_mask_paths``, paths joined by ``;``). Pipeline-v2 chunking resamples them with
the tomogram and stores one packed bit mask per chunk; the dataloader returns them as the batch
key ``negative_mask``. They only feed the contrastive loss's negatives, never the targets.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

NEGATIVE_MASK_PATHS_COLUMN = "negative_mask_paths"
_SUFFIXES = (".mrc", ".rec", ".map")


def parse_negative_masks(text) -> List[Tuple[str, Path]]:
    """``"<name>:<dir>,..."`` -> ``[(name, dir)]``; ``None`` / empty -> ``[]``."""
    if text is None:
        return []
    s = str(text).strip()
    if not s or s in ("None", "???"):
        return []
    out = []
    for entry in (e.strip() for e in s.split(",")):
        if not entry:
            continue
        name, sep, d = entry.partition(":")
        if not sep or not name.strip() or not d.strip():
            raise ValueError(f"Malformed negative_masks entry '{entry}': expected '<name>:<mask dir>'")
        out.append((name.strip(), Path(d.strip())))
    return out


def split_paths(value) -> List[str]:
    """Paths of a ``negative_mask_paths`` CSV cell (``;``-joined; NaN / empty -> ``[]``)."""
    if value is None or (isinstance(value, float) and value != value):
        return []
    return [p for p in str(value).split(";") if p.strip() and p.strip().lower() != "nan"]


def add_negative_mask_paths(prep_cfg, tomo_dir, tracking_csv: Optional[Path]) -> None:
    """Add the column ``negative_mask_paths`` to one dataset's tracking CSV (every mask matched
    to the row's tomogram, over every ``negative_masks`` entry)."""
    import pandas as pd

    from tomocpt.configManager.configManager import safe_getattr
    from tomocpt.labels.filaments import _tomogram_files, match_files_to_tomograms

    entries = parse_negative_masks(safe_getattr(prep_cfg, "negative_masks", None))
    if not entries or tracking_csv is None or not Path(tracking_csv).is_file():
        return
    stems = list(_tomogram_files(Path(tomo_dir)))
    matched = {}
    for name, d in entries:
        if not d.is_dir():
            raise FileNotFoundError(f"Negative mask directory '{name}' not found: {d}")
        files = [f for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in _SUFFIXES]
        for stem, f in match_files_to_tomograms(files, stems).items():
            matched.setdefault(stem, []).append(str(Path(f).resolve()))
    df = pd.read_csv(tracking_csv)
    df[NEGATIVE_MASK_PATHS_COLUMN] = [";".join(matched.get(Path(str(p)).stem, []))
                                      for p in df["tomogram_path"]]
    df.to_csv(tracking_csv, index=False)
