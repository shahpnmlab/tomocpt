"""Compare two golden files (``.json`` or ``.npz``) before regenerating them (spec Section 3.4).

``python -m benchmarks.perf.compare_goldens OLD NEW [--tol-abs X] [--tol-rel Y]`` prints the
largest differences and exits 1 if a key is missing / added / mismatched or a tolerance is exceeded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _flatten(obj, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}/{k}" if prefix else str(k), out)
    elif isinstance(obj, list) and obj and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
        out[prefix] = np.asarray(obj, dtype=np.float64)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten(v, f"{prefix}/{i}" if prefix else str(i), out)
    else:
        out[prefix] = obj
    return out


def _load(path) -> dict:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    if path.suffix == ".json":
        return _flatten(json.loads(path.read_text()))
    raise ValueError(f"unsupported golden file type: {path}")


def _numeric(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    return isinstance(v, np.ndarray) and (np.issubdtype(v.dtype, np.number) or v.dtype == bool)


def compare(old_path, new_path) -> dict:
    """``{"max_abs", "max_rel", "keys": {name: {"max_abs", "max_rel"}}, "missing", "added",
    "mismatched"}``. Numeric values (scalars, arrays) get absolute and relative differences (relative
    to ``|old|``, 0 where both are 0); other values and differing shapes go to ``mismatched``."""
    old, new = _load(old_path), _load(new_path)
    res = {"max_abs": 0.0, "max_rel": 0.0, "keys": {}, "missing": sorted(k for k in old if k not in new),
           "added": sorted(k for k in new if k not in old), "mismatched": []}
    for k in sorted(k for k in old if k in new):
        a, b = old[k], new[k]
        if _numeric(a) and _numeric(b):
            a64 = np.asarray(a, dtype=np.float64)
            b64 = np.asarray(b, dtype=np.float64)
            if a64.shape != b64.shape:
                res["mismatched"].append(k)
                continue
            if a64.size == 0:
                res["keys"][k] = {"max_abs": 0.0, "max_rel": 0.0}
                continue
            both_nan = np.isnan(a64) & np.isnan(b64)
            d = np.where(both_nan, 0.0, np.abs(a64 - b64))
            den = np.abs(a64)
            rel = np.where(d == 0, 0.0, d / np.where(den > 0, den, np.inf))
            rel = np.where((den == 0) & (d > 0), np.inf, rel)
            ma, mr = float(np.nanmax(d)), float(np.nanmax(rel))
            res["keys"][k] = {"max_abs": ma, "max_rel": mr}
            res["max_abs"] = max(res["max_abs"], ma)
            res["max_rel"] = max(res["max_rel"], mr)
        else:
            same = (a == b) if not isinstance(a, np.ndarray) else (
                isinstance(b, np.ndarray) and a.shape == b.shape and bool(np.all(a == b)))
            res["keys"][k] = {"max_abs": 0.0 if same else float("nan"), "max_rel": 0.0 if same else float("nan")}
            if not same:
                res["mismatched"].append(k)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--tol-abs", type=float, default=None)
    p.add_argument("--tol-rel", type=float, default=None)
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args(argv)
    r = compare(a.old, a.new)
    worst = sorted(r["keys"].items(), key=lambda kv: -np.nan_to_num(kv[1]["max_abs"], nan=np.inf))[: a.top]
    print(f"max_abs {r['max_abs']:.3e}  max_rel {r['max_rel']:.3e}  keys {len(r['keys'])}")
    for k, v in worst:
        print(f"  {k}: abs {v['max_abs']:.3e} rel {v['max_rel']:.3e}")
    for name in ("missing", "added", "mismatched"):
        if r[name]:
            print(f"{name}: {r[name]}")
    bad = bool(r["missing"] or r["added"] or r["mismatched"])
    bad |= a.tol_abs is not None and r["max_abs"] > a.tol_abs
    bad |= a.tol_rel is not None and r["max_rel"] > a.tol_rel
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
