"""Benchmark report (T4.6): ``benchmarks/out/<run>/report.json`` and ``report.md``.

``build_report`` aggregates the raw results of several runs (seeds, SNR levels) into
per-class F-beta (beta = 4), precision and recall (mean and sd over runs), cross-class
suppression counts, wall time per tomogram, peak GPU memory and the threshold used per
class with its source. Arms (b)-(d) are scored at the global threshold 0.3 and at the
calibrated thresholds. The key contract of ``report.json`` is in ``tests/gate4/README.md``.

Usage (regenerate the report of a finished run from its ``raw.json``)::

    python benchmarks/report.py benchmarks/out/<run>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ARMS = ("a", "b", "c", "d")
ARM_LABELS = {
    "a": "(a) one v1 model per species",
    "b": "(b) v2 multi-class, conditioned",
    "c": "(c) v2 multi-class, unconditioned",
    "d": "(d) v2 multi-class, conditioned, warm-started",
}
STATS = ("fbeta", "precision", "recall")


def _mean_sd(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return {"mean": 0.0, "sd": 0.0}
    return {"mean": float(v.mean()), "sd": float(v.std())}


def _aggregate_scoring(entries: List[Dict], class_names: Sequence[str]) -> Dict:
    """``entries``: the ``per_class`` dicts of one arm and mode, one per run."""
    per_class = {}
    for name in class_names:
        rows = [e[name] for e in entries if name in e]
        agg = {k: _mean_sd(r[k] for r in rows) for k in STATS}
        sources = Counter(r["threshold_source"] for r in rows)
        agg["threshold"] = float(np.mean([r["threshold"] for r in rows])) if rows else 0.0
        agg["threshold_source"] = sources.most_common(1)[0][0] if sources else "confidence_threshold"
        if len(sources) > 1:
            agg["threshold_sources"] = dict(sources)
        per_class[name] = agg
    return {"per_class": per_class}


def _aggregate_arm(results: List[Dict], class_names: Sequence[str]) -> Dict:
    walls = [r["wall_time_per_tomogram_s"] for r in results]
    peaks = [r["peak_gpu_memory_mb"] for r in results if r.get("peak_gpu_memory_mb") is not None]
    suppression: Dict[str, int] = {}
    for r in results:
        for k, n in r.get("cross_class_suppression", {}).items():
            suppression[k] = suppression.get(k, 0) + int(n)
    modes = sorted({m for r in results for m in r["scoring"]})
    arm = {
        "wall_time_per_tomogram_s": float(np.mean(walls)) if walls else 0.0,
        "wall_time_per_tomogram_sd_s": float(np.std(walls)) if walls else 0.0,
        "peak_gpu_memory_mb": float(max(peaks)) if peaks else None,
        "cross_class_suppression": suppression,
        "scoring": {m: _aggregate_scoring([r["scoring"][m]["per_class"] for r in results if m in r["scoring"]],
                                          class_names) for m in modes},
        "n_runs": len(results),
    }
    return arm


def build_report(raw: Dict) -> Dict:
    """``raw``: ``{"benchmark", "tiny", "classes", "runs": [{"label", "arms": {...}, ...}], ...}``."""
    class_names = [c["name"] for c in raw["classes"]]
    arms = {}
    for arm in ARMS:
        results = [run["arms"][arm] for run in raw["runs"] if arm in run["arms"]]
        if results:
            arms[arm] = _aggregate_arm(results, class_names)
    report = {
        "benchmark": raw["benchmark"],
        "tiny": bool(raw.get("tiny", False)),
        "classes": raw["classes"],
        "arms": arms,
        "runs": [{k: v for k, v in run.items() if k != "arms"} for run in raw["runs"]],
    }
    groups = sorted({run.get("group") for run in raw["runs"] if run.get("group") is not None})
    if len(groups) > 1:
        report["arms_by_group"] = {
            g: {arm: _aggregate_arm([run["arms"][arm] for run in raw["runs"]
                                     if run.get("group") == g and arm in run["arms"]], class_names)
                for arm in ARMS if any(arm in run["arms"] for run in raw["runs"] if run.get("group") == g)}
            for g in groups
        }
    for key in ("settings", "warm_start", "notes"):
        if key in raw:
            report[key] = raw[key]
    return report


def _fmt(e: Dict[str, float]) -> str:
    return f"{e['mean']:.3f} ± {e['sd']:.3f}"


def render_markdown(report: Dict) -> str:
    names = [c["name"] for c in report["classes"]]
    lines = [f"# Benchmark `{report['benchmark']}`" + (" (tiny)" if report.get("tiny") else ""), ""]
    lines.append("Classes: " + ", ".join(f"{c['name']} ({c['number']}, {c['diameter_ang']:g} A)"
                                         for c in report["classes"]))
    lines.append("")
    lines.append("| Arm | Wall time / tomogram (s) | Peak GPU memory (MB) |")
    lines.append("|---|---|---|")
    for arm, a in report["arms"].items():
        gpu = "n/a" if a["peak_gpu_memory_mb"] is None else f"{a['peak_gpu_memory_mb']:.0f}"
        lines.append(f"| {ARM_LABELS.get(arm, arm)} | {a['wall_time_per_tomogram_s']:.2f} | {gpu} |")
    for mode in ("global_0.3", "calibrated"):
        lines += ["", f"## Scoring: {mode} (F-beta, beta = 4; mean ± sd over runs)", ""]
        lines.append("| Class | " + " | ".join(ARM_LABELS.get(a, a) for a in report["arms"]
                                                if mode in report["arms"][a]["scoring"]) + " |")
        arms = [a for a in report["arms"] if mode in report["arms"][a]["scoring"]]
        lines.append("|---" * (len(arms) + 1) + "|")
        for n in names:
            cells = []
            for a in arms:
                e = report["arms"][a]["scoring"][mode]["per_class"][n]
                cells.append(f"F {_fmt(e['fbeta'])}<br>P {_fmt(e['precision'])}<br>R {_fmt(e['recall'])}"
                             f"<br>thr {e['threshold']:.2f} ({e['threshold_source']})")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Cross-class suppression (kept -> removed: count)", ""]
    for arm, a in report["arms"].items():
        s = ", ".join(f"{k}: {v}" for k, v in sorted(a["cross_class_suppression"].items())) or "none"
        lines.append(f"- {ARM_LABELS.get(arm, arm)}: {s}")
    if report.get("warm_start"):
        lines += ["", f"Arm (d) warm start: `{json.dumps(report['warm_start'])}`"]
    if report.get("notes"):
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report["notes"]]
    return "\n".join(lines) + "\n"


def write_report(raw: Dict, out_dir: Path) -> Path:
    """Writes ``raw.json``, ``report.json`` and ``report.md`` into ``out_dir``; returns the
    path of ``report.json``."""
    from benchmarks.common import save_json

    out_dir = Path(out_dir)
    save_json(out_dir / "raw.json", raw)
    report = build_report(json.loads((out_dir / "raw.json").read_text()))
    path = save_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(render_markdown(report))
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="benchmarks/out/<run> (with raw.json)")
    args = parser.parse_args(argv)
    raw = json.loads((args.run_dir / "raw.json").read_text())
    path = write_report(raw, args.run_dir)
    print(f"Wrote {path} and {path.with_name('report.md')}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
