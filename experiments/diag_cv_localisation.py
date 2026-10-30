"""Localisation diagnostics of a k-fold CV run (Gate 4b output): are particles found but
centred poorly, or missed? Each fold model picks on its own validation tomograms (the Gate 4b
calibration path, lowest grid threshold); picks are kept at the ensemble's OOF thresholds.

Per class, pooled over folds:
* F4 with the match radius at 0.5x (the CZII metric) and 1.0x the particle radius;
* centring error of picks matched at 1.0x radius (Å);
* misses at 1.0x radius that touch another true particle (centres closer than the sum of the
  radii x 1.1), compared with the touching fraction of all true particles (merging test).

Usage:
    python experiments/diag_cv_localisation.py <cv_dir> [--label TEXT]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from tomocpt.defaultConfigs.infer_config import InferConfig
from tomocpt.networks.inference import InferenceModel
from tomocpt.predict.calibration import collect_validation_picks

LOG = Path(__file__).resolve().parent / "czii_v2_results.md"


def greedy_match(xyz, score, true, rad):
    """One-to-one matching by descending score; returns (tp mask over picks, matched truth
    index per pick or -1, distance per pick)."""
    order = np.argsort(-score)
    tp = np.zeros(len(xyz), bool)
    idx = np.full(len(xyz), -1)
    dist = np.full(len(xyz), np.inf)
    if len(true) and len(xyz):
        tree, used = cKDTree(true), set()
        for i in order:
            for j in sorted(tree.query_ball_point(xyz[i], rad), key=lambda j: np.linalg.norm(true[j] - xyz[i])):
                if j not in used:
                    used.add(j)
                    tp[i], idx[i], dist[i] = True, j, np.linalg.norm(true[j] - xyz[i])
                    break
    return tp, idx, dist


def f4(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 17 * p * r / max(16 * p + r, 1e-9)


def main():
    cv = Path(sys.argv[1]).resolve()
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else cv.name
    ens = json.load(open(cv / "ensemble.json"))
    thr = {int(k): v["threshold"] for k, v in ens["calibration"]["classes"].items()}
    cfg = InferConfig()
    cfg.use_cuda = torch.cuda.is_available()
    cfg.save_prediction_confidence_map = False
    cfg.invert_contrast = False
    device = torch.device("cuda" if cfg.use_cuda else "cpu")

    stats, table = {}, None
    for fold in sorted(cv.glob("fold*")):
        ckpt = sorted(glob.glob(str(fold / "model" / "**" / "*.ckpt"), recursive=True))[-1]
        val = json.load(open(fold / "chunks" / "chunk_layout.json"))["val_tomograms"]
        model = InferenceModel.load_from_checkpoint(ckpt, map_location=device, strict=True).eval()
        table = model.class_table()
        picks, truths, radii, _ = collect_validation_picks(model, val, cfg, device)
        del model
        torch.cuda.empty_cache()
        for t in range(len(val)):
            all_true = [(truths[r.number][t], radii[r.number]) for r in table if truths[r.number][t] is not None]
            for row in table:
                c = row.number
                true = truths[c][t]
                if true is None:
                    continue
                xyz, score = picks[c][t]
                keep = score >= (thr[c] if thr[c] is not None else 0.5)
                xyz, score = xyz[keep], score[keep]
                s = stats.setdefault(c, {"n_true": 0, "half": [0, 0, 0], "full": [0, 0, 0], "dist": [],
                                         "touch_all": 0, "touch_fn": 0, "fn": 0})
                R = radii[c]
                s["n_true"] += len(true)
                for key, frac in (("half", 0.5), ("full", 1.0)):
                    tp, idx, dist = greedy_match(xyz, score, true, frac * R)
                    s[key][0] += int(tp.sum()); s[key][1] += int((~tp).sum()); s[key][2] += len(true) - int(tp.sum())
                    if key == "full":
                        s["dist"] += list(dist[tp])
                        found = set(idx[tp].tolist())
                # touching: another true particle (any class) closer than (R_a + R_b) * 1.1
                touching = np.zeros(len(true), bool)
                for other, R2 in all_true:
                    if not len(other):
                        continue
                    d, _ = cKDTree(other).query(true, k=2 if other is true else 1)
                    d = d[:, 1] if other is true else d
                    touching |= d < 1.1 * (R + R2)
                missed = np.array([j not in found for j in range(len(true))], bool)
                s["touch_all"] += int(touching.sum()); s["touch_fn"] += int((touching & missed).sum())
                s["fn"] += int(missed.sum())

    apix = float(json.load(open(sorted(cv.glob("fold*"))[0] / "chunks" / "chunk_layout.json"))
                 ["val_tomograms"][0].get("apix_native") or 1.0)
    names = {r.number: r.name for r in table}
    lines = [f"\n## Localisation diagnostics: {label}\n",
             f"- `{cv}`; OOF picks at the ensemble thresholds; distances in Å (native {apix:.3f} Å/px)\n",
             "| class | n true | F4 @0.5R | F4 @1.0R | recall @0.5R | recall @1.0R | centring error median / p90 (Å) "
             "| radius (Å) | misses touching / all touching |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c, s in sorted(stats.items()):
        h, f = s["half"], s["full"]
        d = np.asarray(s["dist"]) * apix
        R = next(r.diameter_ang for r in table if r.number == c) / 2
        med, p90 = (np.median(d), np.percentile(d, 90)) if len(d) else (np.nan, np.nan)
        lines.append(f"| {names[c]} | {s['n_true']} | {f4(*h):.3f} | {f4(*f):.3f} | {h[0] / max(s['n_true'], 1):.2f} "
                     f"| {f[0] / max(s['n_true'], 1):.2f} | {med:.0f} / {p90:.0f} | {R:.0f} "
                     f"| {s['touch_fn'] / max(s['fn'], 1):.2f} / {s['touch_all'] / max(s['n_true'], 1):.2f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(LOG, "a") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
