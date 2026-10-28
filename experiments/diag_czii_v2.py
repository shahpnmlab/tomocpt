"""Threshold-free diagnostics for CZII pipeline-v2 runs, from the saved per-class maps.

1. Per class: average precision (AP), best F4 over all thresholds, and recall at a 0.01
   threshold, so ranking quality is separated from calibration.
2. Where the false picks at the calibrated threshold are: on a particle of another class,
   next to an already-found particle of the same class, near the tomogram border, or in
   background. Local contrast of false picks is compared with true picks and random points.
3. A montage of the top-scoring false picks next to true picks (central XY slices).

Usage:
    python experiments/diag_czii_v2.py <label>::<pred_dir>::<ckpt> [...] [--montage <png>]
"""

import sys
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from tomocpt.defaultConfigs.infer_config import InferConfig
from tomocpt.predict.multiclass import peaks_from_class_maps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_czii_v2 import APIX, COORDS, CZII, DIAM, NAMES, VAL  # noqa: E402

LOG = Path(__file__).resolve().parent / "czii_v2_results.md"
ROWS = {c - 1: c for c in NAMES}  # class-table row -> class number


class _Spec:
    def __init__(self, c):
        self.name, self.diameter_ang = NAMES[c], DIAM[c]


TABLE = [_Spec(ROWS[r]) for r in range(6)]


def load_tomo(t):
    v = mrcfile.read(CZII / "val_tomos" / f"{t}.mrc").astype(np.float32)
    p5, p50, p95 = np.percentile(v[::4, ::4, ::4], [5, 50, 95])
    return np.clip((v - p50) / (p95 - p5), -3, 3)


def peaks(pred_dir, t):
    maps = {r: torch.from_numpy(mrcfile.read(pred_dir / f"{t}_{NAMES[c]}.mrc").astype(np.float32))
            for r, c in ROWS.items()}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p, _ = peaks_from_class_maps(maps, TABLE, APIX, InferConfig(), {r: 0.01 for r in ROWS}, t, dev)
    return {ROWS[r]: v for r, v in p.items()}  # z, y, x, score


def match(pk, true, rad):
    """Greedy one-to-one matching by descending score; returns is_tp per peak (score order)."""
    order = np.argsort(-pk[:, 3])
    pk = pk[order]
    tp = np.zeros(len(pk), bool)
    if len(true) and len(pk):
        tree, used = cKDTree(true), set()
        for i, x in enumerate(pk[:, :3]):
            for j in tree.query_ball_point(x, rad):
                if j not in used:
                    used.add(j)
                    tp[i] = True
                    break
    return pk, tp


def ap_f4(scores, tp, n_true):
    o = np.argsort(-scores)
    tp = tp[o]
    ctp = np.cumsum(tp)
    prec, rec = ctp / np.arange(1, len(tp) + 1), ctp / max(n_true, 1)
    ap = float(np.sum(np.diff(np.r_[0, rec]) * np.maximum.accumulate(prec[::-1])[::-1])) if len(tp) else 0.0
    f4 = 17 * prec * rec / np.maximum(16 * prec + rec, 1e-9)
    i = int(np.argmax(f4)) if len(f4) else 0
    return ap, float(f4[i]) if len(f4) else 0.0, float(scores[o][i]) if len(f4) else np.nan, float(rec[-1]) if len(rec) else 0.0


def contrast(vol, pts, r):
    """Mean in the central cube (half-width r/2) minus mean of the surrounding shell (r..1.5r)."""
    out = []
    a, b = max(1, int(r / 2)), int(1.5 * r)
    for z, y, x in np.round(pts).astype(int):
        sl = tuple(slice(max(q - b, 0), q + b + 1) for q in (z, y, x))
        box = vol[sl]
        c = tuple(q - s.start for q, s in zip((z, y, x), sl))
        g = np.ogrid[tuple(slice(0, n) for n in box.shape)]
        d = np.abs(g[0] - c[0])
        for gi, ci in zip(g[1:], c[1:]):
            d = np.maximum(d, np.abs(gi - ci))
        out.append(box[d <= a].mean() - box[(d > r) & (d <= b)].mean())
    return np.asarray(out)


def run(label, pred_dir, ckpt):
    calib = torch.load(ckpt, map_location="cpu", weights_only=False).get("tomocpt_calibration", {})["classes"]
    thr = {c: calib[c]["threshold"] if c in calib else calib[str(c)]["threshold"] for c in NAMES}
    per = {c: {"s": [], "tp": [], "n": 0} for c in NAMES}
    fp_cat = {c: dict(other=0, dup=0, border=0, background=0) for c in NAMES}
    con = {c: {"tp": [], "fp": [], "rand": []} for c in NAMES}
    top_fp, top_tp = {c: [] for c in NAMES}, {c: [] for c in NAMES}
    rng = np.random.default_rng(0)
    for t in VAL:
        g = pd.read_csv(COORDS / f"{t}.csv")
        true = {c: g[g.class_number == c][["z", "y", "x"]].to_numpy(float) for c in NAMES}
        vol = load_tomo(t)
        pk = peaks(pred_dir, t)
        for c in NAMES:
            rad = DIAM[c] / 2 / APIX
            p, tp = match(pk[c], true[c], 0.5 * rad)
            per[c]["s"].append(p[:, 3]); per[c]["tp"].append(tp); per[c]["n"] += len(true[c])
            keep = p[:, 3] >= thr[c]
            fps, tps = p[keep & ~tp, :3], p[keep & tp, :3]
            for x in fps:
                if any(len(true[o]) and cKDTree(true[o]).query(x)[0] <= DIAM[o] / 2 / APIX for o in NAMES if o != c):
                    fp_cat[c]["other"] += 1
                elif len(true[c]) and cKDTree(true[c]).query(x)[0] <= 2 * rad:
                    fp_cat[c]["dup"] += 1
                elif np.min(np.r_[x, np.array(vol.shape) - 1 - x]) < 2 * rad:
                    fp_cat[c]["border"] += 1
                else:
                    fp_cat[c]["background"] += 1
            m = 2 * int(1.5 * rad) + 1
            rnd = rng.uniform(m, np.array(vol.shape) - m, size=(200, 3))
            con[c]["tp"] += list(contrast(vol, tps, rad)); con[c]["fp"] += list(contrast(vol, fps, rad))
            con[c]["rand"] += list(contrast(vol, rnd, rad))
            sc = p[keep & ~tp, 3]
            top_fp[c] += [(s, t, x) for s, x in zip(sc, fps)]
            top_tp[c] += [(s, t, x) for s, x in zip(p[keep & tp, 3], tps)]
    rows = []
    for c in NAMES:
        s, tp = np.concatenate(per[c]["s"]), np.concatenate(per[c]["tp"])
        ap, f4, best_t, rec = ap_f4(s, tp, per[c]["n"])
        q = lambda v: f"{np.median(v):+.2f}" if len(v) else "n/a"  # noqa: E731
        rows.append(dict(label=label, c=c, ap=ap, f4=f4, best_t=best_t, rec=rec, thr=thr[c], **fp_cat[c],
                         con_tp=q(con[c]["tp"]), con_fp=q(con[c]["fp"]), con_rand=q(con[c]["rand"])))
    return rows, top_fp, top_tp


def montage(path, label, top_fp, top_tp, classes=(1, 2, 3), n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vols = {t: load_tomo(t) for t in VAL}
    fig, ax = plt.subplots(2 * len(classes), n, figsize=(1.6 * n, 3.4 * len(classes)))
    for i, c in enumerate(classes):
        h = int(1.5 * DIAM[c] / APIX)
        for j, (kind, lst) in enumerate((("TP", top_tp[c]), ("FP", top_fp[c]))):
            lst = sorted(lst, key=lambda e: -e[0])[:n]
            for k in range(n):
                a = ax[2 * i + j, k]
                a.axis("off")
                if k >= len(lst):
                    continue
                s, t, (z, y, x) = lst[k]
                z, y, x = (int(round(v)) for v in (z, y, x))
                v = vols[t]
                img = v[max(z - 1, 0):z + 2, max(y - h, 0):y + h, max(x - h, 0):x + h].mean(0)
                a.imshow(img, cmap="gray", vmin=-1.5, vmax=1.5)
                a.set_title(f"{NAMES[c][:6]} {kind} {s:.2f}", fontsize=7)
    fig.suptitle(f"{label}: top true vs false picks (3-slice XY average, box 3 diameters)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    args = [a for a in sys.argv[1:] if "::" in a]
    png = sys.argv[sys.argv.index("--montage") + 1] if "--montage" in sys.argv else None
    allrows, first = [], None
    for a in args:
        label, pred, ckpt = a.split("::")
        rows, fp, tp = run(label, Path(pred), Path(ckpt))
        allrows += rows
        first = first or (label, fp, tp)
    df = pd.DataFrame(allrows)
    lines = ["| run | class | AP | best F4 | at threshold | calibrated | recall at 0.01 | FP on other class "
             "| FP duplicate | FP border | FP background | contrast TP / FP / random |",
             "|---" * 12 + "|"]
    for r in df.itertuples():
        lines.append(f"| {r.label} | {NAMES[r.c]} | {r.ap:.2f} | {r.f4:.2f} | {r.best_t:.2f} | {r.thr:.2f} "
                     f"| {r.rec:.2f} | {r.other} | {r.dup} | {r.border} | {r.background} "
                     f"| {r.con_tp} / {r.con_fp} / {r.con_rand} |")
    for label in df.label.unique():
        d = df[df.label == label]
        lines.append(f"| {label} | **mean** | **{d.ap.mean():.2f}** | **{d.f4.mean():.2f}** | | | | "
                     f"{d.other.sum()} | {d.dup.sum()} | {d.border.sum()} | {d.background.sum()} | |")
    text = "\n".join(lines)
    print(text)
    if png:
        montage(png, *first)
    return text


if __name__ == "__main__":
    main()
