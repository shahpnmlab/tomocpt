"""Evaluate a CZII pipeline-v2 run: training scalars, stored calibration, predict on the
validation tomograms, per-class precision / recall / F1 / F4 and a confusion matrix.

Usage:
    python experiments/eval_czii_v2.py <run_dir> <label> [--skip-predict]

<run_dir> holds events.out.tfevents.* and checkpoints/weights.ckpt. Predictions go to
<run_dir>/pred_val/. The results are printed and appended to experiments/czii_v2_results.md.
"""

import glob
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import starfile
import torch
from scipy.spatial import cKDTree
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tomocpt.metrics import score_classes

CZII = Path(__file__).resolve().parents[1] / "data" / "czii"
NAMES = {1: "beta-amylase", 2: "beta-galactosidase", 3: "thyroglobulin",
         4: "ribosome", 5: "apo-ferritin", 6: "virus-like-particle"}
DIAM = {1: 130, 2: 180, 3: 260, 4: 300, 5: 120, 6: 270}
APIX = 10.012
VAL = ["TS_5_4", "TS_69_2"]
COORDS = CZII / "training_data_v2" / "dataset_1_class_all" / "coords"
LOG = Path(__file__).resolve().parent / "czii_v2_results.md"


def scalars(run_dir):
    ea = EventAccumulator(sorted(glob.glob(str(run_dir / "events.out.tfevents.*")))[-1],
                          size_guidance={"scalars": 0})
    ea.Reload()
    rows = []
    for t in sorted(t for t in ea.Tags()["scalars"] if t.startswith("val_loss")):
        v = [x.value for x in ea.Scalars(t)]
        rows.append(f"| {t} | {len(v)} | {v[0]:.4g} | {min(v):.4g} | {v[-1]:.4g} |")
    return ["| scalar | n | first | min | last |", "|---|---|---|---|---|", *rows]


def predict(run_dir, ckpt):
    out = run_dir / "pred_val"
    out.mkdir(exist_ok=True)
    cmd = ["tomocpt", "predict", "--config_file", str(CZII / "v2_config.yaml"),
           "--tomogram_dir", str(CZII / "val_tomos"), "--predictions_dir", str(out),
           "--weights", str(ckpt), "--predictions_coord_format", "relion_31",
           "--save_prediction_confidence_map"]
    with open(out / "predict.log", "w") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, cwd=CZII)
    return out


def score(pred_dir):
    picks = {c: starfile.read(pred_dir / f"tomocpt_coords_{n}.star")["particles"] for c, n in NAMES.items()}
    agg = {c: [0, 0, 0] for c in NAMES}
    conf = np.zeros((6, 7), int)
    for t in VAL:
        g = pd.read_csv(COORDS / f"{t}.csv")
        pred, true, radii = {}, {}, {}
        for c, n in NAMES.items():
            d = picks[c]
            d = d[d.rlnMicrographName.astype(str).str.contains(t)]
            pred[n] = d[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(float)
            true[n] = g[g.class_number == c][["x", "y", "z"]].to_numpy(float)
            radii[n] = DIAM[c] / 2 / APIX
        res = score_classes(pred, true, radii)
        for c, n in NAMES.items():
            pc = res["per_class"][n]
            agg[c][0] += pc["tp"]; agg[c][1] += pc["fp"]; agg[c][2] += pc["fn"]
        # each true particle -> class of the nearest pick within 0.5 radius (column 6 = none)
        trees = {c2: cKDTree(pred[n2]) for c2, n2 in NAMES.items() if len(pred[n2])}
        for c, n in NAMES.items():
            for x in true[n]:
                best = (np.inf, 6)
                for c2, tree in trees.items():
                    dist, _ = tree.query(x)
                    if dist <= 0.5 * radii[n] and dist < best[0]:
                        best = (dist, c2 - 1)
                conf[c - 1, best[1]] += 1
    rows = ["| class | TP | FP | FN | P | R | F1 | F4 |", "|---|---|---|---|---|---|---|---|"]
    f4s = []
    for c, (tp, fp, fn) in agg.items():
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        f4 = 17 * p * r / max(16 * p + r, 1e-9)
        f4s.append(f4)
        rows.append(f"| {NAMES[c]} | {tp} | {fp} | {fn} | {p:.2f} | {r:.2f} | {f1:.2f} | {f4:.2f} |")
    rows.append(f"| **mean** | | | | | | | **{np.mean(f4s):.2f}** |")
    short = ["amy", "gal", "thy", "ribo", "ferr", "vlp", "none"]
    cm = ["| true \\ picked | " + " | ".join(short) + " |", "|---" * 8 + "|"]
    cm += [f"| {short[i]} | " + " | ".join(str(v) for v in conf[i]) + " |" for i in range(6)]
    return rows, cm


def main():
    run_dir, label = Path(sys.argv[1]).resolve(), sys.argv[2]
    ckpt = run_dir / "checkpoints" / "weights.ckpt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["hyper_parameters"]["config"]
    k = getattr(getattr(cfg.train, "particles", None), "cross_class_neg_weight", 0.0)
    calib = (ck.get("tomocpt_calibration") or {}).get("classes", {})
    pred_dir = run_dir / "pred_val" if "--skip-predict" in sys.argv else predict(run_dir, ckpt)
    rows, cm = score(pred_dir)
    supp = [l.split(" - INFO - ")[-1].strip() for l in open(pred_dir / "predict.log")
            if "all tomograms" in l]
    out = [f"\n## {label} ({datetime.now():%Y-%m-%d %H:%M})", "",
           f"- run: `{run_dir}`; best epoch {ck.get('epoch')}; cross_class_neg_weight = {k}",
           f"- validation tomograms: {', '.join(VAL)} (used for early stopping and calibration)",
           "- calibrated thresholds: " + ", ".join(
               f"{NAMES.get(c, c)} {v['threshold']}" for c, v in calib.items()), "",
           *scalars(run_dir), "", *rows, "", *cm, "", "Cross-class suppression (all tomograms):", "",
           *([f"- {s}" for s in supp] or ["- none logged"]), ""]
    text = "\n".join(out)
    print(text)
    if not LOG.exists():
        LOG.write_text("# CZII 10440 pipeline-v2 experiments\n\nAppended by `experiments/eval_czii_v2.py`. "
                       "F-scores use `tomocpt.metrics` (match radius 0.5 × particle radius).\n")
    with open(LOG, "a") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
