"""Diagnose G6-C8/C9: per-class counts, class confusion, and distance of misses to tubes (joint model)."""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
REPO = Path("/home/pranav/Desktop/pranav/software/tomocpt")
sys.path[:0] = [str(REPO / "tests"), str(REPO)]
from gate6 import test_g6_learning as L
from gate6 import g6_helpers as H
from tomocpt.metrics import match_points

def diag(model, held, table):
    num2d = {n: d for n, _, d in L.PARTICLES}
    out = {"per_class": {}, "confusion": {}, "miss_dist_to_tube_vox": [], "hit_dist_to_tube_vox": [],
           "score_at_true": {}}
    for ph in held:
        pred = L.predict(model, L.normalise(ph.volume))
        peaks = L.particle_peaks(pred, table, L.PICK_THRESHOLD, True)
        tubes = [p for n, _, _ in L.FILAMENTS for p in ph.polylines.get(n, [])]
        allp = {n: pk[:, :3] for n, pk in peaks.items()}
        for c, (n, name, d) in enumerate(table):
            if n not in num2d:
                continue
            t = np.asarray(ph.coords.get(n, np.zeros((0, 3)))).reshape(-1, 3)[:, ::-1].astype(float)  # z,y,x
            a, b, e = match_points(allp[n], t, d / 10 / 4)
            s = out["per_class"].setdefault(name, [0, 0, 0])
            out["per_class"][name] = [s[0] + len(a), s[1] + len(b), s[2] + len(e)]
            # channel values at true centres
            zi = np.round(t).astype(int)
            for c2, (n2, name2, _) in enumerate(table):
                if n2 in num2d:
                    v = pred[c2].numpy()[zi[:, 0], zi[:, 1], zi[:, 2]]
                    out["score_at_true"].setdefault(f"true {name} -> ch {name2}", []).extend(v.tolist())
            # where do the misses go?
            e = np.asarray(e, int).reshape(-1)
            for i in range(len(t)):
                miss = i in set(e.tolist())
                lab = "missed" if miss else "found"
                if miss:
                    other = [n2 for n2 in allp if n2 != n and len(allp[n2]) and
                             np.min(np.linalg.norm(allp[n2] - t[i], axis=1)) <= num2d[n] / 10 / 4]
                    key = f"{name} missed -> picked as {[x for x in other] or 'nothing'}"
                    out["confusion"][key] = out["confusion"].get(key, 0) + 1
                if tubes:
                    dt = min(float(H.point_to_polyline_distance(t[i:i + 1, ::-1], p)[0]) for p in tubes)
                    out["miss_dist_to_tube_vox" if miss else "hit_dist_to_tube_vox"].append(dt)
    for k in list(out["score_at_true"]):
        v = np.array(out["score_at_true"][k]); out["score_at_true"][k] = [round(float(np.median(v)), 3), round(float(v.min()), 3)]
    for k in ("miss_dist_to_tube_vox", "hit_dist_to_tube_vox"):
        v = np.array(out[k]); out[k] = {"n": len(v), "median": float(np.median(v)) if len(v) else None,
                                         "frac_lt_R_tube(12.5)": float((v < 12.5).mean()) if len(v) else None}
    return out

