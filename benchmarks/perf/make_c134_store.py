"""Synthetic 134-class tomogram store for G8c-C9: the CZII fold-0 volumes (symlinked) with random
points of 134 particle classes (d_vox 10-32, ~3000 per tomogram)."""
import json, os, sys
from pathlib import Path
import numpy as np

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
C = 134
rng = np.random.default_rng(0)
lay = json.load(open(src / "chunk_layout.json"))
diam = rng.uniform(100, 320, C)
lay["class_table"] = ([{"number": i + 1, "name": f"c{i}", "diameter_ang": float(d), "kind": "particle"}
                          for i, d in enumerate(diam)])
dst.mkdir(parents=True, exist_ok=True)
json.dump(lay, open(dst / "chunk_layout.json", "w"), indent=1)
(dst / "done_labels_True.txt").write_text("done\n")
for split in ("train", "val"):
    for d in sorted((src / split).iterdir()):
        o = dst / split / d.name
        o.mkdir(parents=True, exist_ok=True)
        if not (o / "volume.npy").exists():
            os.symlink(d / "volume.npy", o / "volume.npy")
        sm = json.load(open(d / "store_meta.json")); sm["drop_p"] = 0.0
        json.dump(sm, open(o / "store_meta.json", "w"))
        cm = json.load(open(d / "chunk_meta.json"))
        apix = cm["apix_work"]
        cm["class_available"] = [True] * C
        cm["class_d_vox"] = [float(x / apix) for x in diam]
        json.dump(cm, open(o / "chunk_meta.json", "w"), indent=1)
        shp = sm["shape"]; n = 3000
        ci = rng.integers(0, C, n)
        pts = np.stack([rng.uniform(0, shp[0], n), rng.uniform(0, shp[1], n), rng.uniform(0, shp[2], n),
                        ci, np.asarray(cm["class_d_vox"])[ci]], 1).astype(np.float32)
        np.save(o / "points.npy", pts)
print("ok", dst)
