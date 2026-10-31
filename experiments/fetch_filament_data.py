"""Download tomograms and segmentation masks (filaments, membranes) from the CryoET Data
Portal for Gate 6 / Gate 7. Resumable: complete files are not downloaded again.

Layout per dataset::

    <out>/<dataset id>/tomograms/<run>.mrc
    <out>/<dataset id>/masks/<object>/<run>.mrc      same shape and voxel size as the tomogram
    <out>/<dataset id>/manifest.json                 sources, ground-truth status, checks

Usage::

    python experiments/fetch_filament_data.py --out data/filaments [--datasets 10490,10477,10513] [--max-runs N]
"""
import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import mrcfile
import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dpd", REPO / "scripts" / "download_portal_data.py")
dpd = importlib.util.module_from_spec(_spec)
sys.modules["dpd"] = dpd  # dataclasses in the module need it registered
_spec.loader.exec_module(dpd)

# dataset -> (tomogram processing, {portal object name: folder name})
DATASETS = {
    10490: ("filtered", {"microtubule": "microtubule", "actin filament": "actin", "membrane": "membrane"}),
    10477: ("filtered", {"microtubule": "microtubule", "membrane": "membrane"}),
    10513: ("raw", {"microtubule": "microtubule", "plasma membrane": "membrane"}),
}
log = logging.getLogger("fetch_filament_data")


def fetch_dataset(cdp, client, ds: int, out: Path, max_runs=None) -> dict:
    processing, objects = DATASETS[ds]
    root = out / str(ds)
    runs = sorted(cdp.Run.find(client, [cdp.Run.dataset_id == ds]), key=lambda r: r.name)[:max_runs]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"dataset_id": ds, "runs": {}}
    for i, r in enumerate(runs):
        entry = manifest["runs"].get(r.name, {})
        if entry.get("complete"):
            continue
        masks = {}
        for a in cdp.Annotation.find(client, [cdp.Annotation.run_id == r.id]):
            folder = objects.get(a.object_name)
            if folder is None:
                continue
            for s in a.annotation_shapes:
                if s.shape_type != "SegmentationMask":
                    continue
                for f in s.annotation_files:
                    if f.format == "mrc" and f.tomogram_voxel_spacing is not None:
                        masks.setdefault(folder, (a, f))
        if not masks:
            log.warning("%s: no masks for %s, skipped", r.name, sorted(objects))
            continue
        spacing = next(iter(masks.values()))[1].tomogram_voxel_spacing.voxel_spacing
        tomos = [t for t in cdp.Tomogram.find(client, [cdp.Tomogram.run_id == r.id])
                 if abs(t.voxel_spacing - spacing) < 0.01 and t.processing == processing]
        if not tomos:
            log.warning("%s: no %s tomogram at %.3f A, skipped", r.name, processing, spacing)
            continue
        t = tomos[0]
        tomo_path = root / "tomograms" / f"{dpd.safe_name(r.name)}.mrc"
        dpd.download_file(t.https_mrc_file, tomo_path, expected_size=t.file_size_mrc)
        with mrcfile.mmap(tomo_path, permissive=True) as m:
            shape, apix = m.data.shape, float(m.voxel_size.x)
        entry = {"tomogram": str(tomo_path), "tomogram_id": t.id, "processing": t.processing,
                 "reconstruction": t.reconstruction_method, "apix_header": apix, "apix_portal": float(t.voxel_spacing),
                 "shape_zyx": list(shape), "masks": {}}
        for folder, (a, f) in sorted(masks.items()):
            mask_path = root / "masks" / folder / f"{dpd.safe_name(r.name)}.mrc"
            dpd.download_file(f.https_path, mask_path, expected_size=f.file_size)
            with mrcfile.mmap(mask_path, permissive=True) as m:
                mshape, mapix = m.data.shape, float(m.voxel_size.x)
                frac = float(np.count_nonzero(m.data[::4, ::4, ::4])) / max(m.data[::4, ::4, ::4].size, 1)
                values = np.unique(m.data[::8, ::8, ::8])[:5].tolist()
            entry["masks"][folder] = {"path": str(mask_path), "object_name": a.object_name,
                                      "ground_truth": bool(a.ground_truth_status), "annotation_id": a.id,
                                      "shape_zyx": list(mshape), "apix_header": mapix,
                                      "shape_matches": list(mshape) == list(shape),
                                      "foreground_fraction": round(frac, 5), "sample_values": values}
            if list(mshape) != list(shape):
                log.warning("%s/%s: mask shape %s != tomogram shape %s", r.name, folder, mshape, shape)
        entry["complete"] = True
        manifest["runs"][r.name] = entry
        root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1))
        log.info("DS %s: %d/%d %s done", ds, i + 1, len(runs), r.name)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--datasets", default="10477,10513,10490")
    p.add_argument("--max-runs", type=int, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import cryoet_data_portal as cdp

    client = cdp.Client()
    for ds in (int(d) for d in args.datasets.split(",")):
        m = fetch_dataset(cdp, client, ds, args.out, args.max_runs)
        runs = m["runs"].values()
        bad = [r for r in runs for k in r["masks"].values() if not k["shape_matches"]]
        log.info("DS %s finished: %d runs, %d mask(s) with a shape mismatch", ds, len(m["runs"]), len(bad))


if __name__ == "__main__":
    main()
