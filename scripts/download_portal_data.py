#!/usr/bin/env python
"""Download tomograms and particle picks from the CryoET Data Portal for tomocpt.

The script downloads one tomogram per run at a chosen voxel spacing, converts the
portal's point annotations into RELION 3.1 STAR files that ``tomocpt prepare_data``
reads, and writes a ``prepData`` snippet for ``config.yaml``.

Example (CZII object-identification training data, 10.012 A/px)::

    pip install -e '.[portal]'
    python scripts/download_portal_data.py --dataset-id 10440 \
        --out data/czii_train --voxel-spacing 10.012 --processing denoised

    python scripts/download_portal_data.py --dataset-id 10440 --out data/czii_train \
        --list            # show runs, tomograms and annotated objects; download nothing

Output layout::

    <out>/tomograms/<run>.mrc
    <out>/coords/<object>.star        one file per object, class number 1
    <out>/coords/all_objects.star     every object, one rlnClassNumber per object
    <out>/coords/raw/<run>/<object>.ndjson
    <out>/manifest.json               what was downloaded, from where, and how
    <out>/prepdata_snippet.yaml       paste into config.yaml (prepData section)

Point locations on the portal are in voxels of the voxel spacing that the
annotation belongs to. The script rescales them to the downloaded tomogram if the
two spacings differ, and stops with an error if the points do not fit inside the
tomogram (the usual sign of a unit or alignment mismatch).

CZII challenge datasets: 10440 (training), 10445 (public test), 10446 (private
test). The portal corrected the handedness of these tomograms after the challenge,
so coordinates from the challenge era may not match the current files.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("download_portal_data")

POINT_SHAPES = ("Point", "OrientedPoint")


# ---------------------------------------------------------------------------
# Pure helpers (no network, no portal client) - covered by tests
# ---------------------------------------------------------------------------


def safe_name(name: str) -> str:
    """Return a file-system and STAR-safe version of an object or run name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return cleaned.strip("_") or "unnamed"


def parse_ndjson_points(lines: Iterable[str]) -> np.ndarray:
    """Read portal ndjson point records into an ``(n, 3)`` array of ``x, y, z``.

    Records without a ``location`` are skipped. Empty input gives a ``(0, 3)`` array.
    """
    points = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        record = json.loads(raw)
        loc = record.get("location")
        if loc is None:
            continue
        points.append((float(loc["x"]), float(loc["y"]), float(loc["z"])))
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def rescale_points(points_xyz: np.ndarray, from_spacing: float, to_spacing: float) -> np.ndarray:
    """Convert voxel coordinates from one voxel spacing to another."""
    if from_spacing <= 0 or to_spacing <= 0:
        raise ValueError(f"voxel spacings must be positive, got {from_spacing} and {to_spacing}")
    return points_xyz * (from_spacing / to_spacing)


def check_points_inside(points_xyz: np.ndarray, size_xyz: Sequence[int], margin: float = 0.5) -> int:
    """Return the number of points outside ``[-margin, size - 1 + margin]`` on any axis."""
    if len(points_xyz) == 0:
        return 0
    size = np.asarray(size_xyz, dtype=np.float64)
    outside = (points_xyz < -margin) | (points_xyz > size - 1 + margin)
    return int(outside.any(axis=1).sum())


def particles_table(
    points_by_tomo: Dict[str, np.ndarray], class_number: int = 1
) -> pd.DataFrame:
    """Build a RELION 3.1 ``data_particles`` table from points per tomogram."""
    rows = []
    for tomo, pts in sorted(points_by_tomo.items()):
        for x, y, z in np.asarray(pts).reshape(-1, 3):
            rows.append((tomo, x, y, z, class_number))
    df = pd.DataFrame(
        rows,
        columns=[
            "rlnMicrographName",
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnClassNumber",
        ],
    )
    df["rlnOpticsGroup"] = 1
    return df


def optics_table(pixel_size: float) -> pd.DataFrame:
    """RELION 3.1 ``data_optics`` table; tomocpt reads ``rlnImagePixelSize`` from it."""
    return pd.DataFrame(
        {"rlnOpticsGroup": [1], "rlnOpticsGroupName": ["opticsGroup1"], "rlnImagePixelSize": [pixel_size]}
    )


def write_star(path: Path, particles: pd.DataFrame, pixel_size: float) -> None:
    """Write a RELION 3.1 STAR file with optics and particles blocks."""
    import starfile  # tomocpt dependency

    path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write({"optics": optics_table(pixel_size), "particles": particles}, path, overwrite=True)


def assign_class_numbers(object_names: Iterable[str]) -> Dict[str, int]:
    """Give each object name a stable class number, starting at 1, in sorted order."""
    return {name: i + 1 for i, name in enumerate(sorted(set(object_names)))}


def prepdata_snippet(
    tomogram_dir: Path,
    star_paths: Dict[str, Path],
    diameters: Dict[str, float],
    training_data_dir: Path,
) -> str:
    """Return a ``prepData`` YAML block with one dataset per object.

    Objects without a known diameter get ``???`` (OmegaConf's MISSING), so
    ``prepare_data`` refuses to run until you fill them in.
    """
    names = sorted(star_paths)
    lengths = [str(diameters[n]) if n in diameters else "???" for n in names]
    def join(values):
        return ",".join(str(v) for v in values)
    lines = [
        "prepData:",
        f"  # objects, in order: {join(names)}",
        f"  raw_data_dir: {join([tomogram_dir] * len(names))}",
        f"  training_data_dir: {training_data_dir}",
        f"  coordinate_files: {join([star_paths[n] for n in names])}",
        f"  particle_length_ang: {join(lengths)}",
        "  coordinate_file_type: star",
        f"  class_id: {join(['all'] * len(names))}",
    ]
    if "???" in lengths:
        lines.insert(1, "  # Fill in particle_length_ang (diameter in A) where it says ???")
    return "\n".join(lines) + "\n"


def parse_diameters(text: Optional[str]) -> Dict[str, float]:
    """Parse ``name=diameter,name=diameter`` into a dict (names made safe)."""
    out: Dict[str, float] = {}
    if not text:
        return out
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"bad --diameters item {item!r}; use name=angstrom")
        name, value = item.split("=", 1)
        out[safe_name(name)] = float(value)
    return out


@dataclass
class TomoCandidate:
    """The fields of a portal tomogram that selection needs."""

    id: int
    voxel_spacing: float
    processing: str
    reconstruction_method: str
    is_portal_standard: bool = False
    is_visualization_default: bool = False
    extra: dict = field(default_factory=dict)


def choose_tomogram(
    candidates: Sequence[TomoCandidate],
    voxel_spacing: Optional[float],
    tolerance: float,
    processing: Optional[str],
    reconstruction_method: Optional[str],
) -> Optional[TomoCandidate]:
    """Pick one tomogram per run.

    Filters by voxel spacing (within ``tolerance`` A), processing and
    reconstruction method (case-insensitive). Among what remains, prefers the
    portal-standard tomogram, then the visualization default, then the smallest
    id (stable across calls).
    """
    pool = list(candidates)
    if voxel_spacing is not None:
        pool = [c for c in pool if abs(c.voxel_spacing - voxel_spacing) <= tolerance]
    if processing:
        pool = [c for c in pool if (c.processing or "").lower() == processing.lower()]
    if reconstruction_method:
        pool = [c for c in pool if (c.reconstruction_method or "").lower() == reconstruction_method.lower()]
    if not pool:
        return None
    pool.sort(key=lambda c: (not c.is_portal_standard, not c.is_visualization_default, c.id))
    return pool[0]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def download_file(
    url: str,
    dest: Path,
    expected_size: Optional[float] = None,
    retries: int = 5,
    chunk: int = 1 << 20,
) -> Path:
    """Download ``url`` to ``dest`` with resume, retries, size check and atomic rename.

    A complete file of the expected size is not downloaded again.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = int(expected_size) if expected_size else None
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        log.info("exists, skipping: %s", dest)
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        if expected is not None and start > expected:
            part.unlink()  # stale or corrupt partial file: start again
            start = 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as resp:
                if resp.status_code == 416:  # nothing left to send for this range
                    if expected is None or start == expected:
                        break
                    part.unlink(missing_ok=True)
                    raise IOError("server rejected the resume range; restarting")
                resp.raise_for_status()
                mode = "ab" if (start and resp.status_code == 206) else "wb"
                with open(part, mode) as fh:
                    for block in resp.iter_content(chunk):
                        fh.write(block)
            size = part.stat().st_size
            if expected is not None and size != expected:
                raise IOError(f"size {size} != expected {expected}")
            break
        except Exception as exc:  # network errors, HTTP errors, short files
            wait = min(60, 2**attempt)
            log.warning("download failed (%s/%s) for %s: %s; retrying in %ss", attempt, retries, url, exc, wait)
            if attempt == retries:
                raise
            time.sleep(wait)
    part.replace(dest)
    return dest


def check_mrc_voxel_size(path: Path, expected: float, fix: bool) -> float:
    """Compare the MRC header voxel size with the portal value; optionally fix it.

    tomocpt reads the voxel size from the header, so a header of 0 or 1 A would
    silently break the particle size in voxels.
    """
    import mrcfile

    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        header = float(mrc.voxel_size.x)
    if abs(header - expected) <= 0.01 * expected:
        return header
    msg = f"{path.name}: header voxel size {header:.4f} A, portal says {expected:.4f} A"
    if not fix:
        log.warning("%s (use --fix-header to write the portal value)", msg)
        return header
    with mrcfile.open(path, mode="r+", permissive=True) as mrc:
        mrc.voxel_size = expected
    log.warning("%s; header fixed", msg)
    return expected


def _portal():
    try:
        import cryoet_data_portal as cdp
    except ImportError:  # pragma: no cover - depends on the environment
        sys.exit(
            "The CryoET Data Portal client is not installed.\n"
            "Install it with:  pip install -e '.[portal]'   (or: pip install cryoet-data-portal)"
        )
    return cdp


def _candidate(t) -> TomoCandidate:
    return TomoCandidate(
        id=int(t.id),
        voxel_spacing=float(t.voxel_spacing),
        processing=t.processing or "",
        reconstruction_method=t.reconstruction_method or "",
        is_portal_standard=bool(t.is_portal_standard),
        is_visualization_default=bool(t.is_visualization_default),
        extra={"obj": t},
    )


def _point_files(cdp, client, run_id: int, ground_truth_only: bool):
    """Yield ``(object_name, annotation, annotation_file, file_spacing)`` for point annotations."""
    annotations = cdp.Annotation.find(client, [cdp.Annotation.run_id == run_id])
    for ann in annotations:
        if ground_truth_only and not ann.ground_truth_status:
            continue
        for shape in ann.annotation_shapes:
            if shape.shape_type not in POINT_SHAPES:
                continue
            for f in shape.annotation_files:
                if f.format != "ndjson":
                    continue
                spacing = float(f.tomogram_voxel_spacing.voxel_spacing)
                yield safe_name(ann.object_name), ann, f, spacing


def _pick_one_annotation(entries: List[tuple], tomo_obj) -> Optional[tuple]:
    """Several annotations can exist for one object. Prefer same alignment, then
    ground truth, then curator-recommended, then the lowest id."""
    if not entries:
        return None
    def key(e):
        _, ann, f, _ = e
        return (
            f.alignment_id != tomo_obj.alignment_id,
            not ann.ground_truth_status,
            not ann.is_curator_recommended,
            int(ann.id),
        )
    return sorted(entries, key=key)[0]


def run(args: argparse.Namespace) -> int:
    cdp = _portal()
    client = cdp.Client()
    out = Path(args.out).resolve()
    tomo_dir = out / "tomograms"
    coord_dir = out / "coords"
    wanted = {safe_name(o) for o in args.objects.split(",")} if args.objects else None
    diameters = parse_diameters(args.diameters)

    runs = sorted(cdp.Run.find(client, [cdp.Run.dataset_id == args.dataset_id]), key=lambda r: r.name)
    if args.runs:
        keep = set(args.runs.split(","))
        runs = [r for r in runs if r.name in keep]
    if args.max_runs:
        runs = runs[: args.max_runs]
    if not runs:
        log.error("no runs found for dataset %s", args.dataset_id)
        return 1

    manifest = {
        "dataset_id": args.dataset_id,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "client_version": getattr(cdp, "__version__", "unknown"),
        "selection": {
            "voxel_spacing": args.voxel_spacing,
            "tolerance": args.tolerance,
            "processing": args.processing,
            "reconstruction_method": args.reconstruction_method,
            "ground_truth_only": args.ground_truth_only,
        },
        "runs": [],
    }
    points: Dict[str, Dict[str, np.ndarray]] = {}  # object -> tomo -> xyz (tomogram voxels)
    pixel_size: Optional[float] = None

    for r in runs:
        cands = [_candidate(t) for t in cdp.Tomogram.find(client, [cdp.Tomogram.run_id == r.id])]
        chosen = choose_tomogram(cands, args.voxel_spacing, args.tolerance, args.processing, args.reconstruction_method)
        if args.list:
            print(f"run {r.name} (id {r.id})")
            for c in cands:
                mark = "*" if chosen and c.id == chosen.id else " "
                print(f"  {mark} tomogram {c.id}: {c.voxel_spacing:.3f} A, processing={c.processing}, "
                      f"reconstruction={c.reconstruction_method}, standard={c.is_portal_standard}")
            objs = sorted({o for o, *_ in _point_files(cdp, client, r.id, args.ground_truth_only)})
            print(f"    point annotations: {', '.join(objs) or '(none)'}")
            continue
        if chosen is None:
            log.warning("run %s: no tomogram matches the selection; skipped (use --list)", r.name)
            continue
        t = chosen.extra["obj"]
        if pixel_size is None:
            pixel_size = chosen.voxel_spacing
        elif abs(pixel_size - chosen.voxel_spacing) > 1e-3:
            log.error("run %s: voxel spacing %.4f differs from %.4f; tomocpt needs one per STAR file",
                      r.name, chosen.voxel_spacing, pixel_size)
            return 1

        name = safe_name(r.name)
        mrc_path = tomo_dir / f"{name}.mrc"
        if not args.coords_only:
            download_file(t.https_mrc_file, mrc_path, expected_size=t.file_size_mrc)
            check_mrc_voxel_size(mrc_path, chosen.voxel_spacing, args.fix_header)
        size_xyz = (int(t.size_x), int(t.size_y), int(t.size_z))

        by_object: Dict[str, List[tuple]] = {}
        for entry in _point_files(cdp, client, r.id, args.ground_truth_only):
            if wanted is None or entry[0] in wanted:
                by_object.setdefault(entry[0], []).append(entry)

        run_record = {"run": r.name, "run_id": int(r.id), "tomogram_id": chosen.id,
                      "mrc": str(mrc_path), "url": t.https_mrc_file, "size_xyz": size_xyz, "objects": {}}
        for obj, entries in sorted(by_object.items()):
            obj_name, ann, f, file_spacing = _pick_one_annotation(entries, t)
            raw_path = coord_dir / "raw" / name / f"{obj_name}.ndjson"
            download_file(f.https_path, raw_path, expected_size=f.file_size)
            pts = parse_ndjson_points(raw_path.read_text().splitlines())
            pts = rescale_points(pts, file_spacing, chosen.voxel_spacing)
            n_out = check_points_inside(pts, size_xyz)
            if n_out:
                log.error("run %s, %s: %d of %d points fall outside the tomogram %s; check units/alignment "
                          "(annotation %s, alignment %s vs tomogram alignment %s)", r.name, obj_name, n_out,
                          len(pts), size_xyz, ann.id, f.alignment_id, t.alignment_id)
                if not args.allow_outside:
                    return 1
            if f.alignment_id != t.alignment_id:
                log.warning("run %s, %s: annotation alignment %s differs from tomogram alignment %s",
                            r.name, obj_name, f.alignment_id, t.alignment_id)
            points.setdefault(obj_name, {})[name] = pts
            run_record["objects"][obj_name] = {
                "annotation_id": int(ann.id), "annotation_file_id": int(f.id), "n_points": int(len(pts)),
                "annotation_voxel_spacing": file_spacing, "ground_truth": bool(ann.ground_truth_status),
                "method_type": ann.method_type, "points_outside": n_out,
            }
        manifest["runs"].append(run_record)
        log.info("run %s: tomogram %s, %s", r.name, chosen.id,
                 ", ".join(f"{k}={v['n_points']}" for k, v in run_record["objects"].items()) or "no points")

    if args.list:
        return 0
    if not points:
        log.error("no point annotations were found for the selection")
        return 1

    classes = assign_class_numbers(points)
    star_paths: Dict[str, Path] = {}
    all_tables = []
    for obj, per_tomo in sorted(points.items()):
        table = particles_table(per_tomo, class_number=1)
        star_paths[obj] = coord_dir / f"{obj}.star"
        write_star(star_paths[obj], table, pixel_size)
        all_tables.append(particles_table(per_tomo, class_number=classes[obj]))
    write_star(coord_dir / "all_objects.star", pd.concat(all_tables, ignore_index=True), pixel_size)

    manifest["pixel_size"] = pixel_size
    manifest["class_numbers"] = classes
    manifest["star_files"] = {k: str(v) for k, v in star_paths.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    snippet = prepdata_snippet(tomo_dir, star_paths, diameters, out / "training_data")
    (out / "prepdata_snippet.yaml").write_text(snippet)
    print(snippet)
    log.info("done: %d runs, objects %s; manifest at %s", len(manifest["runs"]), sorted(points), out / "manifest.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-id", type=int, required=True, help="Portal dataset id, e.g. 10440.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--voxel-spacing", type=float, default=None, help="Tomogram voxel spacing in A, e.g. 10.012.")
    p.add_argument("--tolerance", type=float, default=0.05, help="Voxel-spacing tolerance in A (default 0.05).")
    p.add_argument("--processing", default=None, help="Tomogram processing, e.g. raw, denoised, filtered.")
    p.add_argument("--reconstruction-method", default=None, help="e.g. WBP, SART, SIRT.")
    p.add_argument("--objects", default=None, help="Comma-separated object names to keep (default: all).")
    p.add_argument("--runs", default=None, help="Comma-separated run names to keep (default: all).")
    p.add_argument("--max-runs", type=int, default=None, help="Download at most this many runs.")
    p.add_argument("--diameters", default=None, help="name=diameter_A,... for the prepData snippet.")
    p.add_argument("--ground-truth-only", action="store_true", help="Use only annotations marked ground truth.")
    p.add_argument("--coords-only", action="store_true", help="Download annotations but not tomograms.")
    p.add_argument("--fix-header", action="store_true", help="Write the portal voxel size into MRC headers that disagree.")
    p.add_argument("--allow-outside", action="store_true", help="Keep going when points fall outside the tomogram.")
    p.add_argument("--list", action="store_true", help="List runs, tomograms and objects; download nothing.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
