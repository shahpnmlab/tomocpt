"""Filament classes at inference (Gate 6): tracing of the per-class maps and the filament
output files (one file per class, Gate 4 rule).

Every filament is a list of points every ``infer.filament_spacing_ang`` A along a smoothing
spline (``infer.filament_smoothing_ang``), with the RELION helical priors of its local axis:
``rlnAngleRot = 0`` (the spin is unknown), ``rlnAngleTilt`` / ``rlnAnglePsi`` from the tangent
(``labels.filaments.tangents_to_relion_angles``), the same values as priors, and
``rlnAnglePsiFlipRatio = 0.5`` (polarity unknown).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import starfile
import torch

from tomocpt.logger import get_logger

logger = get_logger()

FILAMENT = "filament"


def is_filament(row) -> bool:
    return str(getattr(row.kind, "value", row.kind)) == FILAMENT


def filament_rows(class_table: Sequence) -> List[int]:
    return [i for i, r in enumerate(class_table) if is_filament(r)]


def _fom(prob: np.ndarray, pts_xyz: np.ndarray) -> float:
    from scipy.ndimage import map_coordinates

    zyx = np.asarray(pts_xyz, dtype=np.float64)[:, ::-1].T
    v = map_coordinates(prob, zyx, order=1, mode="nearest")
    return float(np.mean(v)) if len(v) else 0.0


def trace_class_maps(maps: Dict[int, torch.Tensor], class_table: Sequence, apix: float, infer_config,
                     thresholds: Dict[int, float], tomo_name: str) -> Dict[int, List[Dict[str, Any]]]:
    """``{row: [filament, ...]}`` for the filament rows of ``maps``; a filament is a dict with
    ``points`` (``[K, 3]`` x, y, z native voxels), ``arc_ang``, ``tilt``, ``psi`` (degrees) and
    ``fom`` (mean probability along it)."""
    from tomocpt.labels.filaments import _trace_and_fit, tangents_to_relion_angles

    out: Dict[int, List[Dict[str, Any]]] = {}
    for c, m in maps.items():
        if not is_filament(class_table[c]):
            continue
        prob = m.detach().float().cpu().numpy()
        fits = _trace_and_fit(prob, float(thresholds[c]), float(apix), float(class_table[c].diameter_ang),
                              float(infer_config.filament_min_length_ang), float(infer_config.filament_spacing_ang),
                              float(infer_config.filament_smoothing_ang))
        fils = []
        for pts, tng, arc in fits:
            tilt, psi = tangents_to_relion_angles(tng)
            fils.append({"points": np.asarray(pts, float).tolist(), "arc_ang": np.asarray(arc, float).tolist(),
                         "tilt": np.asarray(tilt, float).tolist(), "psi": np.asarray(psi, float).tolist(),
                         "fom": _fom(prob, pts)})
        logger.info(f"{tomo_name}: {len(fils)} {class_table[c].name} filament(s) traced")
        out[int(c)] = fils
    return out


def _filament_frame(entries, tomo_label: str, warp: bool, class_number=None) -> pd.DataFrame:
    rows = []
    for name, fils in entries:
        tomo = re.sub(r"_\d+\.\d+Apx", ".tomostar", name) if warp else name
        for tid, f in enumerate(fils, start=1):
            pts = np.asarray(f["points"], float)
            for k in range(len(pts)):
                r = {tomo_label: tomo, "rlnCoordinateX": pts[k, 0], "rlnCoordinateY": pts[k, 1],
                     "rlnCoordinateZ": pts[k, 2]}
                if class_number is not None:
                    r["rlnClassNumber"] = int(class_number)
                r.update({"rlnHelicalTubeID": tid, "rlnHelicalTrackLengthAngst": float(f["arc_ang"][k]),
                          "rlnAutopickFigureOfMerit": float(f["fom"]), "rlnAngleRot": 0.0,
                          "rlnAngleTilt": float(f["tilt"][k]), "rlnAnglePsi": float(f["psi"][k])})
                if not warp:
                    r.update({"rlnAngleTiltPrior": float(f["tilt"][k]), "rlnAnglePsiPrior": float(f["psi"][k]),
                              "rlnAnglePsiFlipRatio": 0.5, "rlnOpticsGroup": 1})
                rows.append(r)
    return pd.DataFrame(rows)


def write_filament_star_file(output_dir, entries, voxel_size: float, output_filename: str,
                             version=3.1, class_number=None) -> Path:
    """RELION 3.1 / 5.0 filament STAR: ``entries`` = ``[(tomogram name, [filament, ...]), ...]``."""
    if version == 3.1:
        tomo_label = "rlnMicrographName"
    elif version == 5.0:
        tomo_label = "rlnTomoName"
    else:
        raise ValueError(f"Unsupported Relion version: {version}. Supported versions are 3.1 and 5.0")
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnOpticsGroupName": ["OpticsGroup1"],
                           "rlnSphericalAberration": [2.7], "rlnVoltage": [300],
                           "rlnImagePixelSize": [float(voxel_size)], "rlnImageDimensionality": [3]})
    df = _filament_frame(entries, tomo_label, warp=False, class_number=class_number)
    path = Path(output_dir) / output_filename
    starfile.write({"optics": optics, "particles": df}, path, float_format="%.4f", overwrite=True)
    logger.info(f"Traced filaments are stored in: {path}")
    return path


def write_filament_warp_star_file(output_dir, entries, output_filename: str) -> Path:
    df = _filament_frame(entries, "rlnMicrographName", warp=True)
    path = Path(output_dir) / output_filename
    starfile.write({"particles": df}, path, float_format="%.4f", overwrite=True)
    logger.info(f"Traced filaments are stored in: {path}")
    return path


def write_filament_imod_model(output_dir, tomo_name: str, filaments, output_filename: str):
    """IMOD model: one open contour per filament (points x, y, z in voxels; no angles)."""
    import imodmodel
    from imodmodel.models import ContourType

    rows = []
    for cid, f in enumerate(filaments):
        for x, y, z in np.asarray(f["points"], float):
            rows.append({"object_id": 0, "contour_id": cid, "x": x, "y": y, "z": z})
    if not rows:
        logger.warning(f"No filaments for {tomo_name}, skipping {output_filename}.")
        return None
    path = Path(output_dir) / output_filename
    imodmodel.write(pd.DataFrame(rows), str(path), type=ContourType.OPEN)
    logger.info(f"IMOD filament model for {tomo_name} stored in: {path}")
    return path
