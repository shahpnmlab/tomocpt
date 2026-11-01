"""TODO step 2: CZII OOF localisation diagnostics with cross-class suppression off and/or a
smaller peak spacing. Wraps experiments/diag_cv_localisation.py (unchanged); patches in memory.
Usage: python czii_cheap_checks.py <cv_dir> <nms 0|1> <spacing_fraction_of_diameter>
"""
import sys
from pathlib import Path

REPO = Path("/home/pranav/Desktop/pranav/software/tomocpt")
sys.path[:0] = [str(REPO), str(REPO / "experiments")]
import tomocpt.predict.helpers as H
import diag_cv_localisation as D

cv, nms, frac = sys.argv[1], bool(int(sys.argv[2])), float(sys.argv[3])
_orig_collect = D.collect_validation_picks
_orig_extract = H._extract_peaks_from_mask


def collect(model, val, cfg, device):
    cfg.cross_class_nms = nms
    return _orig_collect(model, val, cfg, device)


def extract(pred, name, apix, cfg, length=None, distance_threshold=None, threshold=None):
    if length is not None and frac != 1.0:
        distance_threshold = frac * length
    return _orig_extract(pred, name, apix, cfg, length=length, distance_threshold=distance_threshold,
                         threshold=threshold)


D.collect_validation_picks = collect
H._extract_peaks_from_mask = extract
sys.argv = [sys.argv[0], cv, "--label", f"5-fold CV, cross_class_nms={nms}, peak spacing {frac} x diameter "
            f"(OOF thresholds unchanged; TODO step 2, 1 Nov 2026)"]
D.main()
