"""Inputs and helpers shared by ``make_goldens_gate4.py`` and ``tests/gate4``.

Loaded by file path. ``tomocpt`` is imported inside the functions only, so the generator
(running in the ``v2-gate-0-base`` worktree) and the tests (running on HEAD) use whatever
``tomocpt`` is first on the import path.

* :func:`c1_tomogram` - the fixed synthetic tomogram of the G4-C1 golden.
* :data:`C1_CASES` - the ``InferConfig`` field values of each golden case.
* :func:`make_reinit_ckpt` - copy of the legacy tiny checkpoint whose network weights are
  replaced by the deterministic ``gate3_common.reinit_`` values (everything else, including
  the pickled legacy ``MainConfig`` and the optimizer state, is kept). The golden then does
  not depend on how the cached legacy checkpoint's one-epoch fit went.
* :func:`run_infer_tomos` - call ``tomocpt.predict.helpers.infer_tomos`` directly (no Dask;
  it is a plain function at ``v2-gate-0-base``) and return ``(confidence map, coords)``.
* :func:`fixed_unpad` - the legacy ``unpad`` with the axis order of its padding argument
  corrected (human decision, 27 Oct 2026: the golden is "base + unpad fix").
"""

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

C1_SHAPE = (48, 112, 112)
C1_APIX = 6.0
#: 150 A at 6 A/px = 25 px, so the fixed-particle rescale (target 20 px) runs; Z is padded.
C1_LENGTH = 150.0
C1_CASES = {
    "plain": dict(length=C1_LENGTH, confidence_threshold=0.5, use_tta=False),
}
C1_NUM_THREADS = 4


def _gate3_common():
    spec = importlib.util.spec_from_file_location("_gate3_common_for_g4", HERE / "gate3_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def c1_tomogram():
    """``(volume float32 (Z, Y, X), apix)``: white noise plus seven dark spheres of 150 A."""
    rs = np.random.RandomState(404)
    vol = 0.3 * rs.standard_normal(C1_SHAPE)
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in C1_SHAPE], indexing="ij")
    R = C1_LENGTH / 2.0 / C1_APIX
    for cz, cy, cx in [(24.3, 30.2, 30.7), (23.6, 30.9, 80.1), (24.8, 81.4, 30.2),
                       (22.9, 80.6, 80.8), (25.2, 56.1, 55.6), (12.0, 56.0, 14.0), (30.0, 100.0, 56.0)]:
        r = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        vol -= np.exp(-(r / R) ** 2 * 2.0)
    return vol.astype(np.float32), C1_APIX


def make_reinit_ckpt(legacy_ckpt, out_path):
    """Write ``out_path``: the legacy checkpoint with ``state_dict`` replaced by the weights of
    ``gate3_common.reinit_(model.model)`` (same keys as the legacy state dict)."""
    import torch

    from tomocpt.networks.inference import InferenceModel

    ckpt = torch.load(str(legacy_ckpt), map_location="cpu", weights_only=False)
    model = InferenceModel.load_from_checkpoint(str(legacy_ckpt), map_location="cpu", strict=True)
    _gate3_common().reinit_(model.model)
    sd = model.state_dict()
    ckpt["state_dict"] = {k: sd[k].detach().clone() for k in ckpt["state_dict"]}
    torch.save(ckpt, str(out_path))
    return Path(out_path)


def write_c1_tomogram(path):
    import mrcfile

    vol, apix = c1_tomogram()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(vol)
        m.voxel_size = apix
    return path


def run_infer_tomos(ckpt, tomo_path, predictions_dir, case):
    """``infer_tomos([tomo], gpu_id=None, worker_id=0, ckpt, cfg)`` on the CPU with the
    ``C1_CASES[case]`` values, the confidence map saved. Returns ``(map float32, coords
    float64 [N, 4] (z, y, x, score) or empty)``."""
    import mrcfile
    import torch

    from tomocpt.defaultConfigs.infer_config import InferConfig
    from tomocpt.predict.helpers import infer_tomos

    torch.set_num_threads(C1_NUM_THREADS)
    cfg = InferConfig()
    cfg.predictions_dir = Path(predictions_dir)
    cfg.use_cuda = False
    cfg.save_prediction_confidence_map = True
    cfg.save_predicted_coords = True
    for k, v in C1_CASES[case].items():
        setattr(cfg, k, v)
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)
    res = infer_tomos([Path(tomo_path)], None, 0, str(ckpt), cfg)
    map_path = Path(predictions_dir) / f"{Path(tomo_path).stem}.mrc"
    with mrcfile.open(str(map_path), permissive=True) as m:
        cmap = m.data.astype(np.float32).copy()
        map_apix = float(m.voxel_size.x)
    coords = np.zeros((0, 4))
    if res:
        coords = np.asarray(res[0]["coords"], dtype=np.float64).reshape(-1, 4)
    return cmap, coords, map_apix, res


def fixed_unpad(legacy_unpad):
    """Wrap the legacy ``tomocpt.predict.helpers.unpad``: ``symmetrize_and_pad`` returns the
    padding in ``F.pad`` order ``(x0, x1, y0, y1, z0, z1)`` but legacy ``unpad`` reads
    ``(z0, z1, y0, y1, x0, x1)``. The wrapper reorders the tuple, so the crop is correct."""

    def unpad(inp_tensor, padding_values):
        if padding_values is None:
            return legacy_unpad(inp_tensor, padding_values)
        x0, x1, y0, y1, z0, z1 = padding_values
        return legacy_unpad(inp_tensor, (z0, z1, y0, y1, x0, x1))

    return unpad
