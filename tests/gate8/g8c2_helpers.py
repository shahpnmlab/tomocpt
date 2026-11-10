"""Fixtures of the Gate 8c.2 tests: one hand-made tomogram store entry (well-separated particles,
straight filaments, a negative mask) written with ``tomostore.write_store_entry``, so point and
segment mapping can be checked exactly."""
from pathlib import Path

import numpy as np

N_CLASSES = 2          # 0 = "ribo" particle (12 vox), 1 = "mt" filament (25 vox): the Gate 7 lock table at 10 A/px
D_PART = 12.0
D_FIL = 25.0
SHAPE = (96, 128, 128)


def _points(rng, n=14, shape=SHAPE, min_sep=26.0):
    pts = []
    while len(pts) < n:
        p = np.array([rng.uniform(-2, s + 2) for s in shape])  # some centres near or past the border
        if all(np.linalg.norm(p - q) >= min_sep for q in pts):
            pts.append(p)
    pts = np.asarray(pts, np.float32)
    return np.concatenate([pts, np.zeros((n, 1), np.float32), np.full((n, 1), D_PART, np.float32)], axis=1)


def _segments(rng, n_fil=3, n_seg=4, shape=SHAPE):
    rows = []
    for _ in range(n_fil):
        a = np.array([rng.uniform(10, s - 10) for s in shape])
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        for k in range(n_seg):
            b = a + 9.0 * d
            rows.append([*a, *b, 1.0, D_FIL])
            a = b
    return np.asarray(rows, np.float32)


def write_store(root: Path, seed: int = 0, name: str = "S__L0s0", with_mask: bool = True,
                with_segments: bool = True):
    """One store entry under ``root/name``; returns the opened :class:`TomogramStore`."""
    from scipy.ndimage import gaussian_filter

    from tomocpt.dataManager.metadata import ChunkMeta
    from tomocpt.dataManager.tomostore import open_store_dir, write_store_entry

    rng = np.random.default_rng(seed)
    vol = gaussian_filter(rng.normal(size=SHAPE).astype(np.float32), 1.5)
    vol = (vol / vol.std()).astype(np.float32) + 0.05  # median != 0: the border pad value is visible
    pts = _points(rng)
    segs = _segments(rng) if with_segments else np.zeros((0, 8), np.float32)
    mask = None
    if with_mask:
        mask = np.zeros(SHAPE, bool)
        mask[40:47] = True
        mask[:, 10:30, 77:101] = True
    meta = ChunkMeta(tomo_name=name.split("__")[0], dataset_id=0, level=0, scale_index=0, apix_native=10.0,
                     apix_work=10.0, class_available=[True, True], class_d_vox=[D_PART, D_FIL])
    d = write_store_entry(Path(root) / name, vol, pts, segs, mask, meta, drop_p=0.0, chunk_size=64,
                          chunk_stride=32, volume_dtype="float16")
    return open_store_dir(d)


def descriptors(corners, store_index=0, train=True, meta=None):
    """A collated batch of crop descriptors (what the loader yields with ``augment_device=cuda``)."""
    import torch

    corners = np.asarray(corners, np.int64).reshape(-1, 3)
    b = len(corners)
    out = {"store_index": torch.full((b,), int(store_index), dtype=torch.int64),
           "corner": torch.as_tensor(corners),
           "otf_train": torch.full((b,), bool(train), dtype=torch.bool)}
    if meta is not None:
        out["class_available"] = torch.tensor([list(meta.class_available)] * b, dtype=torch.bool)
        out["class_d_vox"] = torch.tensor([list(meta.class_d_vox)] * b, dtype=torch.float32)
        out["apix_work"] = torch.full((b, 1), float(meta.apix_work), dtype=torch.float32)
        out["dataset_id"] = torch.full((b,), int(meta.dataset_id), dtype=torch.int64)
    return out


def map_points(points, corner, A, n=64):
    """The documented map: source ``s = corner + c + A (j - c)``, ``c = (n - 1) / 2``, so a point
    ``p`` lands at ``c + A^-1 (p - corner - c)``; ``d_vox`` scales by ``|det A|^(-1/3)``."""
    A = np.asarray(A, np.float64)
    c = (n - 1) / 2.0
    pts = np.asarray(points, np.float64).reshape(-1, 5).copy()
    Ai = np.linalg.inv(A)
    pts[:, :3] = c + (pts[:, :3] - np.asarray(corner, np.float64) - c) @ Ai.T
    pts[:, 4] *= abs(np.linalg.det(A)) ** (-1.0 / 3.0)
    return pts


def map_segments(segs, corner, A, n=64):
    A = np.asarray(A, np.float64)
    c = (n - 1) / 2.0
    s = np.asarray(segs, np.float64).reshape(-1, 8).copy()
    Ai = np.linalg.inv(A)
    off = np.asarray(corner, np.float64) + c
    s[:, 0:3] = c + (s[:, 0:3] - off) @ Ai.T
    s[:, 3:6] = c + (s[:, 3:6] - off) @ Ai.T
    s[:, 7] *= abs(np.linalg.det(A)) ** (-1.0 / 3.0)
    return s


def source_coords(corner, A, n=64, field=None):
    """``[3, n, n, n]`` source positions (z, y, x) of every output voxel (optional displacement
    ``field [3, n, n, n]`` in output voxels, applied before ``A``)."""
    c = (n - 1) / 2.0
    j = np.stack(np.meshgrid(*(np.arange(n, dtype=np.float64),) * 3, indexing="ij"))
    if field is not None:
        j = j + np.asarray(field, np.float64)
    q = j.reshape(3, -1) - c
    s = np.asarray(A, np.float64) @ q + (np.asarray(corner, np.float64) + c)[:, None]
    return s.reshape(3, n, n, n)


def random_affine_np(rng, scale=(0.9, 1.1)):
    from scipy.spatial.transform import Rotation

    R = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
    return R @ np.diag(rng.uniform(*scale, size=3))
