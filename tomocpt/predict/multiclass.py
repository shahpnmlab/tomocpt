"""Pipeline-v2 prediction: every class of a multi-class checkpoint in one run.

The plan of working voxel sizes comes from the checkpoint's class table and band
(``helpers.plan_inference``); each class is read at exactly one level, its map is resampled
back to the native shape, and peaks are found per class with the class diameter and the
class threshold (``infer.class_thresholds`` > checkpoint calibration >
``infer.confidence_threshold``). Training-time settings are read from the checkpoint
config with ``safe_getattr``, never from the global ``mainConfig.train``.
"""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mrcfile
import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader
from tqdm import tqdm

from tomocpt.configManager.configManager import safe_getattr
from tomocpt.dataManager.preprocessing import load_mrc, resize_volume, symmetrize_and_pad
from tomocpt.dataManager.scaling import read_apix, resample_volume, working_shape
from tomocpt.defaultConfigs.infer_config import InferConfig, OutputFormat
from tomocpt.defaultConfigs.scale_config import ScaleConfig
from tomocpt.logger import get_logger
from tomocpt.predict import helpers as H

logger = get_logger()

#: Lower bound of the Hann weights of the per-class accumulators. Without it, the corner
#: weights of a 64^3 window (about 1e-8) underflow in float16 and give NaN or large errors
#: at the volume border.
HANN_MIN_WEIGHT = 1e-2

#: Fraction of the free GPU memory that ``accumulate_device="auto"`` may use for the maps.
AUTO_DEVICE_MEMORY_FRACTION = 0.5

THRESHOLD_SOURCES = ("class_thresholds", "calibration", "confidence_threshold")


# --------------------------------------------------------------------------------------
# checkpoint settings
# --------------------------------------------------------------------------------------

@dataclass
class CheckpointSettings:
    """What the predict driver needs from a checkpoint without building the network."""
    pipeline_version: int
    class_table: list
    calibration: Optional[dict]


def read_checkpoint_settings(ckpt_path) -> CheckpointSettings:
    """Pipeline version, class table and calibration of a checkpoint file (raw load; the
    config fields of later gates are read with ``safe_getattr``)."""
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.predict.ensemble import is_ensemble_file, read_ensemble

    if is_ensemble_file(ckpt_path):
        # An ensemble: the first member's settings, the ensemble's calibration.
        record = read_ensemble(ckpt_path)
        first = read_checkpoint_settings(record["checkpoints"][0])
        return CheckpointSettings(first.pipeline_version, first.class_table, record.get("calibration"))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]
    version = int(safe_getattr(config.train, "data_pipeline_version", 1) or 1)
    table = class_table_from_config(config.prepData) if version == 2 else []
    return CheckpointSettings(version, table, ckpt.get("tomocpt_calibration"))


def scale_config_of(model) -> ScaleConfig:
    """``train.scale`` of the model's checkpoint config (defaults when absent)."""
    return safe_getattr(model.hparams.config.train, "scale", None) or ScaleConfig()


# --------------------------------------------------------------------------------------
# class selection and thresholds
# --------------------------------------------------------------------------------------

def _split_names(text: Optional[str]) -> List[str]:
    return [n.strip() for n in str(text).split(",") if n.strip()] if text else []


def selected_classes(class_table: Sequence, classes: Optional[str]) -> List[int]:
    """Class-table rows named in ``classes`` (comma-separated names), or every row.
    An unknown name raises ValueError naming it."""
    if not classes:
        return list(range(len(class_table)))
    names = [row.name for row in class_table]
    wanted = _split_names(classes)
    unknown = [n for n in wanted if n not in names]
    if unknown:
        raise ValueError(f"infer.classes: unknown class name(s) {unknown}; the model has {names}")
    return [i for i, row in enumerate(class_table) if row.name in wanted]


def parse_class_thresholds(text: Optional[str], class_table: Sequence) -> Dict[str, float]:
    """``"ribosome:0.4,proteasome:0.35"`` -> ``{name: threshold}``. An unknown class name
    raises ValueError naming it."""
    names = [row.name for row in class_table]
    out = {}
    for part in _split_names(text):
        name, sep, value = part.rpartition(":")
        name = name.strip()
        if not sep or not name:
            raise ValueError(f"infer.class_thresholds: expected 'name:threshold', got {part!r}")
        if name not in names:
            raise ValueError(
                f"infer.class_thresholds: unknown class name {name!r}; the model has {names}"
            )
        out[name] = float(value)
    return out


def _calibrated_entry(calibration: Optional[dict], class_number: int) -> Optional[dict]:
    if not calibration:
        return None
    classes = calibration.get("classes") or {}
    return classes.get(class_number, classes.get(str(class_number)))


def resolve_thresholds(
    class_table: Sequence,
    infer_config: InferConfig,
    calibration: Optional[dict],
    log: bool = True,
) -> Dict[int, Tuple[float, str]]:
    """``{row: (threshold, source)}``. Precedence: ``infer.class_thresholds``, the
    checkpoint calibration, ``infer.confidence_threshold``."""
    explicit = parse_class_thresholds(infer_config.class_thresholds, class_table)
    if log and calibration is None:
        logger.warning(
            "This pipeline-v2 checkpoint has no threshold calibration; classes without "
            "`infer.class_thresholds` use `infer.confidence_threshold`."
        )
    out = {}
    for i, row in enumerate(class_table):
        entry = _calibrated_entry(calibration, row.number)
        if row.name in explicit:
            out[i] = (explicit[row.name], "class_thresholds")
        elif entry is not None and entry.get("threshold") is not None:
            out[i] = (float(entry["threshold"]), "calibration")
        else:
            out[i] = (float(infer_config.confidence_threshold), "confidence_threshold")
        if log:
            logger.info(f"Class {row.name}: threshold {out[i][0]:.3f} ({out[i][1]})")
    return out


# --------------------------------------------------------------------------------------
# per-class maps
# --------------------------------------------------------------------------------------

def _accumulator_device(setting: str, compute_device: torch.device, n_maps: int,
                        shape: Sequence[int], dtype: torch.dtype) -> torch.device:
    setting = str(setting or "auto").lower()
    if setting == "cpu":
        return torch.device("cpu")
    if compute_device.type != "cuda":
        if setting in ("cuda", "gpu"):
            logger.warning("accumulate_device=cuda without a GPU; accumulating on the CPU.")
        return torch.device("cpu")
    if setting in ("cuda", "gpu"):
        return compute_device
    n_vox = int(np.prod([int(s) for s in shape]))
    needed = n_vox * (n_maps * torch.tensor([], dtype=dtype).element_size() + 4)
    free, _ = torch.cuda.mem_get_info(compute_device)
    return compute_device if needed < AUTO_DEVICE_MEMORY_FRACTION * free else torch.device("cpu")


def _hann_window(patch_size: int) -> torch.Tensor:
    """TorchIO's Hann window (``GridAggregator``), clamped to :data:`HANN_MIN_WEIGHT`."""
    w1 = torch.hann_window(patch_size + 2, periodic=False)[1:-1]
    w = w1.view(-1, 1, 1) * w1.view(1, -1, 1) * w1.view(1, 1, -1)
    return w.float().clamp_min(HANN_MIN_WEIGHT)


def _predict_level(model, padded: torch.Tensor, classes: List[int], apix_work: float, overlap: int,
                   infer_config: InferConfig, device: torch.device, pbar: Optional[tqdm],
                   desc: str) -> torch.Tensor:
    """Hann-weighted grid prediction of ``classes`` on one padded working volume (with flip
    TTA if enabled). Returns ``[len(classes), *padded.shape]`` float32 on the CPU."""
    patch = int(model.patch_size)
    dtype = getattr(torch, str(infer_config.accumulate_dtype))
    acc_device = _accumulator_device(infer_config.accumulate_device, device, len(classes), padded.shape, dtype)
    hann = _hann_window(patch).to(acc_device)
    # TorchIO's GridSampler needs an even overlap.
    overlap = int(overlap) + int(overlap) % 2
    flips = [None] + ([0, 1, 2] if infer_config.use_tta else [])

    maps = []
    for k, flip in enumerate(flips):
        vol = padded if flip is None else torch.flip(padded, dims=(flip,))
        sampler = tio.GridSampler(tio.Subject(volume=tio.ScalarImage(tensor=vol.unsqueeze(0))),
                                  patch_size=patch, patch_overlap=overlap)
        loader = DataLoader(sampler, batch_size=infer_config.predictions_batch_size, num_workers=0)
        acc = torch.zeros((len(classes),) + tuple(vol.shape), dtype=dtype, device=acc_device)
        weight = torch.zeros(tuple(vol.shape), dtype=torch.float32, device=acc_device)
        if pbar is not None:
            tta = f" [TTA {k + 1}/{len(flips)}]" if len(flips) > 1 else ""
            pbar.reset(total=len(loader))
            pbar.set_description(f"{desc} @ {apix_work:.2f} A/px{tta}")
        with torch.inference_mode():
            for batch in loader:
                x = batch["volume"][tio.DATA].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=infer_config.use_cuda):
                    probs = model.predict_step(x, 0, apix=float(apix_work), classes=classes)
                probs = probs[:, classes].float().to(acc_device)
                for p, loc in zip(probs, batch[tio.LOCATION]):
                    i0, j0, k0, i1, j1, k1 = (int(v) for v in loc)
                    acc[:, i0:i1, j0:j1, k0:k1] += (p * hann).to(dtype)
                    weight[i0:i1, j0:j1, k0:k1] += hann
                if pbar is not None:
                    pbar.update(1)
        m = (acc.float() / weight).cpu()
        del acc, weight
        if flip is not None:
            m = torch.flip(m, dims=(flip + 1,))
        maps.append(m)
    return torch.stack(maps).mean(dim=0) if len(maps) > 1 else maps[0]


def predict_class_maps(model, tomo_fname, infer_config: InferConfig, device: torch.device,
                       classes: Optional[Sequence[int]] = None,
                       pbar: Optional[tqdm] = None) -> Tuple[Dict[int, torch.Tensor], float]:
    """Per-class probability maps of one tomogram at its native shape.

    ``classes``: class-table rows to read (default: ``infer.classes``). The levels are
    planned on the full class table, so a class keeps its level when others are dropped.
    Returns ``({row: [D, H, W] float32 CPU tensor}, apix_native)``.
    """
    tomo_fname = Path(tomo_fname)
    table = model.class_table()
    scale_cfg = scale_config_of(model)
    if classes is None:
        classes = selected_classes(table, infer_config.classes)
    keep = set(int(c) for c in classes)

    apix_native = read_apix(tomo_fname, infer_config.apix_override)
    vol = load_mrc(str(tomo_fname), normalize=True)  # robust normalisation at native size
    if infer_config.invert_contrast:
        vol *= -1
    native_shape = tuple(int(s) for s in vol.shape)
    desired_px = safe_getattr(model.hparams.config.prepData, "desired_particle_pixel_size", 20)
    levels = H.plan_inference(apix_native, table, scale_cfg, desired_px, 2)
    vol_t = torch.from_numpy(vol)
    del vol

    maps: Dict[int, torch.Tensor] = {}
    for level in levels:
        level_classes = [c for c in level.classes if c in keep]
        if not level_classes:
            continue
        logger.info(
            f"{tomo_fname.name}: level {level.apix_work:.3f} A/px, classes "
            f"{[table[c].name for c in level_classes]}, overlap {level.overlap}"
        )
        work_shape = working_shape(native_shape, apix_native, level.apix_work)
        x = vol_t.to(device)
        if tuple(work_shape) != native_shape:
            x = resample_volume(x, work_shape, scale_cfg.resample)
        padded, padding = symmetrize_and_pad(x.float(), int(model.patch_size))
        padded = padded.cpu()  # TorchIO samples patches from a CPU tensor
        del x
        level_maps = _predict_level(model, padded, level_classes, level.apix_work, level.overlap,
                                    infer_config, device, pbar, tomo_fname.name)
        del padded
        zyx_padding = H.fpad_to_zyx(padding)
        for j, c in enumerate(level_classes):
            m = H.unpad(level_maps[j], zyx_padding).contiguous()
            if tuple(m.shape) != native_shape:
                m = resample_volume(m.to(device), native_shape, scale_cfg.resample).cpu()
            # Fourier resampling rings slightly outside [0, 1].
            maps[c] = m.float().clamp_(0.0, 1.0)
        del level_maps

    if infer_config.masks_dir:
        mask_path = Path(infer_config.masks_dir) / tomo_fname.name
        if mask_path.exists():
            mask = torch.from_numpy(load_mrc(str(mask_path), normalize=False)).float()
            if tuple(mask.shape) != native_shape:
                mask = resize_volume(mask, native_shape)
            for c in maps:
                maps[c] *= mask
        else:
            logger.warning(f"Mask file not found for {tomo_fname.name} at {mask_path}, proceeding without it.")
    return maps, float(apix_native)


# --------------------------------------------------------------------------------------
# peaks
# --------------------------------------------------------------------------------------

def peaks_from_class_maps(maps: Dict[int, torch.Tensor], class_table: Sequence, apix: float,
                          infer_config: InferConfig, thresholds: Dict[int, float], tomo_name: str,
                          device: torch.device = torch.device("cpu"),
                          ) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int, int], int]]:
    """Peaks per class (``[N, 4]`` rows ``z, y, x, score`` in native voxels), found with the
    class diameter and ``thresholds[row]``, then cross-class suppression if
    ``infer.cross_class_nms``. Returns ``(peaks, removed_counts)`` keyed by class-table row."""
    peaks = {}
    for c, m in maps.items():
        res = H._extract_peaks_from_mask(
            m.to(device).float(), tomo_name, apix, infer_config,
            length=float(class_table[c].diameter_ang), threshold=float(thresholds[c]),
        )
        peaks[c] = np.asarray(res["coords"], dtype=np.float64).reshape(-1, 4) if res else np.zeros((0, 4))
    counts: Dict[Tuple[int, int], int] = {}
    if infer_config.cross_class_nms and len(peaks) > 1:
        radii = {c: float(class_table[c].diameter_ang) / (2.0 * apix) for c in peaks}
        peaks, counts = H.cross_class_suppression(peaks, radii)
        for (kept, removed), n in sorted(counts.items()):
            logger.info(
                f"{tomo_name}: cross-class suppression removed {n} {class_table[removed].name} "
                f"peak(s) next to {class_table[kept].name}"
            )
    return peaks, counts


def _result(tomo_name: str, apix: float, peaks: Dict[int, np.ndarray],
            counts: Dict[Tuple[int, int], int]) -> Dict[str, Any]:
    return {
        "name": tomo_name,
        "voxel_size": float(apix),
        "class_coords": {int(c): p.tolist() for c, p in peaks.items()},
        "suppressed": {(int(a), int(b)): int(n) for (a, b), n in counts.items()},
    }


# --------------------------------------------------------------------------------------
# maps on disk
# --------------------------------------------------------------------------------------

def class_map_path(predictions_dir, tomo_stem: str, class_name: str) -> Path:
    return Path(predictions_dir) / f"{tomo_stem}_{class_name}.mrc"


def save_class_maps(maps: Dict[int, torch.Tensor], class_table: Sequence, apix: float,
                    thresholds: Dict[int, float], tomo_stem: str, predictions_dir) -> None:
    """``<stem>_<class name>.mrc`` per class (float16, as the v1 map), ``<stem>_heatmap.mrc``
    (float32 ``max_c p_c``) and ``<stem>_classmap.mrc`` (class number of the winning class
    above its threshold, else 0), all at native shape with the native voxel size."""
    rows = sorted(maps)
    for c in rows:
        H.write_segmentation_mask(maps[c], str(class_map_path(predictions_dir, tomo_stem, class_table[c].name)),
                                  angpix=apix, overwrite=True)
    heatmap, classmap = H.merge_class_maps(
        torch.stack([maps[c] for c in rows]), [class_table[c].number for c in rows], [thresholds[c] for c in rows]
    )
    out_dir = Path(predictions_dir)
    mrcfile.write(str(out_dir / f"{tomo_stem}_heatmap.mrc"), data=heatmap.astype(np.float32),
                  voxel_size=apix, overwrite=True)
    # MRC has no uint8 mode: mrcfile stores uint8 as uint16 (mode 6) anyway.
    mrcfile.write(str(out_dir / f"{tomo_stem}_classmap.mrc"), data=classmap.astype(np.uint16),
                  voxel_size=apix, overwrite=True)


# --------------------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------------------

def infer_tomos_v2(model, tomo_fnames: List[Path], gpu_id: Optional[int], worker_id: int,
                   model_fname, infer_config: InferConfig, device: torch.device) -> List[Dict[str, Any]]:
    """Dask worker body for a pipeline-v2 model (called by ``helpers.infer_tomos``).
    Errors propagate (``future.result()`` re-raises them in ``predict``)."""
    from tomocpt.predict.calibration import read_calibration

    table = model.class_table()
    thresholds = {c: t for c, (t, _) in
                  resolve_thresholds(table, infer_config, read_calibration(model_fname)).items()}
    classes = selected_classes(table, infer_config.classes)
    results = []
    worker_desc = f"GPU-{gpu_id}" if gpu_id is not None else f"CPU-{worker_id}"
    with tqdm(total=1, desc=f"{worker_desc} | Waiting...", position=worker_id, leave=True) as pbar:
        for tomo_fname in tomo_fnames:
            tomo_fname = Path(tomo_fname)
            try:
                maps, apix = predict_class_maps(model, tomo_fname, infer_config, device, classes, pbar)
                if infer_config.save_prediction_confidence_map:
                    save_class_maps(maps, table, apix, thresholds, tomo_fname.stem, infer_config.predictions_dir)
                if infer_config.save_predicted_coords:
                    peaks, counts = peaks_from_class_maps(maps, table, apix, infer_config, thresholds,
                                                          tomo_fname.stem, device)
                    results.append(_result(tomo_fname.stem, apix, peaks, counts))
                del maps
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return results


def peaks_from_saved_maps(tomo_fnames: List[Path], settings: CheckpointSettings,
                          infer_config: InferConfig) -> List[Dict[str, Any]]:
    """``skip_inference`` for pipeline v2: re-reads ``<stem>_<class name>.mrc`` from
    ``predictions_dir`` and extracts the peaks per class (in this process)."""
    table = settings.class_table
    thresholds = {c: t for c, (t, _) in resolve_thresholds(table, infer_config, settings.calibration).items()}
    classes = selected_classes(table, infer_config.classes)
    device = torch.device("cuda") if infer_config.use_cuda and torch.cuda.is_available() else torch.device("cpu")
    results = []
    for tomo_fname in tomo_fnames:
        stem = Path(tomo_fname).stem
        maps, apix = {}, None
        for c in classes:
            path = class_map_path(infer_config.predictions_dir, stem, table[c].name)
            if not path.is_file():
                logger.warning(f"No confidence map for class {table[c].name} of {stem} at {path}, skipping it.")
                continue
            data, apix = load_mrc(str(path), normalize=False, return_voxel_size=True)
            maps[c] = torch.from_numpy(data).float()
        if not maps:
            continue
        peaks, counts = peaks_from_class_maps(maps, table, float(apix), infer_config, thresholds, stem, device)
        results.append(_result(stem, float(apix), peaks, counts))
    return results


def write_class_outputs(results: List[Dict[str, Any]], class_table: Sequence, infer_config: InferConfig) -> None:
    """One coordinate file per class in every format: ``<name>_<class name>.<ext>`` for
    RELION 3.1/5.0 (with ``rlnClassNumber`` = the original class number) and Warp, where
    ``<name>`` is the stem of ``predictions_coord_filename``; ``<tomogram stem>_<class name>.mod``
    for IMOD. No merged file."""
    out_dir = str(infer_config.predictions_dir)
    fmt = OutputFormat(infer_config.predictions_coord_format)
    base = Path(infer_config.predictions_coord_filename)
    suffix = base.suffix or ".star"
    rows_present = sorted({c for res in results for c in res["class_coords"]})
    for c in rows_present:
        row = class_table[c]
        entries = [(res["name"], res["voxel_size"], res["class_coords"].get(c, [])) for res in results]
        if not any(coords for _, _, coords in entries):
            logger.warning(f"No coordinates were extracted for class {row.name}, skipping its file.")
            continue
        if fmt == OutputFormat.imod:
            for name, _, coords in entries:
                if coords:
                    H.write_imod_model(out_dir, name, coords, f"{Path(name).stem}_{row.name}.mod")
            continue
        names = [name for name, _, coords in entries for _ in coords]
        coords = [p for _, _, cs in entries for p in cs]
        voxel_sizes = [vs for _, vs, cs in entries for _ in cs]
        if not all(abs(vs - voxel_sizes[0]) < 1e-4 for vs in voxel_sizes):
            logger.warning(
                f"Detected multiple voxel sizes. Using the first ({voxel_sizes[0]:.2f} A/px) for metadata."
            )
        fname = f"{base.stem}_{row.name}{suffix}"
        if fmt == OutputFormat.warp:
            H.write_warp_star_file(out_dir, names, coords, fname)
        elif fmt in (OutputFormat.relion_31, OutputFormat.relion_50):
            H.write_relion_star_file(out_dir, names, coords, voxel_sizes[0], fname,
                                     version=3.1 if fmt == OutputFormat.relion_31 else 5.0,
                                     class_number=int(row.number))
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

    totals: Dict[Tuple[int, int], int] = {}
    for res in results:
        for key, n in res.get("suppressed", {}).items():
            totals[key] = totals.get(key, 0) + n
    for (kept, removed), n in sorted(totals.items()):
        logger.info(f"Cross-class suppression, all tomograms: {class_table[kept].name} removed "
                    f"{n} {class_table[removed].name} peak(s)")
