"""Data path alone (spec 8a): per-stage ms of one v2 training sample (one thread, as a DataLoader
worker) and loader throughput (samples/s) for several worker counts. No GPU step.

``python -m benchmarks.perf.loader --config recipe.yaml [--data chunks:<dir>] --workers 0 4 8 16 --out x.json``
"""
import argparse
import time

import numpy as np
import torch

from benchmarks.perf.common import applied_config, environment, git_commit, load_config, write_json


def _stages(ds, n: int, seed: int) -> dict:
    """Per-stage ms over ``n`` samples, replaying ``VolumeDatsetIO._getitem_v2`` stage by stage."""
    import torchio as tio
    from tomocpt.labels.helpers import render_targets

    rng = np.random.default_rng(seed)
    otf_keys = None
    if hasattr(ds, "stores"):  # Gate 8c on-the-fly dataset: keys are crops (store, z, y, x)
        from tomocpt.dataManager.otf import candidate_weights, draw_crops

        index, w, _ = candidate_weights(ds.stores, int(ds.n_classes))
        otf_keys = draw_crops(index, w, n, seed, 0, 0, 1, True, ds.stores[0].chunk_stride,
                              [s.shape for s in ds.stores], ds.chunk_size)
        idx = np.arange(len(otf_keys))
    else:
        idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    acc = {"read_volume_target": [], "read_segments_mask": [], "render_targets": [], "render_tubes": [],
           "transform": [], "getitem_total": []}
    torch.manual_seed(seed)
    for i in idx:
        t0 = time.perf_counter()
        if otf_keys is not None:
            ds._key = tuple(int(v) for v in otf_keys[i])
            i = 0
        vol, points, dense, meta = ds._load_v2(int(i))
        t1 = time.perf_counter()
        segs = ds._load_segments(ds.filepath_tuples[i][1])
        if ds.has_negative_masks:
            ds._load_negative_mask(ds.filepath_tuples[i][0])
        t2 = time.perf_counter()
        n_classes = int(ds.n_classes if ds.n_classes is not None else len(meta.class_available))
        size = tuple(vol.shape[-3:])
        target = render_targets(points, size, n_classes, "clip") if points is not None else dense.numpy()
        t3 = time.perf_counter()
        if segs is not None:
            from tomocpt.labels.filaments import render_tubes
            target = np.maximum(target, render_tubes(segs, size, n_classes))
        t4 = time.perf_counter()
        if ds.transform is not None:
            subj = tio.Subject(input_data=tio.ScalarImage(tensor=vol),
                               target_data=tio.LabelMap(tensor=torch.from_numpy(np.ascontiguousarray(target))))
            ds.transform(subj)
        t5 = time.perf_counter()
        ds[ds._key if otf_keys is not None else int(i)]
        t6 = time.perf_counter()
        for k, v in zip(acc, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)):
            acc[k].append(v * 1e3)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _throughput(dm, workers: int, batches: int) -> dict:
    dm.hparams.workers_for_data = int(workers)
    loader = dm.train_dataloader()
    it = iter(loader)
    next(it)  # worker start-up
    t0 = time.perf_counter()
    n = 0
    for _ in range(batches):
        try:
            b = next(it)
        except StopIteration:
            break
        n += int(b["input_data"].shape[0])
    dt = time.perf_counter() - t0
    del it
    return {"workers": int(workers), "samples_per_s": n / dt if dt > 0 else 0.0, "batches": batches}


def run_loader_benchmark(config=None, data="chunks", workers=(0, 8), batches=20, stage_samples=30, seed=0,
                         overrides=()) -> dict:
    from pathlib import Path

    cfg = load_config(config, overrides)
    if data.startswith("chunks:"):
        cfg.train.chunks_dir = Path(data.split(":", 1)[1])
    cfg.train.use_gpus = True  # persistent workers as in GPU training
    with applied_config(cfg) as mc:
        from benchmarks.perf.step import _chunks_datamodule

        torch.set_num_threads(1)
        dm = _chunks_datamodule(mc.train)
        out = {"stages_ms": _stages(dm.dataset_training, stage_samples, seed),
               "n_classes": len(dm.class_table or []), "batch": int(mc.train.batch_size)}
        torch.set_num_threads(max(1, torch.get_num_threads()))
        out["throughput"] = [_throughput(dm, w, batches) for w in workers]
    out.update({"commit": git_commit(), "env": environment(), "data": data})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--data", default="chunks")
    p.add_argument("--workers", type=int, nargs="+", default=[0, 8])
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--stage_samples", type=int, default=30)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    r = run_loader_benchmark(a.config, a.data, a.workers, a.batches, a.stage_samples, overrides=a.overrides)
    write_json(a.out, r)
    print(r["stages_ms"], r["throughput"])


if __name__ == "__main__":
    main()
