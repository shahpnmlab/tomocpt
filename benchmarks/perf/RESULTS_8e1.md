# Gate 8e.1 results (predict and calibration, exact), 10 Nov 2026

`benchmarks/perf/pipeline.py`, no profiler, 2 validation tomograms each, A6000, idle host. Baseline =
corrected 8a numbers (`BASELINE_20261109.md`); "now" includes 8b.4 (fused attention).

| Stage | 8a | now | |
|---|---|---|---|
| CZII predict, per tomogram | 68 s | **27 s** | 2.5× |
| Filament predict, per tomogram | 42 s | **18 s** | 2.3× |
| CZII post-fit calibration (2 tomograms) | 231 s | **65 s** | 3.5× |
| Filament calibration (2 tomograms) | 126 s | **52 s** | 2.4× |

Changes, all with identical outputs (frozen tests against frozen copies of the 8a code):
1. `peak_local_max_torch` / `_prune_refined_peaks_gpu`: greedy passes on the CPU (in torch), distances
   and plateau centroids on the image's device; no device sync per candidate. Peak extraction in
   CZII calibration 92 → 9 s.
2. Filament threshold tracing: thresholds in parallel forked worker processes (automatic count, maps
   < 4M voxels in-process). Scoring 43 → 19 s.
3. GPU patch grid: patches sliced from the volume on the GPU at TorchIO's locations, order and batch
   size (`grid_locations`); 12.8 → 36 patch batches/s. Real-data check: CZII picks identical with
   the grid on and off.
4. `accumulate_device=auto` counts the allocator cache as free (robustness; no speed change).

Not done: a larger default `predictions_batch_size` (spec: human decision; would change cudnn kernels,
E1). Calibration already predicts once per tomogram for particles and filaments.
GPU predict picks differ from 8a only through 8b.4 (bf16 attention): CZII TS_6_6 apo-ferritin 211 → 212.
