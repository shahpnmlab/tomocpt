# Gate 8c.1 results (10 Nov 2026, RTX A6000, CZII fold 0, recipe czii_3b_ema_lr)

On-the-fly crops from the tomogram store (`train.sampling=on_the_fly`, float16) vs chunk files.
JSON in `benchmarks/out/perf_8c_20261110/`.

## Data preparation and loader (`benchmarks/perf/loader.py`, 40 stage samples, 20 batches)
| | chunks | on-the-fly |
|---|---|---|
| prepare data (7 tomograms) | minutes (Dask chunking + `.nii.gz` writes) | 10.5 s |
| disk per fold | 8.1 GB | 976 MB (store, written once for all folds) |
| read volume + target per sample | 23.4 ms | 1.6 ms |
| render_targets | 1.0 ms | 0.7 ms |
| TorchIO transform (unchanged) | 139 ms | 137 ms |
| samples/s, 0 / 8 workers | 6.0 / 56.5 | 6.9 / 58.7 |
| training samples per epoch | 6489 (repeat-factor draws) | 6482 (expected, same rule) |
| validation | 2 tomograms' chunks | 2645 fixed crops (`val_seed` 1234) |

The CPU path stays transform-bound (TorchIO affine/elastic); 8c.2 moves it to the GPU.

## 300-step loss overlay (G8c-C9, 8c.1 part; `run_8c1_overlay.sh`, 8 workers, B 16)
50-step moving average of the training loss, mean |difference| over steps 100-300.
| pair | mean abs diff |
|---|---|
| chunks seed 0 vs seed 1 (noise) | 0.0246 |
| on-the-fly seed 0 vs seed 1 | 0.0297 |
| on-the-fly vs chunks (4 pairs) | 0.0032, 0.0227, 0.0316, 0.0071 |
Bar: 3 x noise = 0.074; max cross difference 0.032. **Within the rule.**
Ruling: the noise is two chunk runs with different seeds (data order and init), not two runs with
identical data: an E2 change alters the samples, so identical-data spread (bf16 nondeterminism only)
would fail any sampling change. Step 815-823 ms, data wait 18 ms, peak 9.9 GB in all four runs.
