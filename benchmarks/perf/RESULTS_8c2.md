# Gate 8c.2: GPU augmentation and target rendering (10 Nov 2026)

`train.otf.augment_device=cuda` (opt-in until the CZII CV acceptance run). Recipe czii_3b_ema_lr
(flips, blur, contrast inversion), CZII fold 0 tomogram store, RTX GPU of the development machine.
JSON in `benchmarks/out/perf_8c_20261110/`; scripts `run_8c2_wait.sh`, `make_c134_store.py`.

## Data wait (G8c-C9, 8c.2 part), B = 4, 150 steps after 30 warm-up
| case | workers | step ms | data wait ms | share |
|---|---|---|---|---|
| C = 6, GPU path | 2 | 233.6 | 10.3 | **4.4 %** |
| C = 6, CPU path (reference) | 8 | 227.5 | 4.8 | 2.1 % |
| C = 134, GPU path | 2 | 733.9 | 13.2 | **1.8 %** |

Bar < 5 %: **met** at both class counts. The timer synchronises CUDA, so GPU augmentation and
rendering count as data wait. C = 134 uses the CZII volumes with 134 synthetic particle classes
(~3000 points per tomogram, d_vox 10-32), since no 134-class store exists yet. At B = 4 and C = 6
the CPU path is faster on this machine (8 workers keep up); the GPU path frees the workers
(2 instead of 8) and does not depend on CPU count.

## 300-step loss overlay, GPU path vs chunk baseline (B 16; the 8c.1 rule)
| pair | mean abs diff |
|---|---|
| chunks seed 0 vs seed 1 (noise) | 0.0246 |
| GPU path seed 0 vs seed 1 | 0.0246 |
| GPU path vs chunks (4 pairs) | 0.0068, 0.0179, 0.0312, 0.0068 |
| GPU path vs CPU on-the-fly path (same seed) | 0.0048, 0.0012 |

Bar 3 x noise = 0.074; max 0.031: **within the rule**. Step 820-822 ms, data wait 19-20 ms
(2 workers; CPU path 18 ms with 8), peak 10.7 GB (CPU path 9.9 GB; resident fp16 tomograms).

## Not ported (ValueError naming `augment_device`)
Fourier degradation (`use_mw_aug`, `use_fourier_aug`) and the optional intensity / TorchIO
stand-in transforms. No current recipe turns them on.
