# Gate 8c: windowed tube renderer and on-the-fly snapshot panels (14 Nov 2026)

Found by the DS-10490 actin run (on-the-fly + GPU augmentation): 20 min / epoch, no gain over the
old code. Cause: `render_tubes_torch` measured every segment against every voxel of the crop;
DS-10490 stores segments every 0.5 vox (~100 per crop, up to ~500; actin R ~5 vox).

Fix (`tomocpt/labels/render_torch.py`): each segment only in its box (bounds +/- R, clipped),
profile max-combined (the profile does not increase with the distance, so this equals the nearest
segment's value). Output: identical to the old renderer (max diff 0.0 on 16 real crops), CPU
`render_tubes` within 1.2e-7. CPU, B 16: 11.6 s -> 0.47 s.

Filament step (recipe of the actin run, `train.sampling=on_the_fly`, `augment_device=cuda`, B 16,
8 workers, 150 steps after 30 warm-up; `scratchpad run_tubes_wait.sh`; JSON in
`benchmarks/out/perf_8c_tubes_20261114/`):

| renderer | step | data wait | forward | backward |
|---|---|---|---|---|
| old (eafe09e) | 1061 ms | 312 ms (29 %) | 194 | 502 |
| windowed (b6b166d) | **801 ms** | **54 ms (6.7 %)** | 193 | 502 |

~15 min / epoch instead of 20. The remaining wait (6.7 %) is above the 5 % bar of 8c.2 (set on
particles at B 4); not investigated.

Snapshot panels (`train.snapshot_chunks`) now work with `sampling=on_the_fly`: the fixed crops are
the chunking-grid crops of the store, rendered once by the CPU on-the-fly path. Cost at fit start
on DS-10490: 59 s (train, 20 stores) + 5 s (val) - `_candidate_info` is recomputed; could reuse
the data module's candidate pass if it matters.
