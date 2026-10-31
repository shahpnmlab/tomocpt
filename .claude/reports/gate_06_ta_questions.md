# Gate 6 — Test Author questions

## Open

7. **G6-C8/C9 slow setup is not proven reachable.** After the fix (pick threshold 0.5 plus the Gate 4
   cross-class rule), the scratch reference gives: particle-only F1 0.958 (passes 0.8); joint model
   particle F1 0.667 (tp 32, fp 16, fn 16), mt 1.00, actin 0.94. So C8 fails on particle F1 and C9
   fails (0.667 < 0.958 - 0.03). The joint model's particle scores are lower, which is interference
   in the tiny model. Options, each needing one more slow run of about 12 min:
   (a) more steps for the joint model (for example 4000; the slow tier then takes about 18 min);
   (b) pick threshold 0.3 with the cross-class rule (not measured yet);
   (c) accept C8/C9 as reported numbers until the real implementation is trained.
   Thresholds are unchanged.

## Resolved (31 Oct 2026)

1. Hollow masks (human decision): G6-C2 now has a hollow 250 A wall case (150 A lumen, open
   ends) and hollow crossing tubes, same thresholds as the solid case
   (`test_G6_C2_hollow_*`). Data fact: DS-10490 masks are hollow walls; DS-10477 / DS-10513 solid.
2. `benchmarks.filaments.centerline_scores(pred_polylines, true_polylines, tolerance_vox) -> dict`
   (`precision`, `recall`, `f1`, length-weighted), orchestrator ruling; tested by
   `test_G6_bench_centerline_scores`.
3. Chunking finds the filament CSV by any route; the test goes through public entry points
   (accepted).
4. C10 smoothing: matched to the noise for the noisy test, 2 A for the two-voxel-size test
   (accepted; reasoning in the test docstrings).
5. C1 reference = `v2-dev` before Gate 6 (accepted).
6. C8 particle threshold stays 0.8 (human decision). The setup was changed (see open point 7).
