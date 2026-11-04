# TomoCPT v2 — ordered to-do list (set up 1 Nov 2026)

Status keys: [ ] open, [~] in progress, [x] done. Order of work after Gate 7's CZII CV (human decision
4 Nov 2026): 3.2b (close Gate 7), 3a (ensembling in the package), 3b, 3c, then 4, 5, 6 (DS-10490, the
~130-class run), then 8. Commit after every finished item so
each state can be restored. Details live in `.claude/state/gate_status.json`
notes and the gate reports; this file is only the order of work.

## 1. Close Gate 6 (filaments)
- [x] 1.1 Test dispute + human ruling: slow C8/C9 train with `class_head=softmax`, k=0 (k=0 and
      0.25 both F1 1.00 on seeds 0, 1). `.claude/reports/gate_06_test_dispute.md`.
- [x] 1.2 Re-freeze `tests/gate6/test_g6_learning.py` (hashes in gate_status.json), rerun slow C8/C9: 2 passed (joint 1.00, mt 1.00, actin 0.92).
- [x] 1.3 README section for filaments (prepare_data inputs, infer outputs, settings).
- [x] 1.4 `read_filament_coords`: subtracts `rlnOrigin*`; RELION 5 centred coordinates rejected with a
      clear error (needs the tomogram size).
- [x] 1.5 (deferred, human decision 1 Nov; now item 3c.1) T6.5 filament benchmark on real data (microtubule): train DS-10490 subset + DS-10477,
      test DS-10477 held-out and DS-10513; centreline P/R/F1 (and F4). Smoke run works (1 Nov);
      full run needs a size decision (tiny: 17 min/epoch for 4 train tomograms).
- [x] 1.6 Gate 6 report + gate_status PASSED, tag `v2-gate-6-passed` (human acceptance).

## 2. Cheap CZII checks (no training; 5-fold CV models of 29 Oct)
- [x] 2.1 Out-of-fold F4 with cross-class suppression off: 0.631 -> 0.670.
- [x] 2.2 Smaller peak spacing (0.5 x d): 0.692; both: **0.737** (`experiments/czii_v2_results.md`).
- [x] 2.3 v2 defaults (human decision 1 Nov): `infer.peak_spacing_fraction = 0.5`, `infer.cross_class_nms
      = False`; v1 unchanged. Recalibrated OOF F4 **0.741** (`cv/ensemble_v2defaults.json`).

## 3. Gate 7 — discriminative loss (handoff: `HANDOFF_gate7.md`)
- [x] 3.1 TA tests (phantom with many similar-size species; CZII as yardstick).
- [~] 3.2 Implementation, slow tests (done), CZII CV comparison: focal 0.730 vs 0.741 (fails +0.02;
      thyroglobulin background FPs 937 -> 1282, recall equal; `benchmarks/out/czii_g7_focal/oof_analysis/`),
      contrastive 0.742 (`benchmarks/out/czii_g7_contrastive/oof_analysis/`: thyroglobulin FPs 937 -> 840,
      recall 210 -> 203); focal+contrastive (`both`) 0.737 (`benchmarks/out/czii_g7_both/oof_analysis/`: thyroglobulin FPs 822, recall 201; apo-ferritin and VLP up, beta-galactosidase and beta-amylase down). Exact-coordinate rescoring: baseline 0.741, focal 0.727, contrastive 0.729, both 0.737. All arms stopped early (see 3b.2).
- [x] 3.2b Close Gate 7 (human ruling 4 Nov 2026: failed as tested, re-judge after 3b; focal stays off by
      default; 3b uses the `both` flavour, baseline retrained with the same recipe): `both` result; OOF analysis of `both`; rescore all four arms against the exact
      STAR coordinates; final `gate_07_report.json` (all done 4 Nov); open: human ruling on Gate 7 (pass / fail / re-judge after
      3b) and on the focal default; tag if accepted. Pick the training flavour for 3b.
- [x] 3.2c Housekeeping after `both`: delete the baseline CV fold chunks
      (`benchmarks/out/czii_bench_20261029_2237/cv/fold*/chunks`, 5 x 8.1 GB; the Gate 7 arms link to
      them) and remove the worktree `../tomocpt-g6` (keep `../tomocpt-base`, the legacy reference).
- [ ] 3.3 After Gate 7 closes: update the invariants in `CLAUDE.md` (human decision 3 Nov 2026), e.g.
      invariant 8 (dependency list, `schedulefree` if approved in 7) and Sections 6/7 for Gate 7.

## 3a. Built-in CV + ensembling in the package, opt-in (human decision 4 Nov 2026; first after Gate 7's CV)
Today: `train.cv_folds` / `train.cv_fold`, `tomocpt.predict.ensemble` and `infer.weights = <ensemble JSON>`
are in the package, but the fold loop, the OOF ensemble build and the CV summary live in
`experiments/run_cv.sh`.
- [x] 3a.1 Opt-in CV mode in `tomocpt train` (e.g. a flag that trains every fold of `train.cv_folds` in
      turn, reuses / shares chunks where the split allows, resumes skipped / finished folds), then builds
      the ensemble with OOF calibration and writes the CV summary (per-class OOF F4, Kaggle-style weights
      optional). Default off: `tomocpt train` unchanged (invariant 1). `run_cv.sh` becomes a thin wrapper.
- [x] 3a.2 Save the pooled OOF picks (and truths) with the ensemble so diagnostics (per-fold F4, FP
      breakdown, galleries; `benchmarks/out/czii_g7_focal/oof_analysis/`) need no re-prediction.
- [x] 3a.3 Peak-shape diagnostic in the CV / validation output (human request 4 Nov 2026): per class and
      arm, the mean predicted probability around true centres as a radial profile and as line profiles
      along x, y, z (z shows the missing wedge), overlaid on the target profile
      (0.5 narrow sigma 0.165 R + 0.5 wide sigma 0.33 R, cut at R); plus peak heights and FWHM of the
      narrow core. Question to answer: is the narrow central peak flattened by the soft focal value term
      and/or the softmax head? Compare baseline (Huber, softmax), focal, contrastive, both; and one
      sigmoid-head model as the reference without softmax competition.
      First look (4 Nov 2026, TS_69_2, `benchmarks/out/peak_shapes_TS_69_2/`): every model (softmax
      Huber / focal / contrastive, sigmoid Huber) reproduces only the wide Gaussian: peak height 0.3-0.9
      (target 1.0), FWHM 0.63-0.83 R in xy (target 0.53 R; wide component alone 0.78 R), z wider; the
      tails match. Not softmax- or focal-specific (focal peaks slightly higher, contrastive lower). Likely
      the MSE-type optimum under localisation noise (truncated labels, annotation noise, wedge) blurring a
      1-2.5 voxel core; to test with 3b.1 (exact coordinates) and 3b.3b (centroid weight, offset head).
      Also add a localisation metric to the CV summary: per class, mean / median distance of matched picks
      to the exact true centres (today only F4 is reported, which hides localisation at 0.5 R).
- [x] 3a.4 Bug (found 4 Nov 2026): in one Python process, loading a contrastive checkpoint
      (`train.discriminative.contrastive_weight > 0`) and then one without the head fails:
      `InferenceModel.load_from_checkpoint(strict=True)` -> missing `contrastive_head.*`,
      `contrastive_bank.*`. The earlier load's `train` settings leak via the global `mainConfig`
      (CLAUDE.md 1.10; the Gate 3c snapshot covers CLI runs only). Built-in CV / ensembling loads many
      checkpoints per process, so fix (build the model from the checkpoint's own config) and test this
      in 3a.
- [x] 3a.5 Tests (TA first), README section. Done (Gate 4c: spec `.claude/agents/04c_AGENT_CV_ROUTINE.md`,
      `tests/gate4c/`, report `gate_04c_report.json`); test dispute resolved (G4b-C5 re-frozen, no leaks). Gate 8's "built-in CV mode" (HANDOFF_gate8 step 8d) moves here.

## 3b. Train longer, exact coordinates (human decision 4 Nov 2026; after the Gate 7 `both` arm)
Plan: take the best training flavour of the Gate 7 CZII CV (baseline / focal / contrastive / both, judged
on OOF F4 against the exact STAR coordinates), add the sub-voxel fix, and train with a modern stopping /
LR recipe; expectation: clears +0.02 over 0.741 by training longer.
- [ ] 3b.1 Sub-voxel coordinates: `labels/star.py:137` and `labels/mod.py:53` truncate to int
      (`astype(int)`, legacy): v2 coords CSV / points / targets are biased by -0.5 voxel per axis
      (mean 3D offset 0.94 voxel). Keep floats on the v2 path (v1 unchanged, invariant 1); regenerate
      affected v2 goldens with a ruling; fresh chunks. Rescoring the Gate 7 OOF picks against exact
      coordinates: baseline 0.741, focal 0.727, contrastive 0.729 (apo-ferritin most affected).
- [ ] 3b.2 Configurable early stopping: `train.early_stop_min_delta` (legacy 0.005, absolute) and
      patience; run with 1e-4 so the plateau LR halving and SWA (epoch 48 of 60) can act.
- [ ] 3b.3 Schedule-free optimiser as the alternative arm (moved here from item 7, human decision 4 Nov
      2026). Evidence: in all finished CZII CV folds the LR never left 4e-4 and SWA never started; early
      stopping fired at epochs 24-29 with the best epoch at or near the last in most folds
      (`EarlyStopping(min_delta=0.005)` is absolute, 30-50 % of the final `val_loss_smooth`;
      `ReduceLROnPlateau` uses a relative 1e-4 threshold; legacy `v2-gate-0-base`). Opt-in
      `RAdamScheduleFree` (`schedulefree` 1.4.1 on PyPI; check that it ships the RAdam variant; new
      dependency, pin needs human approval, invariant 8). Lightning: `optimizer.train()` / `.eval()`
      around training, validation, checkpoint saving and inference (the averaged weights are evaluated
      and saved); no LR scheduler; early stopping scale-free (relative, or slope of `val_loss_smooth`
      against its own noise) instead of absolute `min_delta`. Defaults keep RAdam + plateau +
      EarlyStopping (invariant 1); fine-tune differential LR groups kept.
- [ ] 3b.3a Weight averaging without a fixed start epoch (human decision 4 Nov 2026). RAdam arm: EMA of
      the weights from step 0 (warm-up decay, e.g. `min(0.999, (1+t)/(10+t))`; validate and save the EMA
      weights), or SWA started by the scale-free plateau trigger (own callback on
      `torch.optim.swa_utils.AveragedModel`) instead of Lightning's fixed `swa_epoch_start` (0.8 x
      max_epochs). Schedule-free arm: no SWA/EMA (the method already averages). Verify first: (a) whether
      Lightning's `StochasticWeightAveraging` applies the averaged weights when early stopping fires before
      `max_epochs`, and which weights the best checkpoint holds. (Checked 4 Nov: SwinUNETR has only
      LayerNorm / InstanceNorm3d, so no `update_bn` is needed.) Legacy SWA callback stays the default (invariant 1).
- [ ] 3b.3b Sharper peaks and sub-voxel localisation (human decisions 4 Nov 2026; after 3b.1, which may
      already help): the narrow core of the target is not learned (3a.3), and picks are integer voxel
      indices (`peak_local_max_torch`).
      (a) Centroid-weighted Huber: multiply the Section 1.5 value weight by `1 + core_weight * y^p` (or by
      the narrow component's share of y), so the few core voxels count; new field e.g.
      `train.particles.core_weight` = 0.0 (off, invariant 1). Check the Hessian term's weight alongside.
      (b) Sub-voxel peak refinement in `predict` (no training; the reference for (c)): weighted centroid
      or quadratic fit in a small window around each local maximum; opt-in field in `infer`.
      (c) Offset head: a small extra head predicting the sub-voxel offset (dz, dy, dx) from each voxel to
      its particle centre (CenterNet-style), L1 on voxels near labelled centres, class-agnostic (3
      channels) unless per-class proves necessary; zero-initialised (invariant 6), all parameters get a
      gradient every step when on and the module is not created when off (invariant 5); kept out of the
      class logits (invariant 7: `forward` still returns `(logits, hidden4)`; the offsets travel by a
      separate path, e.g. a `forward_offsets` method or the inference wrapper); saved in checkpoints and
      read by `predict` to shift each pick. Needs exact coordinates (3b.1) to have a target worth
      learning. New config fields (default off) and a gate-style spec + TA tests before code.
      Measure (a)-(c) with the 3a.3 peak-shape diagnostic, the localisation metric and OOF F4.
- [ ] 3b.4 Pilot on fold 0 (both recipes), then full CZII CV of the winner. A fair Gate 7 verdict needs the
      baseline (Huber, no contrastive) retrained with the same recipe as well.

## 3c. Filament mode: full training and evaluation (human decision 4 Nov 2026; after 3b)
Filaments are less tested than particles (Gate 6 passed on synthetic data plus a smoke run only).
- [ ] 3c.1 T6.5 real-data benchmark (deferred item 1.5): train on DS-10490 subset + DS-10477, test
      DS-10477 held-out and DS-10513; centreline precision / recall / F1 (and F4); pick the run size first
      (tiny was 17 min/epoch for 4 train tomograms).
- [ ] 3c.2 Same with the Gate 7 losses and membrane negative masks
      (`data/filaments/{10477,10490,10513}/masks/membrane/`), with the 3b recipe (exact coordinates,
      early stopping / schedule-free).
- [ ] 3c.3 Visual check of traced filaments (missed / false / broken traces) as for the CZII particles.

## 4. SSL v2 re-evaluation on CZII (Gate 5b parked)
- [ ] 4.1 kNN / linear probe of dense features on CZII particles; fine-tuning arms
      (`STEPS=ssl experiments/run_benchmarks.sh`, fresh `OUT`).

## 5. Scaling smoke test at ~130 classes
- [ ] 5.1 Memory and step time with C≈134 (logits B×C×64³, per-class Hessian loop); softmax over
      particles only vs particles+filaments; with the 3b recipe and, if adopted, the offset head and
      centroid weight.

## 6. DS-10490 "final boss" (only after 3–5)
- Prep done: points `data/filaments/10490_points/coords/`, diameters
  `experiments/configs/ds10490_species.csv`, config `experiments/configs/ds10490_train.yaml`.

## 7. Schedule-free optimiser
- Moved into item 3b.3 (human decision 4 Nov 2026).

## 8. Gate 8 — performance (only after 3–6 are validated; human decision 2 Nov 2026)
- [ ] 8.1 Codebase scan + spec `.claude/agents/08_AGENT_PERFORMANCE.md` (handoff: `HANDOFF_gate8.md`).
- [ ] 8.2 Implementation on `v2-feat/perf` (own worktree), training first; goldens regenerated per
      change after equivalence tests against the old code.
