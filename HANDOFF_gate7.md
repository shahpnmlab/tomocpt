# HANDOFF — Gate 7 (discriminative training), written 1 Nov 2026

Read first: `CLAUDE.md`, `.claude/agents/00_ORCHESTRATOR.md`, `.claude/agents/07_AGENT_DISCRIMINATIVE.md`
(the spec), `.claude/agents/TA_TEST_AUTHOR.md`, `TODO_v2.md` (ordered work list), `gate_06_report.json`.
Branch `v2-dev`. Environment:
`source /home/pranav/Desktop/pranav/software/miniconda3/etc/profile.d/conda.sh && conda activate tomocpt`
(RTX A6000). User is usage-limited: implement directly as orchestrator (as Gates 3b/3d/4b), no
fan-out of agents, small sweeps, stop at the gate boundary.

## 1. State on entry
- Gates 1–5, 3b/3c/3d, 4b PASSED. Gate 5b PARKED (SSL v2 off by default). Gate 6: attempt 2,
  slow C8/C9 PASS after the human ruling (softmax head in the slow test,
  `.claude/reports/gate_06_test_dispute.md`); T6.5 real-data filament benchmark awaits a human
  decision on its size (see `TODO_v2.md` 1.5). Check `gate_status.json` gates.6 before starting:
  Gate 7 starts only once Gate 6 is PASSED or the human says to go ahead.
- DS-10490 (132 simulated species) is the "final boss": not used in Gate 7 (prep exists:
  `experiments/configs/ds10490_*`).

## 2. Findings that change how Gate 7 should be run (1 Nov 2026)
1. **Softmax is the multi-class head.** Sigmoid confused sphere sizes (C8: 16/16 200 Å spheres
   picked as 120 Å). Softmax with k = 0 fixed it (F1 1.00, seeds 0 and 1); k = 0.25 adds nothing.
   Gate 7 slow tests and CZII arms should run with `class_head = softmax`, k = 0. Keep the spec's
   requirement that both heads work.
2. **Cross-entropy that includes the background logit flattens heatmaps.** Scratch ablation on the
   C8 phantom: adding voxel CE on object cores (bg + classes) or CE at object centres classified
   perfectly but produced flat-topped maps → split peaks (particle F1 0.98 / 0.61; F1 at 0.3 without
   suppression 0.70 / 0.19) and fat tubes (actin precision 0.66 / 0.58). The focal value term (b)
   is the CenterNet penalty-reduced loss, which is designed for peaks, but **add a peak-shape check**
   to the tests: on the phantom, F1 at threshold 0.3 without cross-class suppression must not drop
   vs the Huber value term (otherwise the same failure can pass C9 at threshold 0.5).
   Script: `experiments/diag_g6_size_ablation.py <seed> A,B,C` (env `K_CCN`; loss functions
   `core_ce` / `object_ce`; helpers in `experiments/g6_confusion_lib.py`).
3. **The CZII baseline moved without retraining.** Inference settings alone (TODO step 2,
   `experiments/czii_v2_results.md`, last section; `experiments/diag_czii_inference_settings.py`): Kaggle-weighted OOF F4 0.631 → **0.737** with
   peak spacing 0.5 × diameter and cross-class suppression off (ribosome +0.24, apo-ferritin +0.11,
   beta-gal +0.11). Default v2 picking uses `min_distance = diameter` (`predict/helpers.py
   _extract_peaks_from_mask`, `distance_threshold or length`), which merges touching particles.
   **Done (human decision 1 Nov 2026):** v2 defaults `infer.peak_spacing_fraction = 0.5` (new field)
   and `infer.cross_class_nms = False`; v1 unchanged. Recalibrated OOF thresholds:
   `benchmarks/out/czii_bench_20261029_2237/cv/ensemble_v2defaults.json`, Kaggle-weighted OOF F4
   **0.741**. Gate 7's acceptance (≥ 0.02 OOF F4) is measured against 0.741, with the same
   inference defaults and recalibrated thresholds for every arm.
4. Remaining CZII error after (3): beta-amylase (F4 0.37, weight 0 in Kaggle) and beta-gal (0.60);
   about 95 % of false picks are confident picks on real unlabelled densities — exactly what the
   hard-negative score cap (`hard_negative_max_score = 0.9`) is for. Do not mine near-certain picks.

## 3. Steps
1. Confirm Gate 6 is closed (or human go-ahead).
2. **TA step** (`TA_TEST_AUTHOR.md`): write `tests/gate7/` from spec C1–C9, plus the peak-shape
   check of 2.2 inside C9 (or a C10; record as an orchestrator ruling). Calibrate the C9 phantom with
   a scratch reference (solid vs hollow spheres of one size, 70 Å tube, unlabelled sheets with
   negative masks; `tests/utils/phantoms.make_phantom` supports hollow spheres and tubes; sheets may
   need a helper in `tests/gate7/`). Freeze: hashes into `gate_status.json` gates.7.
3. Implement (paths in the spec's "Allowed write paths"). Public names are fixed by the spec.
   Notes: `forward_features` must not change `forward` (invariant 7); the head is created only when
   `contrastive_weight > 0` (invariant 5) and always gets a gradient; the memory bank is a buffer.
   Negative masks: membrane masks exist for every filament dataset
   (`data/filaments/{10477,10490,10513}/masks/membrane/`).
4. `pytest tests -m "not slow"`, `pytest tests/gate7 -m slow`.
5. Human acceptance: CZII 5-fold CV (`experiments/run_cv.sh`, softmax) for (a), (b), (a)+(b) against
   the 0.741 baseline; plus filament centreline F1 on the Gate 6 data with membranes. Each CV
   arm is five trainings of up to 60 epochs: confirm GPU time with the user before launching.
6. Report `.claude/reports/gate_07_report.json`; status AWAITING_HUMAN; stop.

## 4. Known traps
- Tests are frozen once committed; disputes go to `.claude/reports/gate_07_test_dispute.md`.
- Unpickled checkpoint configs lack new fields: read `train.discriminative` with
  `safe_getattr(cfg, name, default)` before `BaseModel.__init__` (CLAUDE.md 1.9).
- Calibration after training runs even for filament-only models and finds no particles
  (≈ 5 min per validation tomogram wasted) — Gate 4 path, note only.
- bf16: SupCon logits / temperature 0.1 overflow easily; compute the loss in float32.
