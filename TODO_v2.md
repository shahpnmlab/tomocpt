# TomoCPT v2 — ordered to-do list (set up 1 Nov 2026)

Status keys: [ ] open, [~] in progress, [x] done. Details live in `.claude/state/gate_status.json`
notes and the gate reports; this file is only the order of work.

## 1. Close Gate 6 (filaments)
- [x] 1.1 Test dispute + human ruling: slow C8/C9 train with `class_head=softmax`, k=0 (k=0 and
      0.25 both F1 1.00 on seeds 0, 1). `.claude/reports/gate_06_test_dispute.md`.
- [x] 1.2 Re-freeze `tests/gate6/test_g6_learning.py` (hashes in gate_status.json), rerun slow C8/C9: 2 passed (joint 1.00, mt 1.00, actin 0.92).
- [x] 1.3 README section for filaments (prepare_data inputs, infer outputs, settings).
- [x] 1.4 `read_filament_coords`: subtracts `rlnOrigin*`; RELION 5 centred coordinates rejected with a
      clear error (needs the tomogram size).
- [x] 1.5 (deferred, human decision 1 Nov) T6.5 filament benchmark on real data (microtubule): train DS-10490 subset + DS-10477,
      test DS-10477 held-out and DS-10513; centreline P/R/F1 (and F4). Smoke run works (1 Nov);
      full run needs a size decision (tiny: 17 min/epoch for 4 train tomograms).
- [x] 1.6 Gate 6 report + gate_status PASSED, tag `v2-gate-6-passed` (human acceptance).

## 2. Cheap CZII checks (no training; 5-fold CV models of 29 Oct)
- [x] 2.1 Out-of-fold F4 with cross-class suppression off: 0.631 -> 0.670.
- [x] 2.2 Smaller peak spacing (0.5 x d): 0.692; both: **0.737** (`experiments/czii_v2_results.md`).
- [x] 2.3 v2 defaults (human decision 1 Nov): `infer.peak_spacing_fraction = 0.5`, `infer.cross_class_nms
      = False`; v1 unchanged. Recalibrated OOF F4 **0.741** (`cv/ensemble_v2defaults.json`).

## 3. Gate 7 — discriminative loss (handoff: `HANDOFF_gate7.md`)
- [ ] 3.1 TA tests (phantom with many similar-size species; CZII as yardstick).
- [ ] 3.2 Implementation, slow tests, CZII CV comparison.

## 4. SSL v2 re-evaluation on CZII (Gate 5b parked)
- [ ] 4.1 kNN / linear probe of dense features on CZII particles; fine-tuning arms
      (`STEPS=ssl experiments/run_benchmarks.sh`, fresh `OUT`).

## 5. Scaling smoke test at ~130 classes
- [ ] 5.1 Memory and step time with C≈134 (logits B×C×64³, per-class Hessian loop); softmax over
      particles only vs particles+filaments.

## 6. DS-10490 "final boss" (only after 3–5)
- Prep done: points `data/filaments/10490_points/coords/`, diameters
  `experiments/configs/ds10490_species.csv`, config `experiments/configs/ds10490_train.yaml`.
