# Handoff: finish Gate 8 and merge into v2-dev (written 14 Nov 2026)

Read `CLAUDE.md`, `.claude/agents/08_AGENT_PERFORMANCE.md` (spec), `HANDOFF_gate8.md` and the
`benchmarks/perf/RESULTS_*.md` files first. Memory note `gate8-performance-plan` has the history.

## Rules (unchanged, inviolable)
- All Gate 8 work on **`v2-feat/perf`**, worktree `/home/pranav/Desktop/pranav/software/tomocpt-perf`.
  Side branches merge into it, never into `v2-dev`. Never touch `main`.
- **No merge into `v2-dev` until every remaining item below is done or ruled by the human, and
  only with the human's explicit go-ahead.** One merge.
- Speed-ups are defaults; never add `tomocpt` CLI options (fields via YAML / `key=value` only).
  Benchmark scripts may take options.
- Do not edit a worktree while a training run imports from it. Check `nvidia-smi` and
  `pgrep -af "tomocpt|run_cv|filaments.py"` before any GPU timing. Frugal: few subagents, short outputs.
- Python: `/home/pranav/Desktop/pranav/software/miniconda3/envs/tomocpt/bin/python`, `PYTHONPATH=<worktree>`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` + the session line.

## State at handoff
`v2-feat/perf` head `d8fb1ee` (+ this file). Tags: `v2-gate-8a-passed`, `v2-gate-8b-passed`,
`v2-gate-8e1-passed`. Side branch `v2-feat/lr-scaling` (worktree `../tomocpt-lrscale`, head
`d6270ee`, 8g) not merged yet.

### 8c: accepted by the human (13-14 Nov 2026), paperwork left
- CZII 5-fold CV, recipe `czii_3b_ema_lr` (outputs `benchmarks/out/cv_8c/{chunks,otf}/cv/cv_summary.md`):
  weighted OOF F4 chunks **0.748**, on-the-fly + GPU aug **0.763**; no weighted class drop
  (VLP -0.008); wall 29.9 h vs 38.4 h (same min/epoch, early stop fires later: 51-79 vs 45-65 epochs).
  Rule was |dF4| <= 0.01: exceeded on the better side - **human accepted as pass**.
- DS-10490 actin (`benchmarks/out/filaments_8c_otf_20261113/`, run with the OLD tube renderer,
  no snapshot panels): actin F1 **0.836** (P 0.869, R 0.806, point F1 0.780) vs 0.810 reference;
  microtubule 0.9995. 65 epochs, 22.0 h. **Human accepted.** The failed first attempt (panels crash)
  is kept as `filaments_8c_otf_20261113.failed_snapshots`.
- Found and fixed during the run (merged, `b6b166d`, `d8fb1ee`; `benchmarks/perf/RESULTS_8c_tubes.md`):
  windowed `render_tubes_torch` (identical output; filament step 1061 -> 801 ms, data wait 312 -> 54 ms
  = 6.7 %, above the 5 % bar set on particles - note in report); snapshot panels now work with
  `sampling=on_the_fly` (new test `tests/gate8/test_g8c_snapshots_otf.py`; costs ~60 s at fit start on
  DS-10490 because `_candidate_info` is recomputed - optional: reuse the data module's pass).
- `benchmarks/filaments.py` got `--train_set key=value` (`eafe09e`).

## To do, in order
1. **Check the gate-8 test run** started at handoff (`pytest -m "not slow" tests/gate8 tests/test_v2_snapshots.py`
   on `d8fb1ee`); fix anything red.
2. **8c close-out** (human-approved):
   a. Filament recipe default: `benchmarks/filaments.py:apply_recipe` sets `train["sampling"]="on_the_fly"`
      and `train["otf"]={"augment_device": "cuda"}`. Check `tests/gate6c` does not lock the recipe dict
      (if it does: test dispute file, do not edit frozen tests). CZII recipe: human said on_the_fly is the
      default once CV confirms - ask whether `TrainConfig.sampling` default itself flips now or stays opt-in
      for v1 equivalence (invariant 1 suggests: recipes yes, dataclass default no; confirm).
   b. `.claude/reports/gate_08c_report.json` (schema in `00_ORCHESTRATOR.md`; numbers above + 8c.1/8c.2
      results docs), `gate_status.json` entry with test hashes of `tests/gate8/*8c*`, tag `v2-gate-8c-passed`.
3. **8d** (spec section 8d): filament-only calibration skips the particle path (G8d-C1, E0);
   `train.cv_parallel_folds` (G8d-C2 on CPU; GPU scaling unmeasured, no second GPU); OOF picks never
   re-predicted (the CV log shows "reusing ... val_picks.npz", so likely already true - verify, test).
4. **8e.2 and 8f: DROPPED** (human ruling 14 Nov 2026). Record in the Gate 8 report with the 8f numbers:
   torch 2.15.0+cu132 / MONAI 1.6.1 / Kornia 0.9.0 / TorchIO 1.2.1 / Lightning 2.6.6 / numpy 2.5 gave
   step 816 -> 796 ms (defaults), 752 -> 733 (Swin ckpt off) = ~1.025x; regional compile still 22-24 min;
   MONAI >= 1.5 removed `img_size` (`network_config.build_model` would have to drop it); state_dict keys
   identical. Python stays 3.12. (`requires-python` in `pyproject.toml` is a stale `>=3.8`; mention it.)
5. **8g LR scaling** (`v2-feat/lr-scaling`, worktree `../tomocpt-lrscale`): **no GPU validation** (human
   ruling 14 Nov 2026: it is a separate matter, wanted only as an option). Stays opt-in
   (`train.lr_scaling="none"` default). Merge it into `v2-feat/perf` as the **last** step, after all
   performance work above is in; rebase/merge conflicts likely only in `training/helpers.py` and
   `defaultConfigs/train_config.py`; run `tests/gate8/test_g8g_lr_scaling.py` + gate4f after the merge.
6. **Open human decisions** (ask together): Swin checkpointing off in recipes
   (1.09x, 2x activation memory); larger `infer.predictions_batch_size` default.
7. **Before the v2-dev merge:** full `pytest tests -m "not slow"` and `pytest tests/gate8 -m slow` on the
   final `v2-feat/perf`; last full suite was before 8c.2's CVs (1163 passed, 27 skipped). Gate 8 summary
   report; update `CLAUDE.md` (sections 6/7 config rows: `train.sampling`, `train.otf.*`,
   `train.lr_scaling`, `train.lr_reference_batch`, Gate 8 ownership row) with human approval; then ask
   for the merge go-ahead.

## Useful paths
- Results: `benchmarks/perf/RESULTS_8b*.md`, `RESULTS_8c1.md`, `RESULTS_8c2.md`, `RESULTS_8c_tubes.md`, `RESULTS_8e1.md`.
- Step benchmark: `python -m benchmarks.perf.step --config <yaml> --data chunks:<dir> --batch 16 --set k=v`.
- Filament recipe config of the actin run: `benchmarks/out/filaments_8c_otf_20261113/filaments/config.yaml`
  (store in `.../filaments/chunks`).
- CZII recipe: main repo `benchmarks/out/czii_3b_ema_lr/config.yaml`; CV driver `benchmarks/perf/run_8c_cv.sh`.
