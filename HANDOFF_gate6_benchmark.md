# HANDOFF — Gate 6 benchmark (state on 31 Oct 2026)

Read `HANDOFF_scale_free.md`, `CLAUDE.md`, `.claude/agents/06_AGENT_FILAMENTS.md` and
`.claude/reports/gate_06_report.json` first. Branch `v2-dev`, last commits `18ef039` (Gate 6
implementation), `21fd96c` (handoff), `58cc195` (diagnosis). Not pushed.

## 1. Gate 6 status: AWAITING_HUMAN (attempt 1)
- Fast suite 695 passed / 27 skipped; `tests/gate6` fast 55/55; frozen hashes unchanged; every
  changed path is inside Gate 6 ownership.
- Slow G6-C8 / G6-C9 FAIL (seed 0): joint particle F1 0.667 vs particle-only 1.00. Tubes pass
  (mt 1.00, actin 0.91–0.97).
- **Diagnosis (user-requested, seeds 0 and 1, scratch script
  `<scratchpad>/g6_confusion.py`, reuses the frozen slow-test functions):** the loss is class
  confusion between the two sphere classes, not filament interference. All 16/16 200 Å spheres
  are picked as the 120 Å class in both seeds; the 120 Å channel reads 0.87/0.95 at 200 Å centres
  vs 0.83/0.87 for the 200 Å channel, so the cross-class rule keeps the wrong class. The misses
  lie a median 27–29 voxels from any tube, none within the tube radius. Recorded in
  `gate_status.json` gates.6.notes.
- **Open human decision:** pass Gate 6 with C8/C9 as a recorded finding, or treat as a failed
  attempt (possible direction: size discrimination in the joint sigmoid model, e.g. softmax head
  or cross-class negatives in the slow setup — needs a test-dispute/ruling, tests are frozen).

## 2. Benchmark (T6.5): what the user asked for (31 Oct 2026)
"Launch the benchmark with 1 seed and all tomos (synthetic + the filament data with all molecular
species from the real data ...). End the run with F4 and F1 scores for the simulated and
experimental data respectively."

Facts found:
- DS-10490 (CryoETSim, simulated, 6.8 Å, 300 runs): ~125 point-annotated species per run (list via
  `scripts/download_portal_data.py --dataset-id 10490 --out <tmp> --list --max-runs 2`), plus
  microtubule / actin / membrane masks (ground truth). Particle points NOT downloaded yet
  (`--coords-only --voxel-spacing 6.8`; slow: ~2 min per 2 runs for the listing alone, expect hours
  for 300 runs × ~125 objects — run in the background).
- DS-10477 (13.481 Å, 12 runs) and DS-10513 (20.579 Å, 20 runs, labels not ground truth):
  **no particle points**, only microtubule + membrane masks. User: "great — we get to test the
  ability of the network to detect new particles in the experimental data." So experimental
  particle picks can only be reported qualitatively (counts per species, views for the user);
  experimental F1 = microtubule centreline F1 (length-weighted, 0.5 × diameter).
- Scores to report: simulated (DS-10490 held-out runs) **F4** (particles; also filament
  centreline Fβ), experimental (DS-10477 held-out, DS-10513 separately) **F1** (centrelines).
  `benchmarks/filaments.py::_summary` computes particle F-beta (β=4) and centreline P/R/F1; add a
  centreline F4 and a particle F1 column so both are available.
- The contrastive head / memory bank the user mentioned is **Gate 7 (not implemented)**. This
  benchmark trains the Gate 3/6 picker; tell the user again when launching.

User answers to the setup questions (31 Oct 2026):
- **Species:** all ~125 species (user choice). Drop the non-species point names (`membrane`,
  `protein-containing_complex`, and the `Actin` points, which duplicate the actin filament
  masks) — confirm with the user. Tiny species fall below the 10-voxel band at 6.8 Å and are used at
  native size (CLAUDE.md band rule).
- **Diameters:** the user's answer was a pasted CZII config block
  (`class_diameters "1:130,...,6:270"`, `class_names` beta-amylase ... virus-like-particle). Unclear
  whether that means "use this format", "add CZII 10440 as a dataset too", or "these are the
  diameters to use". **Ask before acting.** Options offered: estimate per-species diameters from
  subtomogram averages of the simulated data (radial profile; ~30 min CPU; show the table first),
  or the user supplies a name:diameter list.
- **Arms:** no preference. Recommendation: all three (joint, filaments-only, particles-only), needed
  for the spec's acceptance comparison; ~3× GPU time. Confirm.

## 3. Next steps, in order
1. Ask the user: (a) the meaning of the CZII paste (diameters source / extra dataset), (b) dropping
   the three non-species names, (c) arms (default all three), (d) Gate 6 pass/fail decision.
2. Download DS-10490 particle points (`--coords-only`) in the background; names in the STAR are
   portal run names (`sample_N`).
3. Diameters table (per the user's answer); review it with the user before training.
4. Extend `benchmarks/filaments.py` (driver written, never run):
   - stage per source dataset (`stage/{train,test}/<ds>/tomograms`) — a mixed-voxel-size raw dir
     breaks particle scaling in `prepare_picking_star_multiclass` (it reads one pixel size per
     directory); masks can stay in shared per-object dirs named `<ds>_<run>.mrc`;
   - per-dataset particle STARs with micrographs renamed `<ds>_<run>`; let
     `labels/run.py::_prepare_data_multiclass` accept a literal `none` entry in `coordinate_files`
     for the datasets without particles (10477);
   - with ~125 classes, check memory (logits B×C×64³) and step time (Hessian per class) on a short
     smoke run first (`--tiny --max_runs 4 --epochs 1`), then launch the real run in the background.
5. After the run: F4 (simulated) and F1 (experimental) tables, time and memory, experimental
   particle picks per species with a few views; put them in `gate_06_report.json` "benchmarks"
   and stop for the human (Gate 6 ends with a human acceptance decision).
6. Still open from Gate 6: README section for filaments; `read_filament_coords` ignores
   `rlnOrigin*` and RELION 5 centred coordinates.

Environment: `source /home/pranav/Desktop/pranav/software/miniconda3/etc/profile.d/conda.sh && conda activate tomocpt`
(RTX A6000; GPU idle at the end of this session).
