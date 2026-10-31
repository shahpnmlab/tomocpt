# HANDOFF — TomoCPT v2 (state on 31 Oct 2026)

Read this first in any new session. Then read `CLAUDE.md`, `.claude/agents/00_ORCHESTRATOR.md`
and `.claude/state/gate_status.json` (all on branch `v2-dev`, pushed to origin with the `v2-*` tags).

## 1. Where things are

| Tag | Content |
|---|---|
| `v2-gate-0-base` | released v0.7.0 + `fix/v1-patches` + the plan |
| `v2-gate-1-passed` | legacy goldens, checkpoints, metrics, config backfill fix |
| `v2-gate-2-passed` | pipeline v2: band rule, levels, multi-class points |
| `v2-gate-3-passed` | multi-class heads, FiLM scale conditioning, v2 loss |
| `v2-gate-3b-passed` | cross-class negative weighting (`train.particles.cross_class_neg_weight`) |
| `v2-gate-3c-passed` | CLI config snapshot: checkpoint loads no longer leak into each other |
| `v2-gate-3d-passed` | softmax class head with implicit background (`train.particles.class_head`) |
| `v2-gate-4-passed` | v2 predict (per-class files), calibration, benchmarks, README |
| `v2-gate-4b-passed` | k-fold training, out-of-fold calibration, fold ensembles (`experiments/run_cv.sh`) |
| `v2-gate-5-passed` | SSL v2 (wedge/scale-aware views, masked recon, contrastive, relative scale) — **contrastive parts broken, see Gate 5b** |

Fast suite: 632 passed, 27 skipped (~5 min). Environment:
`source /home/pranav/Desktop/pranav/software/miniconda3/etc/profile.d/conda.sh && conda activate tomocpt`
(RTX A6000). The base worktree `../tomocpt-base` (detached at `v2-gate-0-base`) is used by golden
generators; leave it. Nothing is running on the GPU (SSL benchmark stopped 31 Oct).

## 2. How to work (user preferences)
- Usage limits are tight: limit subagents. A Test Author agent writes frozen tests; small
  implementations are done directly by the orchestrator; a big one may use one implementer agent.
- Keep context lean (short tool outputs). Stop at gate boundaries; new gates in a fresh session.
- Check loss traces / embeddings of anything trained, not only the tests (Gate 5 lesson).
- Commits and `v2-*` tags are fine; pushing `v2-dev` + `v2-*` tags is authorised by the user;
  never touch `main`.

## 3. Next work, in order (user decision 31 Oct 2026)
1. **Gate 5b** — `.claude/agents/05b_AGENT_SSL_FIX.md` (spec written, tests not yet). SSL v2
   global embedding collapsed (all cosine sims 0.9999996, pooled `hidden[4]` constant); dense
   task was detached (`dense_grad_scale = 0`) because views share the noise realisation. Fix:
   independent per-view low-pass + noise, VICReg terms, negative queue, `recon_weight` 1,
   full dense gradient, collapse monitoring. Then re-run `STEPS=ssl experiments/run_benchmarks.sh`.
2. **Gate 6** — `.claude/agents/06_AGENT_FILAMENTS.md` (amended 30–31 Oct: spline fit with smoothing
   and spacing in Å, tilt/psi as RELION helical priors with flip ratio 0.5, one output file per
   class, labels from STAR/IMOD coordinates via the same spline fit, optional tube masks from
   splines, Gaussian tube target identical to the particle profile across the centreline).
   **Data downloaded** to `data/filaments/{10477,10513,10490}/` (tomograms + masks per object,
   `manifest.json`; 332 runs, 146 GB, all mask shapes match; DS-10513 labels not ground truth,
   held-out test only). Particle points of DS-10490 can be added with
   `scripts/download_portal_data.py --coords-only`. Fetch script: `experiments/fetch_filament_data.py`.
3. **Gate 7** — `.claude/agents/07_AGENT_DISCRIMINATIVE.md` (supervised dense contrastive with hard
   negatives incl. membrane masks, focal value term; particles and filaments). After Gate 6.

## 4. Human decisions (details in `gate_status.json` notes)
- Gate 3: C4 Hessian tolerance < 15 %; C13 frozen at seed 0 (known flake with sigmoid, k = 0).
- Gate 4: one output file per class in every format; `imodmodel==0.1.0`; v1 unpad axis-order fix.
- Gate 3b: k = 0.25 tested on CZII: no gain; use k = 0 (softmax makes it unnecessary).
- Gate 3d softmax, Gate 4b k-fold CV, Gate 5b, Gate 6 amendments, Gate 7: all human-requested.
- CZII diameters: public values `1:130,2:180,3:260,4:300,5:120,6:270` Å.
- Target regime: data-poor, non-denoised tomograms ("that's where the real prize is").

## 5. CZII results (details in `experiments/czii_v2_results.md`)
- 5-fold CV, softmax, early stopping (best epochs 11–23): Kaggle-weighted out-of-fold F4
  **0.631** (Kaggle 1st place 0.788 on its hidden test set). Per class OOF F4: beta-amylase 0.36,
  beta-galactosidase 0.49, thyroglobulin 0.64, ribosome 0.58, apo-ferritin 0.76, VLP 0.83.
  Output: `benchmarks/out/czii_bench_20261029_2237/cv/` (fold models, `ensemble.json`).
- Localisation diagnostics (`experiments/diag_cv_localisation.py`): centring is accurate (median
  12–25 Å; F4 barely changes from 0.5 R to 1.0 R); the gap is detection/recall. Touching
  apo-ferritins are lost (79 % of misses touch a neighbour vs 51 % of all). **Cheap check pending:**
  rerun with cross-class suppression off and with a smaller peak spacing (~15 min each, no training).
- About 95 % of false picks are confident picks on real unlabelled densities (`experiments/diag_czii_v2.py`).
- POPSICLE (Schwartz, Kimanius et al. 2026) context: segmentation baselines 0.40–0.65, Octopi 0.763.

## 6. Benchmark entry points
`experiments/run_benchmarks.sh` (STEPS=cv,ssl; CLASS_HEAD softmax; EPOCHS 60; SSL v1 30 / v2 90
epochs; resumable with the same OUT), `experiments/run_cv.sh`, `experiments/eval_czii_v2.py`.
