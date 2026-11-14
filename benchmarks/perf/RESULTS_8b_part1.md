# Gate 8b.1-8b.3 results, 9 Nov 2026

RTX A6000, `benchmarks/perf/run_8b.sh` (40 warm-up + 200 timed steps, B=16, real chunks, recipe
precision bf16-mixed; synthetic C=134: 5 + 20 steps). Raw JSON `benchmarks/out/perf_8b/` (not committed).
Each row adds one change to the row above.

| Change | CZII ms/step | Filaments ms/step | CZII peak GB | Equivalence |
|---|---|---|---|---|
| defaults (= 8a baseline 1218 / 1098) | 1214 | 1096 | 13.1 | — |
| `network.SwinUNETR.use_checkpoint=False` (8b.1) | 1050 (1.16×) | 927 (1.18×) | 26.8 | losses bit-identical over 240 steps |
| `particles.loss_checkpoint=False`, `particles.hessian_block=0` (8b.2) | 968 | 902 | 28.9 | max rel. loss diff 1.3e-7 |
| `train.pin_memory=True` (8b.3; data wait 40 → 18 ms) | **945 (1.28×)** | **891 (1.23×)** | 28.9 | max rel. loss diff 1.3e-7 |

C = 134, B = 4, sigmoid (8b.2):
| hessian_block | loss_checkpoint | ms/step | peak GB |
|---|---|---|---|
| 1 | True (today) | 1449 | 6.3 |
| 1 | False | 1320 | 12.3 |
| 0 | True | 727 | 18.7 |
| 0 | False | **642 (2.26×)** | 19.5 |

G8b-C8 (loss overlay): the step benchmark records the loss of each timed step with the same seed and
data; the 8b runs overlay the defaults to 1.3e-7 relative (200 steps, not the 300 of the spec).

Not done in 8b.3: `panel_every_n_steps` / batched per-class validation loss (no step-time cost seen in
the phase timers; panels do not dominate any phase). EMA via `_foreach` is bit-identical; its gain is
inside the 10 ms optimiser phase (not separable).

Memory: turning Swin checkpointing off doubles peak memory at B = 16 (13 → 27-29 GB). Fits the A6000;
a 24 GB GPU would need B ≤ 8 or checkpointing on. This is why the defaults are not flipped yet
(human decision).

## 8b.4 pre-check (SDPA backend for head_dim 12)
torch 2.5.1, bf16, additive bias + dropout 0.3, stage-1 shape (2000 windows × 3 heads × 343 tokens × 12):
eager 89 ms fwd+bwd; SDPA flash / memory-efficient **refuse head_dim 12** (fall back to math: 128 ms,
slower). Zero-padding q, k, v to head_dim 16 (exact: zero columns add nothing to q·k and the extra
output columns are dropped) enables the memory-efficient kernel: **31 ms (2.9×)**; dropout 0 vs eager:
max abs 0.039, cosine 0.99997 (bf16). → 8b.4 goes ahead with the padding.
