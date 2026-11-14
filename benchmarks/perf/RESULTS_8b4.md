# Gate 8b.4 results (fused window attention), 9 Nov 2026

`tomocpt/networks/attention.py`: MONAI's `WindowAttention` with the same parameters, `forward` via
`scaled_dot_product_attention` (memory-efficient kernel) on CUDA; head dim zero-padded 12 → 16; bias
cast and made contiguous before broadcasting, shifted windows folded into the heads so the mask is
never copied per window. CPU: MONAI's eager attention (goldens unchanged). On by default, no config.

Attention module alone (A6000, bf16, B = 16, forward + backward, attention dropout 0.3):
| Stage | eager | fused |
|---|---|---|
| 1, unshifted / shifted | 104 / 114 ms, 11.8 GB | 50 / 55 ms, 4.4 GB |
| 2, unshifted / shifted | 46 / 50 ms, 5.1 GB | 22 / 24 ms, 1.9 GB |
(The first version cast the bias after broadcasting it, a 1.4 GB copy per block, and passed a
non-contiguous bias: no gain in the step. Fixed before these numbers.)

Training step (`benchmarks/perf/run_8b4.sh`; 40 + 200 steps, B = 16, real chunks, current defaults):
| Recipe | 8a | defaults (Swin checkpointing on) | Swin checkpointing off | per epoch (ckpt off) |
|---|---|---|---|---|
| CZII | 1214 ms, 13.1 GB | **818 ms (1.48×)**, 9.9 GB | **753 ms (1.61×)**, 17.4 GB | 5.1 min (406 steps) |
| Filaments | 1096 ms, 13.0 GB | **769 ms (1.43×)**, 7.4 GB | **697 ms (1.57×)**, 14.9 GB | 12.3 min (1059 steps; was 19.4) |
Loss overlay vs 8a (mean of the last 50 of 200 steps): CZII 0.9817 / 0.9803, filaments 0.9034 / 0.9038.
Per-step losses differ by up to 1.7 % (bf16 attention, chaotic over 240 steps), means agree.
