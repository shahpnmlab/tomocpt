# Gate 8b.5 (regional torch.compile): dropped, 9 Nov 2026

`benchmarks/perf/compile_probe.py` (fs=36, 6 classes, B=16, bf16 autocast, Huber loss, RAdam,
fused attention of 8b.4; `module.compile()` on blocks in place; A6000, torch 2.5.1):

| Variant | Swin checkpointing | first step (compile) | steady step | peak |
|---|---|---|---|---|
| eager | off | 1.4 s | 802 ms | 26.0 GB |
| 22 blocks (Swin + UNETR), cold cache | off | **1407 s** | 622 ms (1.29×) | 22.1 GB |
| 22 blocks, warm inductor cache | off | **538 s** | 619 ms | 22.1 GB |
| 8 Swin blocks only, cold cache | off | **1400 s** | 645 ms (1.24×) | 22.0 GB |
| eager | on | 1.6 s | 975 ms | 13.0 GB |
| 22 blocks | on | crash (`AssertionError: Node add_453 was invalid, but is output`) | — | — |

Decision: dropped. 9-23 min of compile per training process (each CV fold is a new process) against
~180 ms per step only pays after ~3000-7700 steps; with checkpointing on it crashes; no reliable
automatic rule exists and opt-in flags are not wanted (human decision 9 Nov 2026). Revisit in 8f
if a torch upgrade compiles the Swin blocks much faster.
