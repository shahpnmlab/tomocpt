# Gate 8 tests

Spec: `.claude/agents/08_AGENT_PERFORMANCE.md`. Written and frozen by the orchestrator acting as TA
(usage-frugal, as Gates 3b, 3d, 4b, 4c), 9 Nov 2026, on branch `v2-feat/perf`.

## 8a (frozen 7cb10a3)
| Criterion | Test |
|---|---|
| G8a-C1 JSON keys, deterministic CPU loss sequence | `test_g8a_harness.py::test_G8a_C1_step_json_keys_and_determinism` |
| G8a-C1 CLI writes the JSON | `test_g8a_harness.py::test_G8a_C1_step_cli_writes_json` |
| G8a-C2 self-comparison is zero (JSON, NPZ) | `test_g8a_harness.py::test_G8a_C2_compare_self_is_zero` |
| G8a-C2 perturbed JSON / NPZ, missing / added keys | `test_g8a_harness.py::test_G8a_C2_compare_perturbed_json`, `..._npz` |
| G8a-C3 paths | orchestrator check (`git diff --name-only v2-dev...v2-feat/perf`), not a test |

All 8a tests fail on `v2-dev` (b8042e7): `benchmarks.perf` does not exist there (new behaviour).
Runtime on the development machine (CPU, tiny model): ~35 s.

## 8e.1 accumulator device (pulled forward before 8b, human decision 9 Nov 2026)
| Criterion | Test |
|---|---|
| `auto` counts the allocator cache (`memory_reserved - memory_allocated`) as free | `test_g8e1_accumulator.py::test_G8e1_auto_counts_cached_memory_as_free` |
| `cpu` / `cuda` / no-GPU settings unchanged | `test_g8e1_accumulator.py::test_G8e1_explicit_settings_unchanged` |
| Same decision after tomogram 1 filled the cache (slow, GPU) | `test_g8e1_accumulator.py::test_G8e1_same_decision_after_cache_fill` |
| CPU vs GPU accumulation, maps `atol 1e-6`, with and without TTA (slow, GPU) | `test_g8e1_accumulator.py::test_G8e1_cpu_vs_gpu_accumulation_maps` |

On the 8a code the first and third fail (the decision ignores the cache); the others pass (unchanged behaviour).

## 8b.1-8b.3 (training step, no behaviour change) and the gloo DDP check
| Criterion | Test |
|---|---|
| G8b-C1 new fields (`particles.hessian_block=1`, `train.pin_memory=False`, `train.prefetch_factor=None`), loss default, loader flags reach the `DataLoader`, `run.py` passes them | `test_g8b_step.py::test_G8b_C1_*` |
| G8b-C2 `use_checkpoint` on vs off bit-identical (3 SGD steps, dropout on, **one CPU thread**) | `test_g8b_step.py::test_G8b_C2_swin_checkpointing_is_bit_identical` |
| G8b-C3 `hessian_block` 0/2/3 vs 1: C = 1, 3, 7; sigmoid / softmax; checkpoint on / off; class weights + cross-class weight; focal + core weight | `test_g8b_step.py::test_G8b_C3_hessian_block_matches_class_loop` |
| G8b-C3 zero gradient path for a class unavailable in the batch; nothing available | `test_g8b_step.py::test_G8b_C3_unavailable_*`, `..._no_class_available_*` |
| G8b-C3 the model passes `hessian_block` | `test_g8b_step.py::test_G8b_C3_model_passes_hessian_block` |
| G8b-C3 no host sync in the loss (slow, GPU, `torch.cuda.set_sync_debug_mode("error")`) | `test_g8b_step.py::test_G8b_C3_loss_has_no_host_sync` |
| 8b.3 EMA update bit-identical to the per-tensor loop | `test_g8b_step.py::test_G8b_ema_update_is_bit_identical_to_the_tensor_loop` |
| G8b-C6 every parameter gets a gradient with the 8b.1-8b.3 options, contrastive and offset heads on | `test_g8b_step.py::test_G8b_C6_options_on_every_parameter_gets_a_gradient` |
| G8b-C7 / 3.6 gloo: 2 ranks vs 1 process gradients (per tensor `1e-5`); 2-epoch fit saves one checkpoint; defaults and 8b options (slow, ~90 s, CPU) | `test_g8_ddp_gloo.py` (worker: `ddp_gloo_worker.py`) |

Rulings made while writing (before freezing):
* C2 runs with one CPU thread: Swin's `relative_position_bias_table` gradient (indexed accumulation)
  is not bit-reproducible run to run with several threads, with or without checkpointing (measured).
* C7 tolerance is per tensor (max |difference| <= 1e-5 max |g|, and the norm), not elementwise: an
  elementwise relative bound fails on near-zero elements when the batch is summed in a different
  order (measured worst case 5.5e-6 of the tensor maximum).
* `Data(..., pin_memory=False, prefetch_factor=None)`; the factor is passed only with workers.
* `panel_every_n_steps` and the batched per-class validation loss get tests only if the step
  benchmark shows the panels cost time (spec 8b.3: "each kept only if step.py shows a gain").

On the 8a code: C1, C3 and the no-sync test fail (new behaviour); C2, EMA, C6 and C7 pass
(they pin today's behaviour).

### Amendment (human decision 9 Nov 2026: "all speed up changes shouldn't be opt-ins"; CLI too big)
No new CLI options in Gate 8. `train.particles.hessian_block`, `train.pin_memory`, `train.prefetch_factor`
removed: the model picks the block with `multiclass_loss.auto_hessian_block` (4 GiB per call), batches
are pinned whenever training uses a GPU, `particles.loss_checkpoint` defaults to `False`. Swin
checkpointing stays on by default (memory; set off in the recipes). Amended tests:
`test_G8b_C1_new_fields_and_defaults`, `test_G8b_C1_run_passes_loader_flags`,
`test_G8b_C3_model_passes_hessian_block`, `test_G8b_C6_*` (no longer sets the removed field),
`test_g8_ddp_gloo.py` option set "8b".

## 8b.4 fused window attention (on by default; no config field)
| Criterion | Test |
|---|---|
| G8b-C4 same state-dict keys / shapes; checkpoints interchange both ways (strict) | `test_g8b_attention.py::test_G8b_C4_state_dict_*`, `..._tiny_model_forward_equals_eager_and_checkpoints_interchange` |
| G8b-C4 CPU fp32: output `atol 1e-5`, gradients (input and every parameter) `atol 1e-5 rtol 1e-4`, shifted and unshifted | `..._cpu_fp32_output_and_grads_equal_eager` |
| G8b-C4 GPU bf16: `atol 2e-2`, cosine >= 0.9999 (slow) | `..._gpu_bf16_close_to_eager` |
| Dropout only in training | `..._dropout_is_applied_in_training_only` |
| G8b-C6 every parameter gets a gradient (attention dropout on, contrastive + offset heads); bias tables non-zero | `..._fused_attention_every_parameter_gets_a_gradient` |
All fail on the code before 8b.4 (`tomocpt.networks.attention` does not exist).
Ruling (after freezing, before accepting): the CPU gradient comparison is per tensor (max |diff| <= 1e-5
max |g|, absolute sum 1e-4), not a flat `atol 1e-5`: gradients reach ~1e2 and differ by fp32
summation order (measured worst 5.4e-6 of the tensor maximum).
Decision (9 Nov 2026, after the full suite): the fused path runs on CUDA only. On the CPU,
`SDPAWindowAttention` calls MONAI's eager `forward`, so every CPU golden stays bit-identical (the CPU
fused path moved outputs by up to 5.3e-6 and failed 10 bit-exact golden tests; it gives no useful
speed). The CPU equivalence tests above now compare eager with eager; the fused numerics are covered
on the GPU by `test_g8b_attention_gpu.py` (float32: output `atol 1e-5`, gradients per tensor
`1e-5 max|g|`; added, not a frozen test changed) and the bf16 test.

## 8e.1 peak finding (E0)
| Criterion | Test |
|---|---|
| G8e-C1 `peak_local_max_torch` identical peaks and order to the 8a version (`legacy_peaks.py`, frozen copy of e88922c) on maps with plateaus, quantised ties, borders, thousands of candidates, no candidates; CPU and GPU (slow) | `test_g8e1_peaks.py` |
Pass on the 8a code (they pin behaviour).
| 8e.1 `_prune_refined_peaks_gpu` identical kept peaks and order to the 8a version (`legacy_prune.py`); clusters, boundary-distance pairs, equal scores; CPU and GPU (slow) | `test_g8e1_prune.py` |
| G8e-C3 filament threshold scores identical serial vs parallel workers (2, 4, auto) | `test_g8e1_filament_scores.py` |
| G8e-C2 `grid_locations` equals TorchIO locations (odd shapes, overlaps 0/16/30/32; 31 replaced: TorchIO rejects odd overlaps and predict makes them even); GPU patch grid maps and offsets bit-identical to the TorchIO path (TTA, offsets, float16 accumulation; slow) | `test_g8e1_patch_grid.py` |
