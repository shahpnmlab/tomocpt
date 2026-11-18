# Gate 9b tests (class head uses the contrastive embedding)

Spec: `.claude/agents/09b_AGENT_EMBED_HEAD.md`. All CPU, no `slow` tests.

| Criterion | Tests |
|---|---|
| G9b-C1 defaults, module creation | `test_G9b_C1_*` |
| G9b-C2 embedding CE value, cosine map | `test_G9b_C2_*` |
| G9b-C3 training step, DDP (every parameter has a gradient), ramp, SupCon unchanged | `test_G9b_C3_*` |
| G9b-C4 prototype residual (zero init, particle channels only, softmax scope, contract) | `test_G9b_C4_*` |
| G9b-C5 level / availability masking | `test_G9b_C5_*` |
| G9b-C6 old checkpoint predict load, fine-tune with both options, round trip | `test_G9b_C6_*` |
| G9b-C7 config backfill | `test_G9b_C7_*` |
