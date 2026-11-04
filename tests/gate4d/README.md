# Gate 4d tests (exact sub-voxel coordinates in pipeline v2; TODO 3b.1)

Spec: `.claude/agents/04d_AGENT_EXACT_COORDS.md`.

| Criterion | Tests |
|---|---|
| G4d-C1 STAR reader | `test_g4d_coords.py::test_G4d_C1_*` |
| G4d-C2 IMOD reader | `test_g4d_coords.py::test_G4d_C2_*` |
| G4d-C3 prepare_data outputs | `test_g4d_coords.py::test_G4d_C3_*` (multiclass STAR, single-class STAR, IMOD) |
| G4d-C4 v2 chunk points | `test_g4d_coords.py::test_G4d_C4_chunk_points_use_exact_coordinates` |

The legacy reference for C3 is computed in the test: `create_particle_masks` and the tracking CSV
from the truncated (`astype(int)`) coordinates, which is what the code before Gate 4d wrote.
Phantom coordinates (`utils.phantoms`) are sub-voxel floats, so truncation is visible.
