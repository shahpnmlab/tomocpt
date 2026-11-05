# Gate 4f tests (weight averaging, centroid weight, sub-voxel picks, offset head; TODO 3b.3a, 3b.3b)

Spec: `.claude/agents/04f_AGENT_AVERAGING_SUBVOXEL.md`. Interface: docstring of `g4f_helpers.py`.

| Criterion | Tests |
|---|---|
| G4f-C1 defaults | full suite + `test_g4f_loss.py::test_G4f_C1_*`, `test_g4f_offsets.py::test_G4f_C8_off_*` |
| G4f-C2 EMA | `test_g4f_averaging.py::test_G4f_C2_*` |
| G4f-C3 plateau SWA | `test_g4f_averaging.py::test_G4f_C3_*` |
| G4f-C4 schedule-free combinations | `test_g4f_averaging.py::test_G4f_C4_*` |
| G4f-C5 core weight | `test_g4f_loss.py::test_G4f_C5_*` |
| G4f-C6 default fit is sub-voxel | `test_g4f_subvoxel.py::test_G4f_C6_*` |
| G4f-C7 offset targets | `test_g4f_offsets.py::test_G4f_C7_*` |
| G4f-C8 offset head | `test_g4f_offsets.py::test_G4f_C8_*` |
| G4f-C9 predict with offsets | `test_g4f_subvoxel.py::test_G4f_C9_*` |

Fits are tiny CPU runs of the Gate 3 v2 model with the averaging callback taken from
`_setup_callbacks` (other callbacks dropped: no progress bar in tests).
