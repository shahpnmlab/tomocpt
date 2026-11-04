# 04d_AGENT_EXACT_COORDS.md — Gate 4d: exact (sub-voxel) particle coordinates in pipeline v2 (TODO 3b.1)

Added 4 Nov 2026 by human decision: "for v2, I don't want it to be an opt-in. This needs to be
default because of the chunking and splatting strategy."

## Why
`prepare_data` writes one coordinate CSV per tomogram (`<dataset>/coords/<tomo>.csv`, columns
`x, y, z, class_number, diameter_ang`), which pipeline v2 chunking reads, resamples to working
voxels as floats and splats into float targets (`labels.helpers.render_targets`). The coordinates
reaching the CSV are already truncated to whole voxels by the legacy readers:
`labels/star.py` `get_shifted_scaled_coordinates` (`group[xyz].astype(int)`) and `labels/mod.py`
`Mod.get_coordinates` (`coords.astype(int)`). On CZII the truncation moves every particle by
0.5 voxel per axis on average (0.94 voxel in 3D), a systematic bias towards the origin that the
v2 targets reproduce faithfully. It is largest relative to small classes (apo-ferritin match
radius about 3 voxels) and is a candidate cause of the flattened, widened peaks seen in the
peak-shape diagnostic (TODO 3a.3).

Pipeline v1 draws its label MRCs at whole voxels (`_paint_sphere` with integer boxes); v1 stays
unchanged (invariant 1).

## Design
- `Star.get_shifted_scaled_coordinates(particle_data, tomogram_pixel_size, exact=False)` and
  `get_shifted_scaled_coordinates_with_classes(..., exact=False)`: with `exact=True` the shifted
  and scaled coordinates are returned as float64 (no truncation). Default unchanged.
- `Mod.get_coordinates(exact=False)`: with `exact=True` float64 model coordinates. Default unchanged.
  IMOD and STAR coordinates are used as stored (no half-voxel shift).
- `prepare_picking_star`, `prepare_picking_imod`, `prepare_picking_star_multiclass` read exact
  coordinates once, write them to the v2 coordinate CSV **unrounded**, and pass the legacy
  truncated values (`astype(int)` of the same floats) to the tracking CSV and
  `create_particle_masks`. So v1 label MRCs and tracking CSVs are byte-identical to before.
- No config field: exact coordinates are the v2 default. Coordinate CSVs written before this gate
  hold whole voxels; rerun `prepare_data` (and rechunk) to use exact coordinates.

## Allowed write paths
`tomocpt/labels/{star,mod,helpers}.py`, `README.md` (one line), `.claude/reports/gate_04d_report.json`.

## Pass criteria
- **G4d-C1** `get_shifted_scaled_coordinates` default equals the legacy integers; `exact=True`
  gives `(coords - shifts) * scale` as floats (atol 1e-9), shifts from `rlnOrigin*Angst` included,
  with and without classes.
- **G4d-C2** `Mod.get_coordinates(exact=True)` gives the stored floats; default the legacy integers.
- **G4d-C3** `prepare_data` (multiclass STAR, single-class STAR, IMOD) writes coordinate CSVs equal
  to the exact coordinates (atol 1e-6, so fractional parts survive), while label MRCs and tracking
  CSVs equal those written from the truncated coordinates (the legacy outputs).
- **G4d-C4** v2 chunk points of a tomogram prepared this way equal the exact coordinates mapped to
  working voxels (`points_to_working`), not the truncated ones.
- All earlier tests pass; goldens built through `prepare_data` with fractional coordinates are
  regenerated only with a recorded human ruling.
