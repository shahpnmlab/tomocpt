# Example 2: several particle species of different sizes (pipeline v2)

**Situation.** All species of interest are annotated in the same tomograms (one STAR file with
`rlnClassNumber`), with diameters from 120 to 300 A. One model and one `predict` run should pick
them all. This is the CZII 10440 setting used to develop v2.

**Commands.**
```bash
tomocpt prepare_data --config_file config.yaml   # one multi-class dataset, coordinate tables
tomocpt train --config_file config.yaml          # band rule, training, per-class threshold calibration
tomocpt predict --config_file config.yaml        # <name>_<class>.star per class
```

**Settings.**
- `prepData.multiclass`, `class_diameters`, `class_names`: one dataset with every species; the
  diameters (A) set the target sizes, the band and the matching tolerance. Use published diameters.
- `prepData.label_border_policy: clip`: particles touching the border stay positives.
- `train.data_pipeline_version: 2`: no fixed rescale; the tomograms are resampled only as far as
  the 10-32 voxel band needs (at 10 A/px all six species fit at native size).
- `train.particles.class_head: softmax`: fixes size confusion between similar species (see
  docs/tuning.md, section 2).
- `train.chunk_sampling: repeat_factor`: rare classes are seen more often.
- Early stopping recipe (`early_stop_rel_delta`, `lr_plateau_match_early_stop`) and
  `weight_averaging: ema`: docs/tuning.md, section 5.
- Not set because they are already the v2 defaults: on-the-fly crop sampling and GPU augmentation
  (`train.sampling: auto`), the Fourier and missing-wedge augmentation (`fourier_auto`), peak
  spacing 0.5 x diameter and no cross-class suppression in `predict`.

**After the run.** The calibrated per-class thresholds are stored in the checkpoint:
```bash
python -c "from tomocpt.predict.calibration import read_calibration as r; print(r('/path/to/weights.ckpt'))"
```
A class without a threshold had fewer than `infer.calibration_min_particles` (20) true particles in
the validation tomograms and uses `infer.confidence_threshold`. With few tomograms these numbers are noisy: use k-fold CV (example 6) to
compare settings. Per-class thresholds can be overridden at prediction time (example 5).
