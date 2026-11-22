# Example 5: prediction only, re-picking from saved maps, per-class thresholds

**Situation.** You have a trained model (a checkpoint, or a fold ensemble from example 6) and want
coordinates for new tomograms, then tune the thresholds without running the network again.

**Commands.**
```bash
tomocpt predict --config_file config.yaml                       # network + peaks; maps saved
# re-pick from the saved maps with other thresholds (no network pass):
tomocpt predict --config_file config.yaml --skip_inference --class_thresholds "ribosome:0.5"
```

**Settings.**
- `infer.save_prediction_confidence_map: true`: writes `<tomogram>_<class>.mrc` per class (plus a
  heatmap and a class map). Needed for `--skip_inference`, which re-reads them; `infer.weights` is
  still needed then, for the class table.
- `infer.class_thresholds`: per-class thresholds by class name; classes not named keep the
  calibrated threshold, or `infer.confidence_threshold` (0.3) if the model is uncalibrated.
- `infer.predictions_coord_format`: `relion_31` (default), `relion_50`, `warp` or `imod`; one file
  per class.
- Other options worth knowing: `infer.classes` (pick a subset), `infer.peak_spacing_fraction`
  (0.5 x diameter between picks of one class), `infer.cross_class_nms` (off; turn on only for
  sigmoid-head models), `infer.accumulate_device` / `accumulate_dtype` (GPU memory with many
  classes), `infer.apix_override` (wrong MRC header). See docs/tuning.md, section 9.

**After the run.** Raise a class's threshold when its false positives are background, lower it
when particles are missed. Calibrated thresholds maximise F4 (recall weighs 4x precision); for a
clean particle set you usually want higher thresholds than the calibrated ones.
