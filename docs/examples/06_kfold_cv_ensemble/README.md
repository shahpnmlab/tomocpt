# Example 6: k-fold cross-validation with a fold ensemble

**Situation.** Few annotated tomograms (CZII: 7). A single train/validation split leaves 1-2
tomograms for validation, so thresholds and comparisons are noisy. k-fold CV uses every tomogram
once for out-of-fold calibration and gives an ensemble of k models.

**Commands.**
```bash
tomocpt prepare_data --config_file config.yaml
tomocpt train --config_file config.yaml     # all folds, then ensemble.json and cv_summary.md
tomocpt predict --config_file config.yaml   # the fold ensemble (infer.weights = ensemble.json)
```
Running `train` again skips finished folds and restarts unfinished ones.

**Settings.**
- `train.cv_folds: 5`, `train.cv_all_folds: true`: one model per fold, each in its own process
  (`<model_dir>/<experiment_name>/folds/fold<k>/`); thresholds calibrated on the pooled
  out-of-fold predictions; `ensemble.json` and the CV report in `<model_dir>/<experiment_name>`.
- `train.cv_class_weights`: class weights of the report's mean F-beta only (here the CZII
  challenge weights); training is unchanged.
- The rest is example 2.

**After the run.** `cv_summary.md`: per class the out-of-fold F-beta (and per fold), precision,
recall, distance of matched picks to the true centres, and the predicted peak profile against the
target. This is the number to compare settings with on small data (CZII v2 defaults: OOF F4 0.741).
Cost: k trainings (CZII, 5 folds: ~30 h on one GPU). With many tomograms a fixed split
(`train.val_stems`) and a separate test set are enough.
