# Example 4: fine-tuning a trained model on a new dataset with few labels

**Situation.** You have a trained v2 picker (for example from example 2 or 3) and a new dataset
(another sample, microscope or voxel size) with only a few annotated tomograms.

**Commands.**
```bash
tomocpt prepare_data --config_file config.yaml
tomocpt train --config_file config.yaml          # fine-tunes from train.fine_tune_from
tomocpt predict --config_file config.yaml
```

**Settings.**
- `train.fine_tune_from`: a new run from the source weights. The optimizer starts fresh with
  differential learning rates (Swin encoder x0.01, encoder blocks x0.1, decoder and output x1), so
  the backbone moves slowly. `resume_from` is for continuing an interrupted run instead.
- What carries over: the run uses the checkpoint's config with the values you changed from the
  defaults applied. The network width must match. Classes with the same class number keep their
  output channel; a new class starts from the mean of the old channels.
- `prepData.class_diameters`: use the source model's class numbers for the same species. Add
  `prepData.class_annotated` if the new labels cover only some of the species.
- `train.val_stems`: with few tomograms a random split is a lottery; fix the validation tomogram.
- Early stopping recipe and `weight_averaging: ema`: docs/tuning.md, section 5.
- From a v1 or self-supervised checkpoint, every class starts from its single output channel. To
  keep the source behaviour while adapting, see `train.use_distillation` (not benchmarked in v2).

**After the run.** There is no few-label benchmark for v2 yet: compare against the source model
on the new validation tomogram (predict with both and count picks against the labels). With one
validation tomogram the calibrated thresholds are noisy; set `infer.class_thresholds` by eye if
needed (example 5).
