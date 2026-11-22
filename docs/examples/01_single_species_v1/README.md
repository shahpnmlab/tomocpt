# Example 1: one species, legacy pipeline (v1)

**Situation.** You have tomograms and a STAR file with the centres of one species (here a 300 A
ribosome) and want a picker for it, or you want to reproduce a model made with TomoCPT before v2.
For several species, or tomograms at different voxel sizes, use example 2 instead.

**Commands.**
```bash
tomocpt prepare_data --config_file config.yaml   # label MRC per tomogram + tracking CSV
tomocpt train --config_file config.yaml          # chunks, trains, saves checkpoints
tomocpt predict --config_file config.yaml        # tomocpt_coords.star in predictions_dir
```

**Settings.** Pipeline v1 is the default (`train.data_pipeline_version: 1`), so nothing else is
needed. Each tomogram is rescaled so the particle is 20 voxels across
(`prepData.desired_particle_pixel_size`). `train.n_epochs` is raised from 10 because early stopping
(`train.early_stop_patience` 10) decides when to stop. `infer.length` is required for v1
checkpoints. To start from the public pretrained model, add `fine_tune_from` (example 4).

**After the run.** TensorBoard (`--launch_tensorboard`) shows `val_loss_smooth`; check that it
flattened before the run stopped. Open a few tomograms with the STAR output and adjust
`infer.confidence_threshold` (0.3) if there are too many or too few picks; re-picking without
inference is shown in example 5.
