"""Everyday command-line flags per command section (human decision 9 Nov 2026: a short CLI).

Only these fields (plus required fields, which have no default) become ``--flags``. Every other
setting is set in the YAML (``--config_file``; ``tomocpt init`` writes all of them) or as a
``key=value`` override on the command line, with dots for nested sections, e.g.
``tomocpt train --config_file c.yaml particles.class_head=softmax augmentation.flip_p=0.5``.
Old long flags of other fields still work for now (rewritten to ``key=value`` with a warning).
"""

CLI_FLAGS = {
    "prepData": (
        "raw_data_dir",
        "training_data_dir",
        "coordinate_files",
        "coordinate_file_type",
        "particle_length_ang",
        "class_id",
        "multiclass",
        "class_diameters",
        "class_names",
        "apix_override",
        "filament_coords",
    ),
    "train": (
        "model_dir",
        "training_data_dir",
        "chunks_dir",
        "experiment_name",
        "mode",
        "data_pipeline_version",
        "n_epochs",
        "batch_size",
        "use_gpus",
        "n_cpus_for_dataloading",
        "resume_from",
        "fine_tune_from",
        "cv_folds",
        "cv_fold",
        "cv_all_folds",
        "launch_tensorboard",
    ),
    "infer": (
        "tomogram_dir",
        "tomogram_file",
        "predictions_dir",
        "weights",
        "length",
        "confidence_threshold",
        "class_thresholds",
        "classes",
        "predictions_coord_format",
        "predictions_batch_size",
        "use_tta",
        "apix_override",
        "skip_inference",
    ),
}

HELP_EPILOG = (
    "Other settings: put them in the YAML given with --config_file (tomocpt init writes every "
    "setting with its default) or pass key=value after the flags, with dots for nested sections, "
    "e.g. particles.class_head=softmax augmentation.flip_p=0.5. Booleans: true/false; none clears "
    "an optional value."
)
