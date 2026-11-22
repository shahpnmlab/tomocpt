"""Gate 10, sections 3 and 4: generated option reference, tuning guide, worked examples.

Criteria:
* C16  ``python -m tomocpt.configManager.option_docs --out <file>`` reproduces the committed
       ``docs/options.md`` (stale docs fail); every non-internal option is in it, no internal one.
* C17  ``docs/tuning.md`` covers the topics of the spec and names only existing options.
* C18  the six examples exist (config YAML + README); every example YAML loads into ``MainConfig``
       with no unknown keys and valid values.
* C19  example 3 (DS-10490) has the train settings of ``experiments/run_boss_ds10490.sh``
       (``--train_set`` values and driver arguments) plus the recipe of ``benchmarks/filaments.py``.
"""
import os
import re
import subprocess
import sys
import warnings
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
SECTIONS = ("shared", "prepData", "train", "infer")
EXAMPLES = (
    "01_single_species_v1",
    "02_multi_species_v2",
    "03_ds10490_particles_filaments",
    "04_fine_tune_few_labels",
    "05_predict_repick_thresholds",
    "06_kfold_cv_ensemble",
)


def _leaves(obj, prefix=""):
    out = []
    for f in fields(type(obj)):
        v = getattr(obj, f.name)
        if is_dataclass(v):
            out.extend(_leaves(v, f"{prefix}{f.name}."))
        else:
            out.append(prefix + f.name)
    return out


def _options():
    from tomocpt.configManager.option_levels import option_level
    from tomocpt.mainConfig import MainConfig

    m = MainConfig()
    return {(s, p): option_level(s, p) for s in SECTIONS for p in _leaves(getattr(m, s))}


# --- C16 --------------------------------------------------------------------------------

def test_G10_C16_options_md_is_up_to_date(tmp_path):
    out = tmp_path / "options.md"
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="")
    res = subprocess.run([sys.executable, "-m", "tomocpt.configManager.option_docs", "--out", str(out)],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    committed = DOCS / "options.md"
    assert committed.is_file(), "docs/options.md missing"
    assert out.read_text() == committed.read_text(), (
        "docs/options.md is stale: python -m tomocpt.configManager.option_docs --out docs/options.md")
    res2 = subprocess.run([sys.executable, "-m", "tomocpt.configManager.option_docs"],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert res2.returncode == 0 and res2.stdout == committed.read_text()


def test_G10_C16_options_md_lists_user_settings_only():
    text = (DOCS / "options.md").read_text()
    names = set(re.findall(r"^\|\s*`([^`]+)`", text, flags=re.M))
    for (s, p), lv in _options().items():
        key = f"{s}.{p}"
        if lv == "internal":
            assert key not in names, key
        else:
            assert key in names, key


# --- C17 --------------------------------------------------------------------------------

TUNING_TOPICS = (
    "data_pipeline_version", "band_min_vox", "band_max_vox", "apix_override", "class_diameters",
    "class_annotated", "label_border_policy", "negative_masks", "class_head", "softmax_scope",
    "chunk_sampling", "repeat_factor_threshold", "repeat_factor_max", "batch_size", "contrastive_weight",
    "contrastive_warmup_steps", "bank_max_age", "embed_dim", "embed_ce_weight", "prototype_head",
    "optimizer.lr", "lr_warmup_steps", "early_stop_rel_delta", "early_stop_patience",
    "lr_plateau_match_early_stop", "weight_averaging", "schedule_free", "lr_scaling", "use_mw_aug",
    "use_fourier_aug", "rotation_mode", "sampling", "augment_device", "samples_per_epoch",
    "n_cpus_for_dataloading", "accumulate_device", "fine_tune_from", "resume_from", "calibration_beta",
    "calibration_min_particles", "class_thresholds", "peak_spacing_fraction", "cross_class_nms", "use_tta",
    "filament_spacing_ang", "cv_folds", "cv_all_folds", "val_stems",
)


def test_G10_C17_tuning_covers_the_topics():
    text = (DOCS / "tuning.md").read_text()
    missing = [t for t in TUNING_TOPICS if t not in text]
    assert not missing, missing
    assert "0.741" in text  # CZII CV evidence is cited


def test_G10_C17_tuning_names_only_existing_options():
    text = (DOCS / "tuning.md").read_text()
    known = {f"{s}.{p}" for s, p in _options()}
    known |= {k.rsplit(".", i)[0] for k in known for i in range(1, k.count("."))}  # sub-sections
    cited = set(re.findall(r"`((?:prepData|train|infer|shared)\.[A-Za-z0-9_.]+)", text))
    cited = {c.rstrip(".") for c in cited}
    unknown = sorted(c for c in cited if c not in known)
    assert not unknown, unknown


# --- C18 --------------------------------------------------------------------------------

def _check_keys(d, obj, path=""):
    """Unknown keys of the YAML dict ``d`` for the dataclass ``obj``."""
    names = {f.name: f for f in fields(type(obj))}
    bad = []
    for k, v in d.items():
        here = f"{path}{k}"
        if k not in names:
            bad.append(here)
            continue
        sub = getattr(obj, k)
        if is_dataclass(sub):
            if not isinstance(v, dict):
                bad.append(here + " (expected a section)")
            else:
                bad.extend(_check_keys(v, sub, here + "."))
    return bad


def _check_values(d, obj, path=""):
    from tomocpt.configManager.configManager import ConfigurationError, parse_value

    bad = []
    names = {f.name: f for f in fields(type(obj))}
    for k, v in d.items():
        sub = getattr(obj, k)
        if is_dataclass(sub):
            bad.extend(_check_values(v, sub, f"{path}{k}."))
        elif v != "???":
            try:
                parse_value(names[k].type, v, path + k)
            except ConfigurationError as e:
                bad.append(str(e))
    return bad


def _example_yaml(name):
    d = DOCS / "examples" / name
    ymls = sorted(d.glob("*.yaml"))
    assert ymls, f"no YAML in {d}"
    return ymls


@pytest.mark.parametrize("name", EXAMPLES)
def test_G10_C18_example_loads(name, tmp_path, monkeypatch):
    from tomocpt.configManager.configManager import coerce_config_types, merge_config
    from tomocpt.mainConfig import MainConfig

    d = DOCS / "examples" / name
    assert (d / "README.md").is_file()
    for y in _example_yaml(name):
        data = OmegaConf.to_container(OmegaConf.load(y))
        assert set(data) <= set(SECTIONS), (y, set(data))
        m = MainConfig()
        assert not _check_keys(data, m), (y, _check_keys(data, m))
        assert not _check_values(data, m), (y, _check_values(data, m))
        merge_config(OmegaConf.load(y), m)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            coerce_config_types(m)
        for key, value in _flatten(data).items():
            for token in re.findall(r"(?:^|[:,])(/[^,:]*)", value) if isinstance(value, str) else ():
                assert token.startswith("/path/to"), (y, key, token)  # placeholders only


# --- C19 --------------------------------------------------------------------------------

def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


def test_G10_C19_example3_matches_the_boss_run():
    from benchmarks.common import TrainSettings, _train_section
    from benchmarks.filaments import _apply_train_overrides, apply_recipe
    from tomocpt.configManager.configManager import parse_value
    from tomocpt.mainConfig import MainConfig

    script = (REPO / "experiments" / "run_boss_ds10490.sh").read_text()
    epochs = int(re.search(r"\bEPOCHS=(\d+)", script).group(1))
    spe = re.search(r"\bSPE=(\d+)", script).group(1)
    batch = int(re.search(r"--batch_size (\d+)", script).group(1))
    n_cpus = int(re.search(r"--n_cpus (\d+)", script).group(1))
    band_max = float(re.search(r"--band_max_vox (\S+)", script).group(1))
    contrastive = float(re.search(r"--contrastive_weight (\S+)", script).group(1))
    sets = [s.replace("$SPE", spe) for s in re.findall(r"--train_set (\S+)", script)]
    assert len(sets) >= 10

    settings = TrainSettings(n_epochs=epochs, batch_size=batch, n_cpus=n_cpus, class_head="softmax")
    train = apply_recipe(_train_section(Path("/w"), settings, 2, conditioning=True))
    train["val_stems"] = "placeholder"
    train["scale"] = {"band_max_vox": band_max}
    train["discriminative"] = {"contrastive_weight": contrastive}
    _apply_train_overrides(train, sets)
    expected = _flatten(train)
    for k in ("training_data_dir", "chunks_dir", "model_dir", "experiment_name", "use_gpus", "USE_CUDA_FOR_DATA",
              "network.TORCH_FLOAT_PRECISION", "val_stems"):
        expected.pop(k, None)

    (y,) = _example_yaml("03_ds10490_particles_filaments")
    data = OmegaConf.to_container(OmegaConf.load(y))
    got = _flatten(data["train"])
    defaults = MainConfig().train
    for key, value in expected.items():
        *head, last = key.split(".")
        obj = defaults
        for p in head:
            obj = getattr(obj, p)
        ftype = next(f.type for f in fields(type(obj)) if f.name == last)
        want = parse_value(ftype, value, key)
        have = parse_value(ftype, got[key], key) if key in got else getattr(obj, last)
        assert have == want, (key, have, want)
    assert got.get("val_stems"), "the boss run uses a fixed validation split"

    prep = data["prepData"]
    assert prep["multiclass"] is True
    assert prep["label_border_policy"] == "clip"
    assert prep.get("filament_coords") and prep.get("negative_masks")
    assert int(data["infer"]["calibration_min_particles"]) == TrainSettings(n_epochs=1).calibration_min_particles
