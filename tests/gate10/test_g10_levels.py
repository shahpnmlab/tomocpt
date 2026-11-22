"""Gate 10, section 1: option levels (``tomocpt.configManager.option_levels``) and help text.

Criteria:
* C1  ``LEVELS`` is ``("basic", "advanced", "diagnostic")``.
* C2  every leaf field of ``MainConfig`` under shared / prepData / train / infer has a level
      (one of ``LEVELS`` or ``"internal"``).
* C3  basic = ``cli_options.CLI_FLAGS`` of the section plus required fields (the copies of the
      ``shared`` fields in the sections are internal; ``shared``'s own fields are basic).
* C4  every non-internal field has a non-empty help string.
* C5  the 40 formerly unannotated fields are ``Annotated`` with help, same type and default.
* C6  diagnostic list: only leaf fields; includes ``prepData.write_filament_masks``.
* C7  unknown paths raise ``KeyError``; ``train.optimizer._target_`` is internal.
"""
from dataclasses import fields, is_dataclass
from typing import Annotated, Optional, Tuple, get_args, get_origin, get_type_hints

import pytest
import typer
from omegaconf import MISSING

SECTIONS = ("shared", "prepData", "train", "infer")


def _leaves(obj, prefix=""):
    out = []
    for f in fields(type(obj)):
        v = getattr(obj, f.name)
        if is_dataclass(v):
            out.extend(_leaves(v, f"{prefix}{f.name}."))
        else:
            out.append(prefix + f.name)
    return out


def _owner_field(section_obj, dotted):
    *head, last = dotted.split(".")
    obj = section_obj
    for p in head:
        obj = getattr(obj, p)
    f = next(f for f in fields(type(obj)) if f.name == last)
    return obj, f


def _help_of(section_obj, dotted):
    obj, f = _owner_field(section_obj, dotted)
    hint = get_type_hints(type(obj), include_extras=True)[f.name]
    if get_origin(hint) is not Annotated:
        return None
    for m in get_args(hint)[1:]:
        if isinstance(m, typer.models.OptionInfo):
            return m.help
    return None


def _all_paths():
    from tomocpt.mainConfig import MainConfig

    m = MainConfig()
    return [(s, p) for s in SECTIONS for p in _leaves(getattr(m, s))]


def test_G10_C1_levels():
    from tomocpt.configManager.option_levels import LEVELS

    assert tuple(LEVELS) == ("basic", "advanced", "diagnostic")


def test_G10_C2_every_field_has_a_level():
    from tomocpt.configManager.option_levels import LEVELS, option_level

    paths = _all_paths()
    assert len(paths) > 200
    bad = [(s, p, option_level(s, p)) for s, p in paths if option_level(s, p) not in (*LEVELS, "internal")]
    assert not bad, bad


def test_G10_C3_basic_is_cli_flags_plus_required():
    from tomocpt.configManager.cli_options import CLI_FLAGS
    from tomocpt.configManager.option_levels import option_level
    from tomocpt.mainConfig import MainConfig

    m = MainConfig()
    shared = {f.name for f in fields(type(m.shared))}
    for p in _leaves(m.shared):
        assert option_level("shared", p) == "basic", p
    for section in ("prepData", "train", "infer"):
        obj = getattr(m, section)
        required = set()
        for p in _leaves(obj):
            o, f = _owner_field(obj, p)
            if getattr(o, f.name) is MISSING:
                required.add(p)
        expected = (set(CLI_FLAGS[section]) | required) - shared
        got = {p for p in _leaves(obj) if option_level(section, p) == "basic"}
        assert got == expected, (section, sorted(got ^ expected))
        for name in shared & {p for p in _leaves(obj)}:
            assert option_level(section, name) == "internal", (section, name)


def test_G10_C4_every_user_setting_has_help():
    from tomocpt.configManager.option_levels import option_help, option_level
    from tomocpt.mainConfig import MainConfig

    m = MainConfig()
    missing = []
    for s, p in _all_paths():
        if option_level(s, p) == "internal":
            continue
        h = _help_of(getattr(m, s), p)
        if not h or not h.strip() or option_help(s, p) != h:
            missing.append(f"{s}.{p}")
    assert not missing, missing


# path, type, default
FORMERLY_UNANNOTATED = [
    ("prepData", "USE_CUDA_FOR_DATA", bool, True),
    ("prepData", "ALPHA_FOR_DROPPING_EMPTY_CUBES", float, 1),
    ("infer", "PATCH_OVERLAP_FACTOR", int, 4),
    ("infer", "N_CPUS_IF_NO_GPU", int, 8),
    ("train", "optimizer._target_", str, "torch.optim.RAdam"),
    ("train", "optimizer.weight_decay", float, 1e-8),
    ("train", "optimizer.betas", Tuple[float, float], (0.9, 0.999)),
    ("train", "network.NOISE2NOISE_RAND_MASK_SIZE", int, 15),
    ("train", "network.CONTRAST_LOSS_L1_EMB_REGULARIZATION", float, 1e-4),
    ("train", "network.CONTRAST_LOSS_TEMPERATURE", float, 1e-1),
    ("train", "network.TORCH_MATMUL_PRECISION", str, "medium"),
    ("train", "network.SELF_SUPERVISED_EMBEDING_SIZE", int, 256),
    ("train", "network.SwinUNETR.drop_rate", float, 0.3),
    ("train", "network.SwinUNETR.attn_drop_rate", float, 0.3),
    ("train", "network.SwinUNETR.dropout_path_rate", float, 0.2),
    ("train", "network.SwinUNETR.in_channels", int, 1),
    ("train", "network.SwinUNETR.out_channels", int, 1),
    ("train", "network.SwinUNETR.feature_size", int, 36),
    ("train", "network.SwinUNETR.use_v2", bool, True),
    ("train", "network.SwinUNETR.use_checkpoint", bool, True),
    ("train", "network.UNET.in_channels", int, 1),
    ("train", "network.UNET.first_layer_out_channels", int, 32),
    ("train", "network.UNET.num_levels", int, 5),
    ("train", "network.UNET.channels_increase_factor", int, 2),
    ("train", "network.UNET.kernel_size", int, 5),
    ("train", "network.UNET.convolution_dilation", int, 1),
    ("train", "network.UNET.last_activation", str, "linear"),
    ("train", "network.UNET.stride_conv_instead_pooling", bool, False),
    ("train", "network.UNET.output_dim", str, "same"),
    ("train", "OVERFIT_N_BATCHES", Optional[int], None),
    ("train", "N_GPUS", int, 4),
    ("train", "N_CPUS_IF_NO_GPU", int, 32),
    ("train", "USE_CUDA_FOR_DATA", bool, True),
    ("train", "FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS", float, 0.5),
    ("train", "COSINE_LR_SCHEDULE_N_EPOCHS", int, 6),
    ("train", "PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS", int, 6),
    ("train", "CHUNK_SIZE", int, 64),
    ("train", "CHUNK_STRIDE", int, 32),
    ("train", "RANDOM_FRACTION_TO_SAMPLE_TRAIN", float, -1.0),
    ("train", "SEED_FOR_TRAIN_VAL_SPLIT", int, 42),
]


def test_G10_C5_formerly_unannotated_fields_have_help_same_type_and_default():
    from tomocpt.mainConfig import MainConfig

    assert len(FORMERLY_UNANNOTATED) == 40
    m = MainConfig()
    for section, path, tp, default in FORMERLY_UNANNOTATED:
        obj, f = _owner_field(getattr(m, section), path)
        assert get_origin(f.type) is Annotated, f"{section}.{path} is not Annotated"
        assert get_args(f.type)[0] == tp, (path, get_args(f.type)[0])
        value = getattr(obj, f.name)
        assert value == default and type(value) is type(default), (path, value)
        h = _help_of(getattr(m, section), path)
        assert h and h.strip(), path


def test_G10_C5_wrapping_adds_no_cli_flag():
    """Help text only: the formerly unannotated fields do not become flags of the commands."""
    from tomocpt.configManager.cli_options import CLI_FLAGS

    for section, path, _, _ in FORMERLY_UNANNOTATED:
        if section in CLI_FLAGS:
            assert path not in CLI_FLAGS[section]


def test_G10_C6_diagnostic_list():
    from tomocpt.configManager.option_levels import DIAGNOSTIC, option_level

    paths = set(_all_paths())
    assert option_level("prepData", "write_filament_masks") == "diagnostic"
    n = 0
    for section, names in DIAGNOSTIC.items():
        for name in names:
            assert (section, name) in paths, (section, name)
            assert option_level(section, name) == "diagnostic"
            n += 1
    assert n == sum(1 for s, p in paths if option_level(s, p) == "diagnostic")


def test_G10_C7_unknown_paths_and_internal_examples():
    from tomocpt.configManager.option_levels import option_level

    with pytest.raises(KeyError):
        option_level("train", "no_such_field")
    with pytest.raises(KeyError):
        option_level("train", "particles")  # a sub-section, not a setting
    with pytest.raises(KeyError):
        option_level("nosection", "x")
    assert option_level("train", "optimizer._target_") == "internal"
    assert option_level("train", "optimizer.lr") == "advanced"
    assert option_level("train", "n_epochs") == "basic"
