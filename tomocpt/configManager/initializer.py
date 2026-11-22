"""``tomocpt init``: a commented YAML config with every setting up to a level (Gate 10).

The YAML is written by hand (no new dependency): each field is preceded by a comment with its help
text and type, fields are grouped by level (``option_levels``) and sub-sections stay nested. Values
are the current defaults, or the ``key=value`` overrides given to ``init`` (an overridden field is
written even when its level is above the one asked for). The file loads with ``--config_file``;
fields left out keep their defaults.
"""
import json
import math
import textwrap
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, List, Optional

import typer
from omegaconf import MISSING, DictConfig

from tomocpt.configManager.option_levels import (
    INTERNAL,
    LEVELS,
    SECTIONS,
    describe_type,
    level_rank,
    option_help,
    option_level,
    subsection_help,
)

WIDTH = 100


class InitLevel(str, Enum):
    basic = "basic"
    advanced = "advanced"
    diagnostic = "diagnostic"


HEADER = """\
TomoCPT configuration, written by `tomocpt init --level {level}`.

Levels: basic (everyday settings; the command-line flags of prepare_data, train and predict),
advanced (every other setting that changes training or prediction) and diagnostic (settings that
only add output for inspection). A level includes the ones below it: `tomocpt init --level
advanced` (or `--level diagnostic`) writes more settings. A setting left out of this file keeps
its default, so delete what you do not change.

Use: tomocpt prepare_data | train | predict --config_file config.yaml
Override a value on the command line with key=value after the flags, relative to the command's
section, with dots for sub-sections, e.g.
  tomocpt train --config_file config.yaml n_epochs=50 particles.class_head=softmax
(`tomocpt init` takes the full path: train.particles.class_head=softmax.)
`shared` holds the paths used by several commands. ??? marks a required value.

All options: docs/options.md. What to change, when and why: docs/tuning.md.
Worked examples: docs/examples/."""


def _yaml_scalar(value: Any) -> str:
    """``value`` as a YAML flow scalar that OmegaConf reads back with the same meaning."""
    if value is MISSING or (isinstance(value, str) and value == "???"):
        return "???"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return json.dumps(str(value.value), ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ".nan"
        if math.isinf(value):
            return ".inf" if value > 0 else "-.inf"
        text = repr(value)
        if "e" in text:
            mantissa, exp = text.split("e")
            if "." not in mantissa:
                mantissa += ".0"
            if exp[0] not in "+-":
                exp = "+" + exp
            text = f"{mantissa}e{exp}"
        return text
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def _comment(text: str, indent: int) -> List[str]:
    pad = " " * indent
    width = max(40, WIDTH - indent - 2)
    out = []
    for para in str(text).splitlines() or [""]:
        lines = textwrap.wrap(para, width=width, break_long_words=True, break_on_hyphens=False) or [""]
        out.extend(f"{pad}# {line}".rstrip() for line in lines)
    return out


def _field_lines(section: str, path: str, name: str, field_type: Any, value: Any, indent: int) -> List[str]:
    pad = " " * indent
    lines = [""]
    help_text = option_help(section, path)
    if help_text:
        lines += _comment(help_text, indent)
    lines.append(f"{pad}# type: {describe_type(field_type)}")
    lines.append(f"{pad}{name}: {_yaml_scalar(value)}")
    return lines


def _render_node(section: str, obj: Any, default_obj: Any, prefix: str, indent: int, max_rank: int,
                 headers: str) -> List[str]:
    """Lines of the fields of ``obj`` to write; ``headers``: 'always' or 'mixed' (only when the node
    has more than one level)."""
    groups = {}  # rank -> list of line blocks
    for f in fields(type(obj)):
        value = getattr(obj, f.name)
        default = getattr(default_obj, f.name, MISSING)
        path = prefix + f.name
        if is_dataclass(value):
            child = _render_node(section, value, default, path + ".", indent + 2, max_rank, "mixed")
            if not child:
                continue
            rank = child_rank = min(r for r, _ in child)
            block = [""]
            sub_help = subsection_help(obj, f.name)
            if sub_help:
                block += _comment(sub_help, indent)
            block.append(f"{' ' * indent}{f.name}:")
            for _, lines in child:
                block += lines
            groups.setdefault(child_rank, []).append(block)
            continue
        level = option_level(section, path)
        rank = level_rank(level)
        changed = not _same(value, default)
        if rank > max_rank and not changed:
            continue
        groups.setdefault(rank, []).append(_field_lines(section, path, f.name, f.type, value, indent))
    if not groups:
        return []
    show = headers == "always" or len(groups) > 1
    out = []
    for rank in sorted(groups):
        block = []
        if show:
            name = LEVELS[rank] if rank < len(LEVELS) else INTERNAL
            block += ["", f"{' ' * indent}# --- {name} ---"]
        for lines in groups[rank]:
            block += lines
        out.append((rank, block))
    return out


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return a is b


def render_config(config: Any = None, level: str = "basic") -> str:
    """The commented YAML text of ``config`` (default: ``MainConfig()``) up to ``level``."""
    from tomocpt.mainConfig import MainConfig

    level = str(getattr(level, "value", level))
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    if config is None:
        config = MainConfig()
    defaults = MainConfig()
    max_rank = level_rank(level)
    lines = _comment(HEADER.format(level=level), 0)
    for section in SECTIONS:
        node = _render_node(section, getattr(config, section), getattr(defaults, section), "", 2, max_rank,
                            "always")
        if not node:
            continue
        lines += ["", f"{section}:"]
        for _, block in node:
            lines += block
    return "\n".join(lines) + "\n"


def init(
    output_path: Path = Path.cwd() / "config.yaml",
    level: Annotated[InitLevel, typer.Option(
        help="Write every setting up to this level: basic (the everyday flags), advanced (every "
             "setting) or diagnostic (also settings that only add output for inspection)")] = InitLevel.basic,
    config: DictConfig = None,
) -> None:
    """
    Write a commented config file for running tomoCPT: every setting up to ``level``, with its help
    text, type and default. ``key=value`` overrides (``train.n_epochs=50``) are written too.

    Args:
        output_path (Path): Path where the config file should be saved.
            Defaults to 'config.yaml' in the current working directory.
        level: basic, advanced or diagnostic.
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)
    text = render_config(config, level)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    print(f"Config file created at: {output_path} (level {getattr(level, 'value', level)})")


if __name__ == "__main__":
    init(Path("/tmp/config.yaml"))
