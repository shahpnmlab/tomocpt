"""Class table of data pipeline v2 (CLAUDE.md Section 3, "class table").

Channel ``c`` of the target and of the model output is row ``c`` of the class table:
particles sorted by class number, then filaments sorted by class number.
:func:`class_table_from_config` is the only way to obtain the table; Gate 3 builds the
model and Gate 4 plans inference from the ``prepData`` saved in the checkpoint.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

PARTICLE = "particle"
FILAMENT = "filament"


@dataclass
class ClassSpec:
    number: int
    name: str
    diameter_ang: float
    kind: str = PARTICLE


def _safe_get(obj, name, default=None):
    try:
        from tomocpt.configManager.configManager import safe_getattr
        return safe_getattr(obj, name, default)
    except ImportError:  # pragma: no cover
        return getattr(obj, name, default)


def _split_entries(text: str, what: str) -> List[tuple]:
    if text is None:
        return []
    out = []
    for raw in str(text).split(","):
        entry = raw.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f"Malformed {what} entry '{entry}': expected '<class number>:<value>'")
        key_txt, val_txt = parts[0].strip(), parts[1].strip()
        try:
            key = int(key_txt)
        except ValueError:
            raise ValueError(f"Malformed {what} entry '{entry}': class number '{key_txt}' is not an integer")
        out.append((key, val_txt, entry))
    return out


def parse_class_diameters(text: str) -> Dict[int, float]:
    """``"1:120, 2:130"`` -> ``{1: 120.0, 2: 130.0}`` (A). Spaces are tolerated; a malformed
    entry, a non-positive diameter or a duplicate class raises ValueError naming it."""
    out: Dict[int, float] = {}
    for key, val, entry in _split_entries(text, "class_diameters"):
        try:
            d = float(val)
        except ValueError:
            raise ValueError(f"Malformed class_diameters entry '{entry}': diameter '{val}' is not a number")
        if not d > 0:
            raise ValueError(f"class_diameters entry '{entry}': the diameter of class {key} must be > 0")
        if key in out:
            raise ValueError(f"Duplicate class {key} in class_diameters")
        out[key] = d
    return out


def parse_class_names(text: str) -> Dict[int, str]:
    """``"1:apo-ferritin, 2:ribosome"`` -> ``{1: "apo-ferritin", 2: "ribosome"}``."""
    out: Dict[int, str] = {}
    for key, val, entry in _split_entries(text, "class_names"):
        if key in out:
            raise ValueError(f"Duplicate class {key} in class_names")
        out[key] = val
    names = list(out.values())
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"Duplicate class name(s) in class_names: {dupes}")
    return out


def build_class_table(diameters: Dict[int, float], names: Optional[Dict[int, str]] = None,
                      kinds: Optional[Dict[int, str]] = None) -> List[ClassSpec]:
    """Particles sorted by class number, then filaments sorted by class number.

    A class named in ``names`` or ``kinds`` without a diameter raises ValueError naming it.
    Classes without a name are called ``class_<number>``.
    """
    names = dict(names or {})
    kinds = dict(kinds or {})
    diameters = {int(k): float(v) for k, v in diameters.items()}
    if not diameters:
        raise ValueError("The class table needs at least one class with a diameter")
    for k in list(names) + list(kinds):
        if int(k) not in diameters:
            raise ValueError(f"Class {k} has no diameter (add it to class_diameters)")
    rows = []
    for number, d in diameters.items():
        kind = str(getattr(kinds.get(number, PARTICLE), "value", kinds.get(number, PARTICLE)))
        if kind not in (PARTICLE, FILAMENT):
            raise ValueError(f"Class {number}: kind must be '{PARTICLE}' or '{FILAMENT}', got '{kind}'")
        rows.append(ClassSpec(number=int(number), name=str(names.get(number, f"class_{number}")),
                              diameter_ang=float(d), kind=kind))
    seen = [r.name for r in rows]
    dupes = sorted({n for n in seen if seen.count(n) > 1})
    if dupes:
        raise ValueError(f"Duplicate class name(s) in the class table: {dupes}")
    rows.sort(key=lambda r: (0 if r.kind == PARTICLE else 1, r.number))
    return rows


def single_class_number(class_id: str) -> int:
    """Class number of single-class data: the integer ``class_id``, otherwise 1 (``"all"``,
    or a subset such as ``"1:2"``, maps to 1)."""
    text = str(class_id).strip()
    return int(text) if re.fullmatch(r"\d+", text) else 1


def class_table_from_config(prep_cfg) -> List[ClassSpec]:
    """The class table of a ``PrepdataConfig`` (the only way to obtain it).

    Multi-class (``multiclass=True``): from ``class_diameters`` (required) and
    ``class_names``. Single-class: one row (number from ``class_id``, ``"all"`` -> 1; name
    ``class_id``; diameter ``particle_length_ang``). Legacy configs with several single-class
    datasets give one row per dataset (numeric ``class_id`` values needed).
    """
    if bool(_safe_get(prep_cfg, "multiclass", False)):
        text = _safe_get(prep_cfg, "class_diameters", None)
        if not text:
            raise ValueError("prepData.multiclass=True needs prepData.class_diameters (e.g. '1:120,2:130')")
        diameters = parse_class_diameters(text)
        names_text = _safe_get(prep_cfg, "class_names", None)
        names = parse_class_names(names_text) if names_text else {}
        return build_class_table(diameters, names)

    lengths_txt = str(_safe_get(prep_cfg, "particle_length_ang", ""))
    ids_txt = str(_safe_get(prep_cfg, "class_id", "all"))
    try:
        lengths = [float(x.strip()) for x in lengths_txt.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"prepData.particle_length_ang '{lengths_txt}' is not a list of numbers")
    ids = [x.strip() for x in ids_txt.split(",") if x.strip()] or ["all"]
    if not lengths:
        raise ValueError("prepData.particle_length_ang is needed for the class table")
    if len(lengths) == 1:
        return [ClassSpec(number=single_class_number(ids[0]), name=ids[0], diameter_ang=lengths[0])]
    if len(ids) != len(lengths):
        raise ValueError(
            f"prepData.particle_length_ang has {len(lengths)} entries but class_id has {len(ids)}")
    numbers = [single_class_number(i) for i in ids]
    if len(set(numbers)) != len(numbers) or not all(re.fullmatch(r"\d+", i) for i in ids):
        raise ValueError(
            "Several single-class datasets need distinct numeric class_id values to form a class table "
            f"(got class_id '{ids_txt}'); use prepData.multiclass with class_diameters instead.")
    rows = {}
    for n, i, L in zip(numbers, ids, lengths):
        rows[n] = (i, L)
    return build_class_table({n: L for n, (i, L) in rows.items()}, {n: i for n, (i, L) in rows.items()})


def parse_class_annotated(text, class_table: Sequence[ClassSpec]) -> Optional[List[int]]:
    """Class numbers listed in ``class_annotated`` (numbers or names, separated by commas,
    colons, semicolons or spaces). None/empty -> None (meaning: every class present)."""
    if text is None:
        return None
    if isinstance(text, float):
        if text != text:  # NaN from a CSV
            return None
        text = str(int(text)) if float(text).is_integer() else str(text)
    tokens = [t for t in re.split(r"[,:;\s]+", str(text).strip()) if t]
    if not tokens:
        return None
    by_name = {r.name: r.number for r in class_table}
    numbers = {r.number for r in class_table}
    out = []
    for t in tokens:
        if re.fullmatch(r"\d+", t):
            n = int(t)
        elif re.fullmatch(r"\d+\.0+", t):
            n = int(float(t))
        elif t in by_name:
            n = by_name[t]
        else:
            raise ValueError(f"class_annotated entry '{t}' is neither a class number nor a class name")
        if numbers and n not in numbers:
            raise ValueError(f"class_annotated entry '{t}': class {n} is not in the class table")
        if n not in out:
            out.append(n)
    return out


def class_table_to_json(class_table: Iterable[ClassSpec]) -> List[dict]:
    """Rows as ``dataclasses.asdict`` dicts (``number``, ``name``, ``diameter_ang``, ``kind``)."""
    out = []
    for r in class_table:
        d = asdict(r) if isinstance(r, ClassSpec) else dict(r)
        out.append({"number": int(d["number"]), "name": str(d["name"]),
                    "diameter_ang": float(d["diameter_ang"]),
                    "kind": str(getattr(d.get("kind", PARTICLE), "value", d.get("kind", PARTICLE)))})
    return out


def class_table_from_json(rows: Iterable) -> List[ClassSpec]:
    out = []
    for e in rows:
        if isinstance(e, ClassSpec):
            out.append(e)
        elif isinstance(e, dict):
            out.append(ClassSpec(int(e["number"]), str(e["name"]), float(e["diameter_ang"]),
                                 str(e.get("kind", PARTICLE))))
        else:
            e = list(e)
            out.append(ClassSpec(int(e[0]), str(e[1]), float(e[2]), str(e[3]) if len(e) > 3 else PARTICLE))
    return out


def same_class_table(a: Sequence[ClassSpec], b: Sequence[ClassSpec], rtol: float = 1e-9) -> bool:
    a, b = class_table_to_json(a), class_table_to_json(b)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x["number"], x["name"], x["kind"]) != (y["number"], y["name"], y["kind"]):
            return False
        if abs(x["diameter_ang"] - y["diameter_ang"]) > rtol * max(abs(x["diameter_ang"]), abs(y["diameter_ang"])):
            return False
    return True
