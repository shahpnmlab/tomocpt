"""G6-C6: ``scripts/download_portal_data.py --masks`` on a fake portal client.

The fake ``cryoet_data_portal`` module (installed in ``sys.modules``) mimics the object model the
script already uses (``Run``, ``Tomogram``, ``Annotation`` -> ``annotation_shapes`` ->
``annotation_files`` with ``shape_type``, ``format``, ``https_path``, ``file_size``,
``tomogram_voxel_spacing.voxel_spacing``, ``alignment_id``; ``ground_truth_status``) plus the
usual extras (``AnnotationShape`` / ``AnnotationFile`` / ``TomogramVoxelSpacing`` classes with
``find``, nested-attribute filters such as ``AnnotationFile.annotation_shape.shape_type ==
"SegmentationMask"``, ``Annotation.files``). Mask annotation files are MRC
(``shape_type="SegmentationMask"``, ``format="mrc"``). Downloads are served from memory by URL
through ``download_file`` (patched on the script module) and ``requests.get``, and by
``download(...)`` methods on the fake objects.

Two runs: ``TS_A`` (microtubule mask marked ground truth) and ``TS_B`` (not ground truth); both
have ribosome point annotations, so the script's point path also runs.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_portal_data.py"
APIX = 10.0
SHAPE_ZYX = (20, 24, 32)


def _load_script():
    spec = importlib.util.spec_from_file_location("download_portal_data_g6", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mrc_bytes(vol, apix):
    import tempfile

    import mrcfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.mrc"
        with mrcfile.new(str(p), overwrite=True) as m:
            m.set_data(vol)
            m.voxel_size = apix
        return p.read_bytes()


# --------------------------------------------------------------------------------------
# fake portal
# --------------------------------------------------------------------------------------

class _Filter:
    def __init__(self, path, op, value):
        self.path, self.op, self.value = path, op, value

    def test(self, obj):
        v = obj
        for name in self.path:
            v = getattr(v, name, None)
            if v is None:
                return False
        if self.op == "eq":
            return v == self.value
        if self.op == "ne":
            return v != self.value
        if self.op == "in":
            return v in self.value
        return True


class _Field:
    def __init__(self, path):
        self._path = tuple(path)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Field(self._path + (name,))

    def __eq__(self, other):
        return _Filter(self._path, "eq", other)

    def __ne__(self, other):
        return _Filter(self._path, "ne", other)

    def _in(self, values):
        return _Filter(self._path, "in", list(values))

    def __hash__(self):
        return hash(self._path)


class _Meta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Field((name,))


class _Model(metaclass=_Meta):
    _registry: list = []

    def __init__(self, **kw):
        self.__dict__.update(kw)
        type(self)._registry.append(self)

    @classmethod
    def find(cls, client, query_filters=None, **kwargs):
        out = list(cls._registry)
        for f in query_filters or []:
            if isinstance(f, _Filter):
                out = [o for o in out if f.test(o)]
        return out

    @classmethod
    def get_by_id(cls, client, id):
        return next(o for o in cls._registry if o.id == id)


def make_fake_portal(files: dict, mask_shape=SHAPE_ZYX, mask_apix=APIX):
    """A fake ``cryoet_data_portal`` module; ``files`` (url -> bytes) is filled here."""
    cdp = types.ModuleType("cryoet_data_portal")
    cdp.__version__ = "fake-0"

    def model(name):
        return type(name, (_Model,), {"_registry": []})

    for n in ("Run", "Tomogram", "TomogramVoxelSpacing", "Annotation", "AnnotationShape", "AnnotationFile",
              "Dataset", "Alignment"):
        setattr(cdp, n, model(n))

    class Client:
        def __init__(self, *a, **k):
            pass

    cdp.Client = Client

    def _download(url):
        def _dl(self, dest_path=None, *a, **k):
            dest = Path(dest_path) if dest_path else Path.cwd()
            if dest.is_dir() or dest_path is None:
                dest = dest / url.rsplit("/", 1)[-1]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(files[url])
            return str(dest)
        return _dl

    rng = np.random.default_rng(0)
    cdp.Dataset(id=7, title="fake")
    for rid, (rname, gt) in enumerate((("TS_A", True), ("TS_B", False)), start=1):
        run = cdp.Run(id=rid, name=rname, dataset_id=7)
        tvs = cdp.TomogramVoxelSpacing(id=100 + rid, voxel_spacing=APIX, run_id=rid, run=run)
        url_t = f"https://fake/{rname}/tomo.mrc"
        files[url_t] = _mrc_bytes(rng.standard_normal(SHAPE_ZYX).astype(np.float32), APIX)
        tomo = cdp.Tomogram(
            id=200 + rid, run_id=rid, run=run, voxel_spacing=APIX, processing="raw", reconstruction_method="WBP",
            is_portal_standard=True, is_visualization_default=True, https_mrc_file=url_t,
            file_size_mrc=len(files[url_t]), size_x=SHAPE_ZYX[2], size_y=SHAPE_ZYX[1], size_z=SHAPE_ZYX[0],
            alignment_id=1, tomogram_voxel_spacing=tvs, tomogram_voxel_spacing_id=tvs.id)
        tomo.download_mrcfile = types.MethodType(_download(url_t), tomo)
        # ribosome points (ground truth in both runs)
        ann_p = cdp.Annotation(id=300 + rid, run_id=rid, run=run, object_name="ribosome", ground_truth_status=True,
                               is_curator_recommended=True, method_type="manual", annotation_shapes=[], files=[])
        url_p = f"https://fake/{rname}/ribosome.ndjson"
        files[url_p] = "\n".join(json.dumps({"type": "orientedPoint", "location": {"x": x, "y": y, "z": z}})
                                 for x, y, z in ((5, 6, 7), (20, 10, 12))).encode()
        sh_p = cdp.AnnotationShape(id=400 + rid, shape_type="OrientedPoint", annotation=ann_p,
                                   annotation_id=ann_p.id, annotation_files=[])
        f_p = cdp.AnnotationFile(id=500 + rid, format="ndjson", https_path=url_p, s3_path=url_p,
                                 file_size=len(files[url_p]), tomogram_voxel_spacing=tvs,
                                 tomogram_voxel_spacing_id=tvs.id, alignment_id=1, annotation_shape=sh_p,
                                 annotation_shape_id=sh_p.id, shape_type="OrientedPoint",
                                 is_visualization_default=True)
        f_p.download = types.MethodType(_download(url_p), f_p)
        sh_p.annotation_files.append(f_p)
        ann_p.annotation_shapes.append(sh_p)
        ann_p.files.append(f_p)
        # microtubule segmentation mask
        ann_m = cdp.Annotation(id=600 + rid, run_id=rid, run=run, object_name="microtubule", ground_truth_status=gt,
                               is_curator_recommended=True, method_type="automated", annotation_shapes=[], files=[])
        url_m = f"https://fake/{rname}/microtubule.mrc"
        mask = np.zeros(mask_shape, np.int8)
        mask[mask_shape[0] // 2 - 2:mask_shape[0] // 2 + 2, mask_shape[1] // 2 - 2:mask_shape[1] // 2 + 2, :] = 1
        files[url_m] = _mrc_bytes(mask, mask_apix)
        sh_m = cdp.AnnotationShape(id=700 + rid, shape_type="SegmentationMask", annotation=ann_m,
                                   annotation_id=ann_m.id, annotation_files=[])
        f_m = cdp.AnnotationFile(id=800 + rid, format="mrc", https_path=url_m, s3_path=url_m,
                                 file_size=len(files[url_m]), tomogram_voxel_spacing=tvs,
                                 tomogram_voxel_spacing_id=tvs.id, alignment_id=1, annotation_shape=sh_m,
                                 annotation_shape_id=sh_m.id, shape_type="SegmentationMask",
                                 is_visualization_default=True)
        f_m.download = types.MethodType(_download(url_m), f_m)
        sh_m.annotation_files.append(f_m)
        ann_m.annotation_shapes.append(sh_m)
        ann_m.files.append(f_m)
    return cdp


class _Resp:
    def __init__(self, data, start=0):
        self._data = data[start:]
        self.status_code = 206 if start else 200
        self.headers = {"content-length": str(len(self._data))}
        self.content = self._data

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1 << 16, *a, **k):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install(monkeypatch, dpd, cdp, files):
    monkeypatch.setitem(sys.modules, "cryoet_data_portal", cdp)

    def fake_download_file(url, dest, expected_size=None, *a, **k):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(files[url])
        return dest

    monkeypatch.setattr(dpd, "download_file", fake_download_file)
    try:
        import requests

        def fake_get(url, *a, **k):
            h = (k.get("headers") or {}).get("Range")
            start = int(h.split("=")[1].rstrip("-")) if h else 0
            return _Resp(files[url], start)

        monkeypatch.setattr(requests, "get", fake_get)
    except ImportError:  # pragma: no cover
        pass


def _ground_truth_entries(obj, path=()):
    """Every dict in the manifest with a ``ground_truth`` key, with its key path."""
    if isinstance(obj, dict):
        if "ground_truth" in obj:
            yield path, obj
        for k, v in obj.items():
            yield from _ground_truth_entries(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _ground_truth_entries(v, path + (str(i),))


def _mentions(path, entry, word):
    return any(word in p for p in path) or any(isinstance(v, str) and word in v for v in entry.values())


def _run(tmp_path, monkeypatch, **fake_kw):
    dpd = _load_script()
    files = {}
    cdp = make_fake_portal(files, **fake_kw)
    _install(monkeypatch, dpd, cdp, files)
    out = tmp_path / "portal"
    argv = ["--dataset-id", "7", "--out", str(out), "--voxel-spacing", str(APIX), "--masks", "microtubule"]
    return dpd, out, argv


def test_G6_C6_masks_written_with_tomogram_shape_and_voxel_size(tmp_path, monkeypatch):
    import mrcfile

    dpd, out, argv = _run(tmp_path, monkeypatch)
    rc = dpd.main(argv)
    assert rc in (0, None), f"exit code {rc}"
    tomo_dir = out / "tomograms"
    masks = [p for p in out.rglob("*.mrc") if tomo_dir not in p.parents]
    for run in ("TS_A", "TS_B"):
        mine = [p for p in masks if run in str(p.relative_to(out)) and "microtubule" in str(p.relative_to(out))]
        assert len(mine) == 1, f"{run}: expected one microtubule mask MRC, got {mine} (all: {masks})"
        with mrcfile.open(str(mine[0]), permissive=True) as m:
            assert m.data.shape == SHAPE_ZYX
            assert float(m.voxel_size.x) == pytest.approx(APIX)
            assert int(np.count_nonzero(m.data)) == 4 * 4 * SHAPE_ZYX[2]

    manifest = json.loads((out / "manifest.json").read_text())
    runs = {r["run"]: r for r in manifest["runs"]}
    for run, gt in (("TS_A", True), ("TS_B", False)):
        entries = [e for p, e in _ground_truth_entries(runs[run]) if _mentions(p, e, "microtubule")]
        assert entries, f"{run}: no ground_truth entry for the microtubule mask in the manifest"
        assert all(bool(e["ground_truth"]) is gt for e in entries), (run, entries)
    snippet = (out / "prepdata_snippet.yaml").read_text()
    assert "filament_masks" in snippet, "the prepData snippet must name the masks (prepData.filament_masks)"


def test_G6_C6_mask_with_a_different_shape_is_refused(tmp_path, monkeypatch):
    """A mask whose shape differs from the tomogram's is not accepted silently."""
    dpd, out, argv = _run(tmp_path, monkeypatch, mask_shape=(20, 24, 30))
    dpd.build_parser().parse_args(argv)  # --masks is an option of the script
    try:
        rc = dpd.main(argv)
    except (SystemExit, Exception) as e:  # an error stops the script: accepted
        rc = getattr(e, "code", 1) or 1
    tomo_dir = out / "tomograms"
    masks = [p for p in out.rglob("*.mrc") if tomo_dir not in p.parents] if out.exists() else []
    import mrcfile

    bad = []
    for p in masks:
        with mrcfile.open(str(p), permissive=True, header_only=True) as m:
            shape = (int(m.header.nz), int(m.header.ny), int(m.header.nx))
        if shape != SHAPE_ZYX:
            bad.append(p)
    assert rc not in (0, None) or not bad, f"mask with the wrong shape accepted: {bad}"
