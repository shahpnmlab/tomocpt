#!/usr/bin/env python
"""Training dashboard window: input | target | prediction | overlay of training and validation
chunks, scrubbed by slice and epoch, plus the loss curves.

    python scripts/train_dashboard.py <run dir>      # over ssh -X / -Y, the window opens locally

<run dir> is any folder above the TensorBoard events file (e.g. benchmarks/out/<run>). Volumes come
from ``<log_dir>/snapshots`` (``train.snapshot_chunks > 0``); a run without them shows the 2D panels
TensorBoard logged (epoch only, no slice). The window re-reads the run every 30 s while it trains.

The panels are rendered off-screen on the GPU (vispy, EGL) and shown as plain images, so no OpenGL
goes over the X connection. Needs ``pip install 'tomocpt[dashboard]'`` (vispy, PySide6) and an EGL
driver (the NVIDIA driver ships one).
"""
import argparse
import io
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("QT_API", "pyside6")  # matplotlib's Qt backend
import vispy  # noqa: E402

vispy.use(app="egl")  # off-screen; Qt below only shows the result
from vispy import scene  # noqa: E402
from vispy.visuals.transforms import STTransform  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

GAP = 2  # pixels between panels (before magnification)
CLASS_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


# ------------------------------------------------------------------------------------- data
class Run:
    """The events file and the snapshots of one training run."""

    def __init__(self, root: Path):
        events = sorted(root.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
        if not events:
            raise SystemExit(f"no TensorBoard events file under {root}")
        self.root = root
        self._class_names = None
        self.events = events[-1]
        self.log_dir = self.events.parent
        self.snap_dir = self.log_dir / "snapshots"
        self.acc = None
        self._chunks = None
        self._preds = {}

    def load_events(self):
        """Re-read the events file (slow for long runs: call it off the GUI thread)."""
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        images = 1 if self.has_volumes() else 0  # 0 = keep all
        a = EventAccumulator(str(self.events), size_guidance={"scalars": 0, "images": images})
        a.Reload()
        return a

    def losses(self) -> dict:
        """``{name: [(epoch, value), ...]}`` of ``loss`` (train) and the ``val_loss*`` curves."""
        a = self.acc
        if a is None:
            return {}
        tags = a.Tags()["scalars"]
        steps = sorted((e.step, int(e.value)) for e in a.Scalars("epoch")) if "epoch" in tags else []
        keys = [s for s, _ in steps]
        out = {}
        for t in tags:
            if t == "loss" or (t.startswith("val_loss") and t != "val_loss_smooth"):
                pts = {}
                for e in a.Scalars(t):
                    i = int(np.searchsorted(keys, e.step, side="right")) - 1
                    if i >= 0:
                        pts[steps[i][1]] = float(e.value)
                out["train" if t == "loss" else t.replace("val_loss_", "val ").replace("val_loss", "val")] = \
                    sorted(pts.items())
        return out

    def class_names(self) -> list:
        """Class-table order (fixes the colours): the snapshots, else the run's chunk_layout.json."""
        if self.has_volumes():
            return [str(x) for x in self.chunks()["classes"]]
        if self._class_names is None:
            import json

            self._class_names = []
            for p in sorted(self.root.rglob("chunk_layout.json")):
                try:
                    self._class_names = [r["name"] for r in json.loads(p.read_text()).get("class_table", [])]
                    break
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return self._class_names

    def has_volumes(self) -> bool:
        return (self.snap_dir / "chunks.npz").is_file()

    def chunks(self) -> dict:
        if self._chunks is None:
            with np.load(self.snap_dir / "chunks.npz") as z:
                self._chunks = {k: z[k] for k in z.files}
        return self._chunks

    def epochs(self) -> list:
        if self.has_volumes():
            return sorted(int(m.group(1)) for p in self.snap_dir.glob("epoch_*.npz")
                          if (m := re.fullmatch(r"epoch_(\d+)\.npz", p.name)))
        if self.acc is None:
            return []
        return sorted({e.step for t in self.acc.Tags()["images"] for e in self.acc.Images(t)})

    def items(self) -> list:
        """Chunk names (volumes) or TensorBoard image tags (2D panels)."""
        if self.has_volumes():
            return [str(x) for x in self.chunks()["name"]]
        return sorted(self.acc.Tags()["images"]) if self.acc is not None else []

    def pred(self, epoch: int) -> np.ndarray:
        if epoch not in self._preds:
            with np.load(self.snap_dir / f"epoch_{epoch:04d}.npz") as z:
                self._preds[epoch] = z["pred"]
            if len(self._preds) > 64:
                self._preds.pop(next(iter(self._preds)))
        return self._preds[epoch]

    def tb_panel(self, tag: str, epoch: int) -> np.ndarray:
        ims = {e.step: e for e in self.acc.Images(tag)}
        older = [s for s in ims if s <= epoch]
        e = ims[max(older)] if older else ims[min(ims)]
        return np.asarray(Image.open(io.BytesIO(e.encoded_image_string)).convert("RGB"))  # gray or turbo panels


# -------------------------------------------------------------------------------- rendering
class Renderer:
    """One off-screen vispy canvas (EGL); the panels are drawn magnified without smoothing."""

    def __init__(self, scale: int):
        self.scale = scale
        self.canvas = scene.SceneCanvas(size=(64, 64), show=False, bgcolor="white")
        # canvas.scene is in pixels with the origin at the top left
        self.image = scene.visuals.Image(np.zeros((2, 2, 3), np.uint8), parent=self.canvas.scene,
                                         interpolation="nearest")
        self.image.transform = STTransform(scale=(scale, scale))

    def render(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        self.canvas.size = (w * self.scale, h * self.scale)
        self.image.set_data(rgb)
        return np.ascontiguousarray(self.canvas.render(alpha=False)[..., :3])


def gray(a: np.ndarray) -> np.ndarray:
    return np.repeat(a[..., None], 3, axis=-1)


TURBO = None


def turbo(a: np.ndarray) -> np.ndarray:
    """uint8 values (0-255 = probability 0-1) in matplotlib's turbo, fixed scale."""
    global TURBO
    if TURBO is None:
        from matplotlib import colormaps
        TURBO = (colormaps["turbo"](np.arange(256))[:, :3] * 255).round().astype(np.uint8)
    return TURBO[a]


def overlay(inp: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Input in gray, prediction in orange (opacity = probability), target 0.5 outline in blue."""
    rgb = gray(inp).astype(np.float32) * 0.8
    p = pred.astype(np.float32)[..., None] / 255.0
    rgb = rgb * (1 - p) + p * np.array([230, 159, 0], np.float32)
    t = target > 127
    edge = t & ~(np.roll(t, 1, 0) & np.roll(t, -1, 0) & np.roll(t, 1, 1) & np.roll(t, -1, 1))
    rgb[edge] = (86, 180, 233)
    return rgb.clip(0, 255).astype(np.uint8)


def hstack(tiles) -> np.ndarray:
    gap = np.full((tiles[0].shape[0], GAP, 3), 255, np.uint8)
    out = [tiles[0]]
    for t in tiles[1:]:
        out += [gap, t]
    return np.concatenate(out, axis=1)


def take(vol: np.ndarray, axis: int, i: int) -> np.ndarray:
    return np.take(vol, int(np.clip(i, 0, vol.shape[axis] - 1)), axis=axis)


# ------------------------------------------------------------------------------------ window
class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, run: Run, scale: int):
        super().__init__()
        self.run, self.renderer = run, Renderer(scale)
        self.epochs, self.n_items = [], 0
        self._loading = None
        self.setWindowTitle(f"Training dashboard - {run.log_dir}")

        self.item = QtWidgets.QComboBox()
        self.item.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cls = QtWidgets.QComboBox()
        self.axis = QtWidgets.QComboBox()
        self.axis.addItems(["Z", "Y", "X"])
        self.slice = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_lab = QtWidgets.QLabel()
        self.epoch = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.epoch_lab = QtWidgets.QLabel()
        self.follow = QtWidgets.QCheckBox("follow latest")
        self.follow.setChecked(True)
        self.heads = QtWidgets.QLabel()
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.logy = QtWidgets.QCheckBox("log loss axis")
        self.fig = Figure(figsize=(8, 3), layout="constrained")
        self.plot = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)

        top = QtWidgets.QHBoxLayout()
        for lab, w in (("Chunk", self.item), ("Class", self.cls), ("Axis", self.axis)):
            top.addWidget(QtWidgets.QLabel(lab))
            top.addWidget(w)
        top.addStretch(1)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Slice"), 0, 0)
        grid.addWidget(self.slice, 0, 1)
        grid.addWidget(self.slice_lab, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Epoch"), 1, 0)
        grid.addWidget(self.epoch, 1, 1)
        grid.addWidget(self.epoch_lab, 1, 2)
        grid.addWidget(self.follow, 1, 3)
        lay = QtWidgets.QVBoxLayout()
        lay.addLayout(top)
        lay.addLayout(grid)
        lay.addWidget(self.heads)
        lay.addWidget(self.view)
        lay.addWidget(self.note)
        lay.addWidget(self.logy)
        lay.addWidget(self.plot, 1)
        central = QtWidgets.QWidget()
        central.setLayout(lay)
        self.setCentralWidget(central)
        self.statusBar().showMessage("reading the events file...")

        self.item.currentIndexChanged.connect(self._item_changed)
        for sig in (self.cls.currentIndexChanged, self.axis.currentIndexChanged, self.slice.valueChanged):
            sig.connect(self.draw)
        self.epoch.valueChanged.connect(self._epoch_changed)
        self.logy.toggled.connect(self.draw_losses)

        self._setup_mode()
        self.refresh()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(30_000)
        self.poll = QtCore.QTimer(self)
        self.poll.timeout.connect(self._apply_loaded)
        self.poll.start(200)

    # -- setup
    def _setup_mode(self):
        volumes = self.run.has_volumes()
        for w in (self.cls, self.axis, self.slice, self.slice_lab):
            w.setEnabled(volumes)
        names = ["input", "target", "prediction"] + (["overlay (pred orange, target blue)"] if volumes else [])
        self.heads.setText("   |   ".join(names))
        if volumes:
            c = self.run.chunks()
            self.cls.addItems([str(x) for x in c["classes"]])
            self.slice.setRange(0, c["input"].shape[1] - 1)
            self.slice.setValue(c["input"].shape[1] // 2)
            self.note.setText("")
        else:
            self.note.setText("This run has no volume snapshots (train.snapshot_chunks = 0): these are the 2D "
                              "panels TensorBoard logged, one per epoch; training panels show a different chunk "
                              "each epoch.")

    # -- background reload of the events file
    def refresh(self):
        if self._loading is not None:
            return
        box = {}

        def work():
            try:
                box["acc"] = self.run.load_events()
            except Exception as e:  # noqa: BLE001
                box["err"] = e

        self._loading = (threading.Thread(target=work, daemon=True), box)
        self._loading[0].start()

    def _apply_loaded(self):
        if self._loading is None or self._loading[0].is_alive():
            return
        _, box = self._loading
        self._loading = None
        if "err" in box:
            self.statusBar().showMessage(f"events file: {box['err']}")
            return
        self.run.acc = box["acc"]
        items = self.run.items()
        if len(items) != self.n_items:
            self.item.blockSignals(True)
            self.item.clear()
            self.item.addItems(items)
            self.item.blockSignals(False)
            self.n_items = len(items)
            self._item_changed()
        self.epochs = self.run.epochs()
        at_end = self.follow.isChecked() or self.epoch.value() >= self.epoch.maximum()
        self.epoch.blockSignals(True)
        self.epoch.setRange(0, max(0, len(self.epochs) - 1))
        if at_end:
            self.epoch.setValue(self.epoch.maximum())
        self.epoch.blockSignals(False)
        self.statusBar().showMessage(f"{self.run.log_dir}   {len(self.epochs)} epochs   "
                                     f"{'volumes' if self.run.has_volumes() else 'TensorBoard 2D panels'}")
        self.draw()
        self.draw_losses()

    # -- events
    def _item_changed(self, *_):
        if self.run.has_volumes() and self.item.currentIndex() >= 0:
            self.cls.blockSignals(True)
            self.cls.setCurrentIndex(int(self.run.chunks()["cls"][self.item.currentIndex()]))
            self.cls.blockSignals(False)
        self.draw()

    def _epoch_changed(self, *_):
        self.follow.setChecked(self.epoch.value() >= self.epoch.maximum())
        self.draw()
        self.draw_losses()

    def current_epoch(self):
        return self.epochs[self.epoch.value()] if self.epochs else None

    # -- drawing
    def draw(self, *_):
        ep = self.current_epoch()
        self.slice_lab.setText(str(self.slice.value()))
        self.epoch_lab.setText("-" if ep is None else str(ep))
        if ep is None or self.item.currentIndex() < 0:
            return
        k = self.item.currentIndex()
        if self.run.has_volumes():
            c = self.run.chunks()
            cls, axis, z = self.cls.currentIndex(), self.axis.currentIndex(), self.slice.value()
            inp, tgt = take(c["input"][k], axis, z), take(c["target"][k, cls], axis, z)
            prd = take(self.run.pred(ep)[k, cls], axis, z)
            rgb = hstack([gray(inp), turbo(tgt), turbo(prd), overlay(inp, prd, tgt)])
        else:
            rgb = self.run.tb_panel(self.item.currentText(), ep)
        out = self.renderer.render(rgb)
        h, w = out.shape[:2]
        self._img = out  # QImage does not own the buffer
        qimg = QtGui.QImage(out.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.view.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def draw_losses(self):
        ax = self.ax
        ax.clear()
        losses = self.run.losses()
        classes = self.run.class_names()
        for name in sorted(losses):
            pts = losses[name]
            if not pts:
                continue
            x, y = zip(*pts)
            cname = name[4:] if name.startswith("val ") else None
            if name == "train":
                kw = dict(color="#999999", ls="--")
            elif name == "val":
                kw = dict(color="#444444")
            else:
                i = classes.index(cname) if cname in classes else sorted(losses).index(name)
                kw = dict(color=CLASS_COLORS[i % len(CLASS_COLORS)])
            ax.plot(x, y, lw=2, label=name, **kw)
        ep = self.current_epoch()
        if ep is not None:
            ax.axvline(ep, color="#888888", ls=":", lw=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        if self.logy.isChecked():
            from matplotlib.ticker import FormatStrFormatter

            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
            ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2g"))
        else:
            ax.set_ylim(bottom=0)
        ax.grid(True, color="#eeeeee")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if losses:
            ax.legend(frameon=False, ncol=min(4, len(losses)), loc="upper right")
        self.plot.draw_idle()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--scale", type=int, default=4, help="screen pixels per voxel")
    a = ap.parse_args()
    run = Run(a.run_dir.resolve())
    app = QtWidgets.QApplication(sys.argv[:1])
    w = Dashboard(run, a.scale)
    w.resize(1120, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
