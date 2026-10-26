"""Legacy (``v2-gate-0-base``) checkpoints for compatibility tests (Gate 1, frozen afterwards).

* :func:`legacy_checkpoint` - tiny ``BasePickingModel`` checkpoint written by base code.
* :func:`legacy_ssl_checkpoint` - the same for ``SelfSupervisedModel``.
* :func:`legacy_reference` - ``np.load`` of the outputs the legacy model gave on a fixed input
  (keys ``x``, ``logits``, ``hidden4``; eval mode, CPU fp32).
* :func:`zenodo_checkpoint` - path from ``$TOMOCPT_ZENODO_CKPT`` or ``None``.

Checkpoints are built once by running ``tests/fixtures/make_legacy_ckpt.py`` in a git worktree
of ``v2-gate-0-base`` (``<repo>/../tomocpt-base``, created if missing) through ``subprocess``,
and cached in ``tests/.cache/legacy/`` (git-ignored). The cache key contains the base commit
and a hash of the generator script. Any failure raises ``RuntimeError`` (tests fail, never skip).
The checkpoints are tiny SwinUNETRs (``feature_size=12``, no dropout, ``use_checkpoint=False``)
trained for one epoch on random tensors, with ``train_continue=None``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

__all__ = ["BASE_REF", "REPO_ROOT", "WORKTREE", "ensure_worktree", "legacy_checkpoint",
           "legacy_ssl_checkpoint", "legacy_reference", "zenodo_checkpoint"]

BASE_REF = "v2-gate-0-base"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKTREE = Path(os.environ.get("TOMOCPT_BASE_WORKTREE", REPO_ROOT.parent / "tomocpt-base"))
SCRIPT = REPO_ROOT / "tests" / "fixtures" / "make_legacy_ckpt.py"
DEFAULT_CACHE = REPO_ROOT / "tests" / ".cache" / "legacy"


def _git(*args, cwd=REPO_ROOT) -> str:
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}:\n{res.stderr}")
    return res.stdout.strip()


def ensure_worktree() -> Path:
    """Return the path of a worktree checked out at ``v2-gate-0-base``; create it if missing."""
    base_commit = _git("rev-parse", f"{BASE_REF}^{{commit}}")
    if not (WORKTREE / "tomocpt" / "__init__.py").is_file():
        _git("worktree", "prune")
        _git("worktree", "add", "--detach", str(WORKTREE), BASE_REF)
    head = _git("rev-parse", "HEAD", cwd=WORKTREE)
    if head != base_commit:
        raise RuntimeError(f"worktree {WORKTREE} is at {head}, expected {BASE_REF} = {base_commit}")
    return WORKTREE


def _build(kind: str, tmp_cache_dir: Optional[Path]) -> Path:
    worktree = ensure_worktree()
    base_commit = _git("rev-parse", "HEAD", cwd=worktree)
    script_hash = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()[:10]
    cache = Path(tmp_cache_dir) if tmp_cache_dir is not None else DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    ckpt = cache / f"legacy_{kind}_{base_commit[:10]}_{script_hash}.ckpt"
    ref = ckpt.with_suffix(".ref.npz")
    if ckpt.is_file() and ref.is_file():
        return ckpt

    tmp_ckpt = cache / f".building_{kind}_{os.getpid()}.ckpt"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--kind", kind, "--out", str(tmp_ckpt), "--expect-root", str(worktree)],
        cwd=worktree, env=env, capture_output=True, text=True, timeout=1800,
    )
    if res.returncode != 0 or not tmp_ckpt.is_file():
        raise RuntimeError(
            f"building the legacy {kind} checkpoint in {worktree} failed (exit {res.returncode}):\n"
            f"--- stdout ---\n{res.stdout[-4000:]}\n--- stderr ---\n{res.stderr[-8000:]}"
        )
    os.replace(tmp_ckpt.with_suffix(".ref.npz"), ref)
    os.replace(tmp_ckpt, ckpt)
    return ckpt


def legacy_checkpoint(tmp_cache_dir: Optional[Path] = None) -> Path:
    """Tiny legacy ``BasePickingModel`` checkpoint (``train.mode = picking``)."""
    return _build("picking", tmp_cache_dir)


def legacy_ssl_checkpoint(tmp_cache_dir: Optional[Path] = None) -> Path:
    """Tiny legacy ``SelfSupervisedModel`` checkpoint (``train.mode = selfSupervised``)."""
    return _build("ssl", tmp_cache_dir)


def legacy_reference(ckpt: Path):
    """Outputs of the legacy model of ``ckpt`` (keys ``x``, ``logits``, ``hidden4``)."""
    import numpy as np

    return np.load(Path(ckpt).with_suffix(".ref.npz"))


def zenodo_checkpoint() -> Optional[Path]:
    """The public pretrained checkpoint if ``$TOMOCPT_ZENODO_CKPT`` points to a file, else None."""
    p = os.environ.get("TOMOCPT_ZENODO_CKPT")
    if p and Path(p).is_file():
        return Path(p)
    return None
