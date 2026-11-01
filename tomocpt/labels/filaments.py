"""Filament classes (Gate 6): centrelines from masks and coordinates, soft tube targets,
tracing of predicted maps, spline fits in physical units and RELION helical angles.

Conventions (every public function):

* polylines are ``[K, 3]`` arrays of ``x, y, z`` in native voxels, ordered along the filament;
* volumes and masks are ``(Z, Y, X)``;
* segments (chunk storage, CLAUDE.md Section 5.2) are ``[M, 8]`` float32 rows
  ``z0, y0, x0, z1, y1, x1, class_index, d_vox`` in voxels of the array they belong to.

The tube target is the particle profile of CLAUDE.md Section 1.3 applied to the perpendicular
distance from the centreline (the union of the segments, so the ends are round caps), with
``R = d_vox / 2``; several filaments of one channel are max-combined.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------------------------------
# small geometry helpers
# ------------------------------------------------------------------------------------------

_SS = 0.33 / 2.0  # sigma_s / R
_SL = 0.66 / 2.0  # sigma_l / R


def _profile_unit(u: np.ndarray) -> np.ndarray:
    """Section 1.3 profile as a function of ``u = r / R`` (1 at 0, 0 for ``u > 1``)."""
    u = np.asarray(u, dtype=np.float64)
    u2 = u * u
    v = 0.5 * np.exp(-u2 / (2.0 * _SS * _SS)) + 0.5 * np.exp(-u2 / (2.0 * _SL * _SL))
    v[~(u <= 1.0)] = 0.0
    return v


def _as_polyline(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    if len(p) > 1:
        keep = np.ones(len(p), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9
        p = p[keep]
    return p


def _cumlen(p: np.ndarray) -> np.ndarray:
    if len(p) < 2:
        return np.zeros(len(p))
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def polyline_length(p) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    return float(_cumlen(p)[-1]) if len(p) > 1 else 0.0


def segments_from_polyline(poly_xyz, class_index: int, d_vox: float) -> np.ndarray:
    """``[M, 8]`` float32 segment rows of one polyline (``x, y, z`` -> ``z, y, x``)."""
    p = np.asarray(poly_xyz, dtype=np.float64).reshape(-1, 3)
    if len(p) < 2:
        if len(p) == 1:  # a single point is a zero-length segment (a sphere)
            p = np.concatenate([p, p])
        else:
            return np.zeros((0, 8), np.float32)
    zyx = p[:, ::-1]
    n = len(p) - 1
    seg = np.concatenate([zyx[:-1], zyx[1:], np.full((n, 1), float(class_index)), np.full((n, 1), float(d_vox))],
                         axis=1)
    return seg.astype(np.float32)


def _segment_groups(segs: np.ndarray, group: int = 16):
    """Runs of at most ``group`` consecutive segments with the same class and diameter."""
    i, n = 0, len(segs)
    while i < n:
        j = i + 1
        while j < n and j - i < group and segs[j, 6] == segs[i, 6] and segs[j, 7] == segs[i, 7]:
            j += 1
        yield i, j
        i = j


def _visit_tube_boxes(segments, shape, reach_scale: float = 1.0):
    """Yield ``(class_index, R, box_slices, u)`` for groups of segments: ``u`` is the distance
    of every voxel centre in the box to the nearest segment of the group, divided by ``R``."""
    segs = np.asarray(segments, dtype=np.float32).reshape(-1, 8).astype(np.float64)
    shape = tuple(int(s) for s in shape)
    for i, j in _segment_groups(segs):
        g = segs[i:j]
        R = float(g[0, 7]) / 2.0
        if not R > 0:
            continue
        reach = R * reach_scale
        p0, p1 = g[:, 0:3], g[:, 3:6]
        lo_f = np.minimum(p0, p1).min(axis=0) - reach
        hi_f = np.maximum(p0, p1).max(axis=0) + reach
        lo = [max(0, int(math.ceil(lo_f[a]))) for a in range(3)]
        hi = [min(shape[a], int(math.floor(hi_f[a])) + 1) for a in range(3)]
        if any(h <= l for l, h in zip(lo, hi)):
            continue
        gz = np.arange(lo[0], hi[0], dtype=np.float64)
        gy = np.arange(lo[1], hi[1], dtype=np.float64)
        gx = np.arange(lo[2], hi[2], dtype=np.float64)
        d2min = None
        for s in range(len(g)):
            a, b = p0[s], p1[s]
            ab = b - a
            L2 = float(ab @ ab)
            wz = (gz - a[0])[:, None, None]
            wy = (gy - a[1])[None, :, None]
            wx = (gx - a[2])[None, None, :]
            if L2 > 0:
                t = (wz * ab[0] + wy * ab[1] + wx * ab[2]) / L2
                np.clip(t, 0.0, 1.0, out=t)
                dz, dy, dx = wz - t * ab[0], wy - t * ab[1], wx - t * ab[2]
                d2 = dz * dz + dy * dy + dx * dx
            else:
                d2 = wz * wz + wy * wy + wx * wx
                d2 = np.broadcast_to(d2, (len(gz), len(gy), len(gx)))
            d2min = d2 if d2min is None else np.minimum(d2min, d2)
        u = np.sqrt(d2min) / R
        ci = int(round(float(g[0, 6])))
        yield ci, R, (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2])), u


# ------------------------------------------------------------------------------------------
# targets and masks
# ------------------------------------------------------------------------------------------

def render_tubes(segments, out_shape_zyx, n_classes: int, border_policy: str = "clip") -> np.ndarray:
    """Soft tube targets ``[n_classes, *out_shape_zyx]`` (float32) from segments.

    ``segments``: ``[M, 8]`` = ``z0, y0, x0, z1, y1, x1, class_index, d_vox`` (array voxels).
    The value of a voxel is the Section 1.3 profile of its distance to the nearest segment of
    the channel (``R = d_vox / 2``): 1 on the centreline, constant along it, round caps at the
    ends, max-combined over filaments. Only the in-bounds part is drawn (``"clip"``; a filament
    is never dropped at the border, so ``"skip"`` behaves the same).
    """
    policy = str(getattr(border_policy, "value", border_policy))
    if policy not in ("clip", "skip"):
        raise ValueError(f"border_policy must be 'clip' or 'skip', got '{border_policy}'")
    shape = tuple(int(s) for s in out_shape_zyx)
    n_classes = int(n_classes)
    segs = np.asarray(segments, dtype=np.float32).reshape(-1, 8)
    out = np.zeros((n_classes,) + shape, dtype=np.float32)
    if len(segs) == 0:
        return out
    cls = np.round(segs[:, 6]).astype(np.int64)
    if np.any((cls < 0) | (cls >= n_classes)):
        raise ValueError(f"segment class index outside [0, {n_classes})")
    umin = np.full((n_classes,) + shape, np.inf, dtype=np.float32)
    touched = set()
    for ci, _, box, u in _visit_tube_boxes(segs, shape):
        sl = (ci,) + box
        np.minimum(umin[sl], u.astype(np.float32), out=umin[sl])
        touched.add(ci)
    for ci in touched:
        out[ci] = _profile_unit(umin[ci]).astype(np.float32)
    return out


def centerlines_to_mask(polylines: Sequence[np.ndarray], shape_zyx, apix: float, diameter_ang: float) -> np.ndarray:
    """Binary tube mask (``uint8``, tomogram shape): voxels whose centre lies within the class
    radius ``diameter_ang / 2`` of a polyline (``x, y, z`` native voxels)."""
    shape = tuple(int(s) for s in shape_zyx)
    mask = np.zeros(shape, dtype=np.uint8)
    d_vox = float(diameter_ang) / float(apix)
    segs = [segments_from_polyline(p, 0, d_vox) for p in polylines if len(np.asarray(p).reshape(-1, 3))]
    if not segs:
        return mask
    for _, _, box, u in _visit_tube_boxes(np.concatenate(segs), shape):
        mask[box] |= (u <= 1.0).astype(np.uint8)
    return mask


# ------------------------------------------------------------------------------------------
# resampling and spline fits
# ------------------------------------------------------------------------------------------

def resample_polyline(polyline, spacing_vox: float) -> np.ndarray:
    """Points every ``spacing_vox`` voxels of arc length along a polyline, from its first
    point; the last point of the polyline is appended when it lies beyond the last sample."""
    p = _as_polyline(polyline)
    spacing = float(spacing_vox)
    if not spacing > 0:
        raise ValueError("spacing_vox must be > 0")
    if len(p) < 2:
        return p.copy()
    s = _cumlen(p)
    L = s[-1]
    n = int(math.floor(L / spacing + 1e-9))
    t = np.arange(n + 1, dtype=np.float64) * spacing
    if L - t[-1] > 1e-6 * spacing:
        t = np.append(t, L)
    out = np.stack([np.interp(t, s, p[:, a]) for a in range(3)], axis=1)
    return out


def _consistent_signs(t: np.ndarray) -> np.ndarray:
    t = np.array(t, dtype=np.float64, copy=True)
    for i in range(1, len(t)):
        if float(t[i] @ t[i - 1]) < 0:
            t[i] = -t[i]
    return t


def fit_filament_spline(polyline, apix: float, smoothing_ang: float, spacing_ang: float
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic smoothing spline through a polyline, sampled at equal arc length.

    The polyline (``x, y, z`` native voxels) is converted to A and fitted with
    ``scipy.interpolate.splprep`` (``k = min(3, n - 1)``, ``s = n * smoothing_ang**2``, i.e. an
    RMS deviation of about ``smoothing_ang`` A). Points are placed every ``spacing_ang`` A of arc
    length (dense numerical integration), from the first point; the end point is appended when
    it lies beyond the last sample.

    Returns ``(points [K, 3] x, y, z native voxels, unit tangents [K, 3] with one consistent
    direction, arc_length_ang [K] from the first point)``.
    """
    from scipy.interpolate import splev, splprep

    apix = float(apix)
    spacing = float(spacing_ang)
    if not spacing > 0:
        raise ValueError("spacing_ang must be > 0")
    p = _as_polyline(polyline) * apix
    n = len(p)
    if n == 0:
        z = np.zeros((0, 3))
        return z, z.copy(), np.zeros(0)
    if n == 1:
        return p / apix, np.array([[0.0, 0.0, 1.0]]), np.zeros(1)
    k = min(3, n - 1)
    s = float(n) * float(smoothing_ang) ** 2
    u0 = _cumlen(p)
    u0 = u0 / u0[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tck, _ = splprep(p.T, u=u0, k=k, s=s)
        except Exception:
            tck, _ = splprep(p.T, u=u0, k=1, s=0.0)
    # dense arc length
    L_chord = float(_cumlen(p)[-1])
    m = int(min(2_000_000, max(2001, 20 * L_chord / max(min(spacing, apix), 1e-6))))
    uu = np.linspace(0.0, 1.0, m)
    dense = np.stack(splev(uu, tck), axis=1)
    sd = _cumlen(dense)
    L = float(sd[-1])
    nst = int(math.floor(L / spacing + 1e-9))
    arc = np.arange(nst + 1, dtype=np.float64) * spacing
    if L - arc[-1] > 1e-6 * spacing:
        arc = np.append(arc, L)
    u = np.interp(arc, sd, uu)
    pts = np.stack(splev(u, tck), axis=1)
    der = np.stack(splev(u, tck, der=1), axis=1)
    norm = np.linalg.norm(der, axis=1, keepdims=True)
    bad = norm[:, 0] < 1e-12
    if bad.any():  # degenerate derivative: use the chord direction
        fd = np.gradient(pts, axis=0) if len(pts) > 1 else np.array([[0.0, 0.0, 1.0]])
        der[bad] = fd[bad]
        norm = np.linalg.norm(der, axis=1, keepdims=True)
    tng = _consistent_signs(der / np.maximum(norm, 1e-12))
    return pts / apix, tng, arc


def tangents_to_relion_angles(tangents) -> Tuple[np.ndarray, np.ndarray]:
    """``(tilt_deg, psi_deg)`` per unit tangent, RELION ZYZ convention (``src/euler.cpp``,
    ``Euler_angles2matrix(rot=0, tilt, psi)``): the third column of that matrix, the image of the
    reference filament axis Z, is ``(-cos psi sin tilt, sin psi sin tilt, cos tilt) = t``.
    Polarity is unknown (``t`` and ``-t`` give the same filament; RELION's flip ratio handles
    it). ``rot`` (the spin about the axis) is not determined."""
    t = np.asarray(tangents, dtype=np.float64).reshape(-1, 3)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.maximum(n, 1e-12)
    tilt = np.degrees(np.arccos(np.clip(t[:, 2], -1.0, 1.0)))
    psi = np.degrees(np.arctan2(t[:, 1], -t[:, 0]))
    on_axis = np.hypot(t[:, 0], t[:, 1]) < 1e-12
    psi[on_axis] = 0.0
    return tilt, psi


# ------------------------------------------------------------------------------------------
# skeleton graph -> polylines
# ------------------------------------------------------------------------------------------

_OFFS = np.array([(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (dz, dy, dx) != (0, 0, 0)], dtype=np.int64)


class _Graph:
    """Skeleton graph: nodes are clusters of non-degree-2 voxels; edges are voxel paths."""

    def __init__(self, skel: np.ndarray):
        vox = np.argwhere(skel)
        self.vox = vox.astype(np.float64)
        index = {tuple(v): i for i, v in enumerate(vox.tolist())}
        nbrs: List[List[int]] = []
        for v in vox:
            lst = []
            for o in _OFFS:
                j = index.get((int(v[0] + o[0]), int(v[1] + o[1]), int(v[2] + o[2])))
                if j is not None:
                    lst.append(j)
            nbrs.append(lst)
        self.nbrs = nbrs
        deg = np.array([len(n) for n in nbrs], dtype=np.int64)
        is_node = deg != 2
        # cluster adjacent node voxels
        node_of = -np.ones(len(vox), dtype=np.int64)
        n_nodes = 0
        for i in np.nonzero(is_node)[0]:
            if node_of[i] >= 0:
                continue
            stack = [i]
            node_of[i] = n_nodes
            while stack:
                a = stack.pop()
                for b in nbrs[a]:
                    if is_node[b] and node_of[b] < 0:
                        node_of[b] = n_nodes
                        stack.append(b)
            n_nodes += 1
        self.node_of = node_of
        self.node_pos = [self.vox[node_of == k].mean(axis=0) for k in range(n_nodes)]
        # edges
        edges = []  # (u, v, [voxel indices from u side to v side])
        used = np.zeros(len(vox), dtype=bool)
        for i in np.nonzero(is_node)[0]:
            for b in nbrs[i]:
                if is_node[b]:
                    continue
                if used[b]:
                    continue
                path = [i, b]
                used[b] = True
                prev, cur = i, b
                ok = True
                while not is_node[cur]:
                    nxt = [c for c in nbrs[cur] if c != prev]
                    if not nxt:  # cannot happen for a degree-2 voxel
                        ok = False
                        break
                    c = nxt[0]
                    path.append(c)
                    if not is_node[c]:
                        used[c] = True
                    prev, cur = cur, c
                if ok:
                    edges.append([int(node_of[path[0]]), int(node_of[path[-1]]), path])
        # direct node-node adjacency between different clusters cannot happen (they would be one
        # cluster). Loops without any node: break them at one voxel.
        for i in range(len(vox)):
            if is_node[i] or used[i]:
                continue
            path, prev, cur = [i], -1, i
            used[i] = True
            while True:
                nxt = [c for c in nbrs[cur] if c != prev and not used[c]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                used[cur] = True
                path.append(cur)
            k = len(self.node_pos)
            self.node_pos.append(self.vox[path[0]])
            k2 = len(self.node_pos)
            self.node_pos.append(self.vox[path[-1]])
            edges.append([k, k2, path])
        self.edges = edges

    def edge_points(self, e) -> np.ndarray:
        u, v, path = e
        pts = self.vox[path]
        return pts


def _edge_len(pts: np.ndarray) -> float:
    return float(_cumlen(pts)[-1]) if len(pts) > 1 else 0.0


def _end_direction(pts: np.ndarray, centre: np.ndarray, near: float, far: float) -> np.ndarray:
    """Outgoing direction of a path that starts at a node ``centre``: from the centre to the
    mean of the path points at arc distance ``[near, far]`` from the start."""
    s = _cumlen(pts)
    sel = (s >= near) & (s <= far)
    if not sel.any():
        sel = s >= s[-1] / 2.0
    d = pts[sel].mean(axis=0) - centre
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.zeros(3)


def _graph_polylines(skel: np.ndarray, R: float) -> List[np.ndarray]:
    """Polylines (``z, y, x`` voxels) of a skeleton: spurs shorter than ``R`` pruned, short
    junction-junction bridges contracted, arms paired at junctions by direction."""
    g = _Graph(skel)
    n_nodes = len(g.node_pos)
    edges = [[u, v, g.edge_points([u, v, p])] for u, v, p in g.edges]
    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def degrees():
        d = np.zeros(n_nodes, dtype=np.int64)
        for u, v, _ in edges:
            d[find(u)] += 1
            d[find(v)] += 1
        return d

    # 1. prune spurs (endpoint - junction edges shorter than R), repeatedly
    changed = True
    while changed:
        changed = False
        deg = degrees()
        keep = []
        for e in edges:
            u, v = find(e[0]), find(e[1])
            L = _edge_len(e[2])
            spur = (deg[u] == 1 and deg[v] >= 3) or (deg[v] == 1 and deg[u] >= 3)
            small_loop = u == v and deg[u] > 2 and L < 4.0 * R
            if (spur and L < R) or small_loop:
                changed = True
                continue
            keep.append(e)
        edges = keep
    # 2. contract junction-junction bridges shorter than the diameter
    deg = degrees()
    keep = []
    for e in edges:
        u, v = find(e[0]), find(e[1])
        if u != v and deg[u] >= 3 and deg[v] >= 3 and _edge_len(e[2]) < 2.0 * R:
            parent[v] = u
            continue
        keep.append(e)
    edges = keep
    # merged node centres
    groups: Dict[int, List[int]] = {}
    for k in range(n_nodes):
        groups.setdefault(find(k), []).append(k)
    centre = {r: np.mean([g.node_pos[k] for k in ks], axis=0) for r, ks in groups.items()}

    # half-edge ends: (edge index, side 0 = start / 1 = end)
    ends_at: Dict[int, List[Tuple[int, int]]] = {}
    for i, (u, v, _) in enumerate(edges):
        ends_at.setdefault(find(u), []).append((i, 0))
        ends_at.setdefault(find(v), []).append((i, 1))

    def oriented(i, side):  # points of edge i leaving the node at `side`
        pts = edges[i][2]
        return pts if side == 0 else pts[::-1]

    partner: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for node, ends in ends_at.items():
        if len(ends) == 2:
            a, b = ends
            if a[0] != b[0]:
                partner[a], partner[b] = b, a
            continue
        if len(ends) < 3:
            continue
        c = centre[node]
        dirs = [_end_direction(oriented(i, s), c, 0.5 * R, 3.0 * R) for i, s in ends]
        cand = []
        for x in range(len(ends)):
            for y in range(x + 1, len(ends)):
                if ends[x][0] == ends[y][0]:
                    continue
                cand.append((float(dirs[x] @ dirs[y]), x, y))
        cand.sort()
        taken = set()
        for dot, x, y in cand:
            if dot > -0.5:
                break
            if x in taken or y in taken:
                continue
            taken.update((x, y))
            partner[ends[x]], partner[ends[y]] = ends[y], ends[x]

    # chains
    visited = set()
    out = []

    def walk(i, side):
        """Start at edge i entering through `side`; follow partners."""
        pieces = []
        cur, s = i, side
        while True:
            visited.add(cur)
            pts = oriented(cur, s)
            pieces.append((cur, s, pts))
            far_end = (cur, 1 - s)
            nxt = partner.get(far_end)
            if nxt is None or nxt[0] in visited:
                break
            cur, s = nxt
        return pieces

    def join(pieces):
        """Concatenate oriented edge paths; at a junction drop points within R of its centre."""
        allp = []
        for k, (i, s, pts) in enumerate(pieces):
            pts = np.asarray(pts, dtype=np.float64)
            node_start = find(edges[i][0] if s == 0 else edges[i][1])
            node_end = find(edges[i][1] if s == 0 else edges[i][0])
            m = np.ones(len(pts), dtype=bool)
            if k > 0 and len(ends_at.get(node_start, [])) >= 3:
                m &= np.linalg.norm(pts - centre[node_start], axis=1) > R
            if k < len(pieces) - 1 and len(ends_at.get(node_end, [])) >= 3:
                m &= np.linalg.norm(pts - centre[node_end], axis=1) > R
            sel = pts[m] if m.sum() >= 2 else pts
            allp.append(sel)
        return np.concatenate(allp, axis=0) if allp else np.zeros((0, 3))

    # open chains first: start from ends without a partner
    for i in range(len(edges)):
        for side in (0, 1):
            if i in visited:
                continue
            if (i, side) not in partner:
                out.append(join(walk(i, side)))
    for i in range(len(edges)):  # closed chains
        if i not in visited:
            out.append(join(walk(i, 0)))
    return [p for p in out if len(p) >= 2]


def _extend_to_boundary(poly_zyx: np.ndarray, region: np.ndarray, R: float, step: float = 0.5) -> np.ndarray:
    """Extend both ends of a polyline along its end tangent while inside ``region``."""
    p = np.asarray(poly_zyx, dtype=np.float64)
    if len(p) < 2:
        return p
    shape = np.array(region.shape)

    def inside(q):
        i = np.round(q).astype(np.int64)
        if np.any(i < 0) or np.any(i >= shape):
            return False
        return bool(region[tuple(i)])

    def ext(pts):
        s = _cumlen(pts)
        sel = s <= min(2.0 * R, s[-1])
        base = pts[sel]
        if len(base) < 2:
            base = pts[:2]
        d = base[0] - base[-1]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return pts
        d /= n
        add = []
        q = pts[0].copy()
        for _ in range(int(4 * R / step) + 1):
            q2 = q + step * d
            if not inside(q2):
                break
            q = q2
            add.append(q.copy())
        if add:
            return np.concatenate([np.array(add[::-1]), pts], axis=0)
        return pts

    p = ext(p)
    p = ext(p[::-1])[::-1]
    return p


def _join_gaps(polys: List[np.ndarray], max_gap: float, R: float) -> List[np.ndarray]:
    """Join polyline ends closer than ``max_gap`` whose directions agree (end tangents roughly
    opposite and aligned with the gap)."""
    polys = [np.asarray(p, dtype=np.float64) for p in polys if len(p) >= 2]

    def end_dir(p, at_start):
        q = p if at_start else p[::-1]
        s = _cumlen(q)
        sel = s <= min(2.0 * R, s[-1])
        b = q[sel] if sel.sum() >= 2 else q[:2]
        d = b[0] - b[-1]
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.zeros(3)

    merged = True
    while merged and len(polys) > 1:
        merged = False
        best = None
        for a in range(len(polys)):
            for b in range(len(polys)):
                if a == b:
                    continue
                for ea in (0, 1):
                    for eb in (0, 1):
                        pa = polys[a][0] if ea == 0 else polys[a][-1]
                        pb = polys[b][0] if eb == 0 else polys[b][-1]
                        gap = float(np.linalg.norm(pb - pa))
                        if gap > max_gap:
                            continue
                        da, db = end_dir(polys[a], ea == 0), end_dir(polys[b], eb == 0)
                        if float(da @ db) > -0.7:
                            continue
                        if gap > 1e-6:
                            gv = (pb - pa) / gap
                            if float(da @ gv) < 0.5:
                                continue
                        if best is None or gap < best[0]:
                            best = (gap, a, ea, b, eb)
        if best is not None:
            _, a, ea, b, eb = best
            pa = polys[a] if ea == 1 else polys[a][::-1]   # ends at the join
            pb = polys[b] if eb == 0 else polys[b][::-1]   # starts at the join
            new = np.concatenate([pa, pb], axis=0)
            polys = [p for k, p in enumerate(polys) if k not in (a, b)] + [new]
            merged = True
    return polys


def _region_polylines(region: np.ndarray, apix: float, diameter_ang: float, min_length_ang: float,
                      smooth_ang: Optional[float] = None) -> List[np.ndarray]:
    """Shared core of :func:`mask_to_centerlines` and :func:`trace_filaments`: skeletonise each
    connected component of ``region``, trace, extend the ends to the region boundary, join short
    gaps, smooth (spline, sampled every voxel) and drop polylines shorter than
    ``min_length_ang``. Returns ``x, y, z`` voxel polylines."""
    from scipy import ndimage
    from skimage.morphology import skeletonize

    R = float(diameter_ang) / 2.0 / float(apix)
    region = np.asarray(region, dtype=bool)
    lab, n = ndimage.label(region, structure=np.ones((3, 3, 3), dtype=bool))
    polys: List[np.ndarray] = []
    margin = 2
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - margin), min(dim, s.stop + margin)) for s, dim in zip(sl, region.shape))
        comp = lab[sl] == k
        if comp.sum() < 2:
            continue
        skel = skeletonize(comp)
        skel = np.asarray(skel > 0)
        if not skel.any():
            continue
        offset = np.array([s.start for s in sl], dtype=np.float64)
        for p in _graph_polylines(skel, max(R, 1.0)):
            p = _extend_to_boundary(p, comp, max(R, 1.0))
            polys.append(p + offset)
    polys = _join_gaps(polys, 0.5 * 2.0 * R + 1.0, max(R, 1.0))
    out = []
    sm = (0.5 * float(apix)) if smooth_ang is None else float(smooth_ang)
    for p in polys:
        xyz = p[:, ::-1]
        if len(xyz) >= 2:
            pts, _, _ = fit_filament_spline(xyz, apix, sm, float(apix))
        else:
            pts = xyz
        if polyline_length(pts) * float(apix) < float(min_length_ang):
            continue
        out.append(pts)
    return out


def _fill_lumen(mask: np.ndarray, R: float) -> np.ndarray:
    """Morphological closing with a ball of radius ``R`` (via distance transforms): fills the
    lumen of hollow tube masks and holes smaller than the tube."""
    from scipy import ndimage

    if R < 1.0:
        return mask
    pad = int(math.ceil(R)) + 2
    m = np.pad(mask, pad)
    dil = ndimage.distance_transform_edt(~m) <= R
    clo = ndimage.distance_transform_edt(dil) > R
    out = clo[pad:-pad, pad:-pad, pad:-pad] | mask
    return ndimage.binary_fill_holes(out)


def mask_to_centerlines(mask, apix: float, diameter_ang: float, min_length_ang: float) -> List[np.ndarray]:
    """Centrelines of a filament segmentation mask (``(Z, Y, X)``, bool or integer; solid or
    hollow tubes): close holes and the lumen at the tube radius, skeletonise, trace (spurs
    shorter than the radius pruned, crossings resolved by direction, gaps shorter than half the
    diameter joined), extend the ends to the mask boundary, smooth. Polylines shorter than
    ``min_length_ang`` are dropped. Returns ``[K, 3]`` ``x, y, z`` native voxels."""
    m = np.asarray(mask) > 0
    if not m.any():
        return []
    R = float(diameter_ang) / 2.0 / float(apix)
    # Work at a coarser grid when the tube is wide (skeletonisation cost and noise).
    b = max(1, int(R // 6.0))
    if b > 1:
        Z, Y, X = (s // b * b for s in m.shape)
        mb = m[:Z, :Y, :X].reshape(Z // b, b, Y // b, b, X // b, b).mean(axis=(1, 3, 5)) >= 0.5
        region = _fill_lumen(mb, R / b)
        polys = _region_polylines(region, apix * b, diameter_ang, min_length_ang)
        return [p * b + (b - 1) / 2.0 for p in polys]
    region = _fill_lumen(m, R)
    return _region_polylines(region, apix, diameter_ang, min_length_ang)


def trace_filaments(prob_map, threshold: float, apix: float, diameter_ang: float, min_length_ang: float,
                    spacing_ang: float, smoothing_ang: float = 10.0) -> List[np.ndarray]:
    """Polylines of one filament channel of a probability map (``(Z, Y, X)``, native voxels):
    threshold, skeletonise, trace (split at branch points, crossings paired by direction, gaps
    shorter than half the diameter joined), drop polylines shorter than ``min_length_ang``,
    fit a smoothing spline (``smoothing_ang``) and sample every ``spacing_ang`` A. Returns
    ``[K, 3]`` ``x, y, z`` native voxels."""
    return [pts for pts, _, _ in _trace_and_fit(prob_map, threshold, apix, diameter_ang, min_length_ang,
                                                spacing_ang, smoothing_ang)]


def _trace_and_fit(prob_map, threshold, apix, diameter_ang, min_length_ang, spacing_ang, smoothing_ang):
    prob = np.asarray(prob_map, dtype=np.float32)
    region = prob > float(threshold)
    if not region.any():
        return []
    polys = _region_polylines(region, apix, diameter_ang, min_length_ang)
    out = []
    for p in polys:
        pts, tng, arc = fit_filament_spline(p, apix, smoothing_ang, spacing_ang)
        if len(pts) < 2 or arc[-1] < float(min_length_ang):
            continue
        out.append((pts, tng, arc))
    return out


# ------------------------------------------------------------------------------------------
# filament coordinate files
# ------------------------------------------------------------------------------------------

def _chain_nearest(points: np.ndarray) -> np.ndarray:
    """Order unordered points of one filament by nearest-neighbour chaining from an end."""
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    if n < 3:
        return p
    a = int(np.argmax(np.linalg.norm(p - p[0], axis=1)))
    start = int(np.argmax(np.linalg.norm(p - p[a], axis=1)))
    start = a if start == a else start
    # start at the farthest point from `a` is the other end; begin at `a` (an end)
    order = [a]
    left = np.ones(n, dtype=bool)
    left[a] = False
    cur = a
    for _ in range(n - 1):
        idx = np.nonzero(left)[0]
        d = np.linalg.norm(p[idx] - p[cur], axis=1)
        cur = int(idx[np.argmin(d)])
        order.append(cur)
        left[cur] = False
    return p[order]


def read_filament_coords(path, apix: float) -> List[np.ndarray]:
    """Filaments of a coordinate file, one polyline (``x, y, z`` native voxels) per filament, in
    order along it.

    * STAR (RELION 3.1/5.0 or Warp style, as ``predict`` writes): ``rlnCoordinateX/Y/Z`` in
      voxels (scaled by ``rlnImagePixelSize / apix`` when the optics table gives another pixel
      size), grouped by ``rlnHelicalTubeID`` (and the tomogram name when there are several),
      ordered by ``rlnHelicalTrackLengthAngst`` when present, else by nearest-neighbour chaining.
    * IMOD model: one polyline per contour (points in voxels, in contour order).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".mod", ".imod"):
        import imodmodel

        df = imodmodel.read(str(path))
        out = []
        for _, g in df.groupby(["object_id", "contour_id"], sort=True):
            out.append(g[["x", "y", "z"]].to_numpy(np.float64))
        return [p for p in out if len(p)]

    import starfile

    data = starfile.read(str(path))
    optics = None
    if isinstance(data, dict):
        df = data.get("particles")
        if df is None:
            df = next(v for v in data.values() if hasattr(v, "columns") and "rlnCoordinateX" in v.columns)
        optics = data.get("optics")
    else:
        df = data
    scale = 1.0
    if optics is not None and "rlnImagePixelSize" in getattr(optics, "columns", []) and apix:
        pix = float(optics["rlnImagePixelSize"].iloc[0])
        if pix > 0 and abs(pix - float(apix)) > 1e-3 * float(apix):
            scale = pix / float(apix)
    cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    missing = [c for c in cols if c not in df.columns]
    if missing and "rlnCenteredCoordinateXAngst" in df.columns:
        raise ValueError(f"{path}: RELION 5 centred coordinates (rlnCenteredCoordinate*Angst) need the "
                         "tomogram size to convert; export rlnCoordinateX/Y/Z (voxels) instead")
    if missing:
        raise ValueError(f"{path}: STAR file lacks {missing}")
    # subpixel shifts, as labels.star does for particles: rlnOrigin* (pixels, RELION 3.0) or
    # rlnOrigin*Angst (RELION 3.1+, divided by the STAR pixel size); coordinate - origin
    px_origin = [f"rlnOrigin{a}" for a in "XYZ"]
    ang_origin = [f"rlnOrigin{a}Angst" for a in "XYZ"]
    if all(c in df.columns for c in ang_origin) or all(c in df.columns for c in px_origin):
        df = df.copy()
        if all(c in df.columns for c in ang_origin):
            star_pix = float(apix) * scale if apix else 1.0  # rlnImagePixelSize, or apix when absent
            shifts = df[ang_origin].to_numpy(np.float64) / star_pix
        else:
            shifts = df[px_origin].to_numpy(np.float64)
        df[cols] = df[cols].to_numpy(np.float64) - shifts
    keys = []
    for k in ("rlnMicrographName", "rlnTomoName"):
        if k in df.columns and df[k].nunique() > 1:
            keys.append(k)
    if "rlnHelicalTubeID" in df.columns:
        keys.append("rlnHelicalTubeID")
    groups = df.groupby(keys, sort=True) if keys else [((), df)]
    out = []
    for _, g in groups:
        if "rlnHelicalTrackLengthAngst" in g.columns:
            g = g.sort_values("rlnHelicalTrackLengthAngst", kind="stable")
            pts = g[cols].to_numpy(np.float64)
        else:
            pts = _chain_nearest(g[cols].to_numpy(np.float64))
        out.append(pts * scale)
    return [p for p in out if len(p)]


# ------------------------------------------------------------------------------------------
# prepare_data (T6.2): filament CSVs (and optional tube masks) per tomogram
# ------------------------------------------------------------------------------------------

FILAMENT_CSV_COLUMNS = ["filament_id", "x", "y", "z", "class_number", "diameter_ang"]
FILAMENTS_DIR_NAME = "filaments"
FILAMENT_MASKS_DIR_NAME = "filament_masks"
_MASK_SUFFIXES = (".mrc", ".rec", ".map")
_COORD_SUFFIXES = (".star", ".mod", ".imod")


def _tomogram_files(tomo_dir: Path) -> Dict[str, Path]:
    files = sorted(list(Path(tomo_dir).glob("*.mrc")) + list(Path(tomo_dir).glob("*.rec")))
    return {f.stem: f for f in files}


def match_files_to_tomograms(files: Sequence[Path], tomo_stems: Sequence[str]) -> Dict[str, Path]:
    """``{tomogram stem: file}``: exact stem match first, then the longest tomogram stem that the
    file stem starts with (followed by ``_``, ``-`` or ``.``), then a unique tomogram whose stem
    starts with the file stem (the rule of ``labels.helpers.match_data_to_tomograms``)."""
    stems = list(tomo_stems)
    out: Dict[str, Path] = {}
    for f in sorted(Path(x) for x in files):
        s = f.stem
        hit = None
        if s in stems:
            hit = s
        else:
            pre = [t for t in stems if s.startswith(t) and s[len(t):len(t) + 1] in ("_", "-", ".")]
            if pre:
                hit = max(pre, key=len)
            else:
                post = [t for t in stems if t.startswith(s) and t[len(s):len(s) + 1] in ("_", "-", ".")]
                if len(post) == 1:
                    hit = post[0]
        if hit is None:
            continue
        if hit in out and out[hit].stem != hit:
            if f.stem == hit:
                out[hit] = f
            continue
        out.setdefault(hit, f)
    return out


def write_filament_csv(path, polylines_by_class: Sequence[Tuple[int, float, Sequence[np.ndarray]]]) -> Path:
    """``filament_id, x, y, z, class_number, diameter_ang`` (native voxels; ids 1-based per
    tomogram, rows of a filament in order along it)."""
    import pandas as pd

    rows = []
    fid = 0
    for number, diameter, polys in polylines_by_class:
        for p in polys:
            p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
            if len(p) < 2:
                continue
            fid += 1
            for x, y, z in p:
                rows.append((fid, float(x), float(y), float(z), int(number), float(diameter)))
    df = pd.DataFrame(rows, columns=FILAMENT_CSV_COLUMNS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def read_filament_csv(path) -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_csv(path)
    missing = set(FILAMENT_CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Filament file {path} lacks columns {sorted(missing)}")
    return df


def filament_csv_to_segments(df, apix_native: float, apix_work: float, class_table,
                             coord_map=None, shift_zyx=None) -> np.ndarray:
    """Rows of a filament CSV (native voxels) -> ``[M, 8]`` float64 segments in working voxels
    (``z0, y0, x0, z1, y1, x1, class_index, d_vox``), mapped like ``labels.helpers.points_to_working``
    and shifted by ``shift_zyx`` (the padding). Consecutive rows of one filament form a segment."""
    if df is None or len(df) == 0:
        return np.zeros((0, 8), dtype=np.float64)
    index = {int(r.number): i for i, r in enumerate(class_table)}
    diam = [float(r.diameter_ang) for r in class_table]
    numbers = sorted({int(n) for n in df["class_number"]})
    unknown = [n for n in numbers if n not in index]
    if unknown:
        raise ValueError(f"Filament class number(s) {unknown} are not in the class table {sorted(index)}")
    f = float(apix_native) / float(apix_work)
    if coord_map is None:
        scale, offset = np.full(3, f), np.zeros(3)
    else:
        scale = np.asarray(coord_map[0], dtype=np.float64).reshape(3)
        offset = np.asarray(coord_map[1], dtype=np.float64).reshape(3)
    shift = np.zeros(3) if shift_zyx is None else np.asarray(shift_zyx, dtype=np.float64).reshape(3)
    out = []
    for _, g in df.groupby("filament_id", sort=True):
        zyx = np.stack([np.asarray(g[c], dtype=np.float64) * scale[a] + offset[a] + shift[a]
                        for a, c in enumerate(("z", "y", "x"))], axis=1)
        ci = index[int(g["class_number"].iloc[0])]
        d = diam[ci] / float(apix_work)
        if len(zyx) == 1:
            zyx = np.concatenate([zyx, zyx])
        n = len(zyx) - 1
        out.append(np.concatenate([zyx[:-1], zyx[1:], np.full((n, 1), float(ci)), np.full((n, 1), d)], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 8), dtype=np.float64)


def segments_near_box(segs: np.ndarray, lo_zyx, size: int) -> np.ndarray:
    """Bool ``[M]``: segments that pass within ``R`` of the box of voxel centres
    ``[lo, lo + size - 1]`` (conservative: the bounding box of the segment grown by ``R``)."""
    segs = np.asarray(segs).reshape(-1, 8)
    if len(segs) == 0:
        return np.zeros(0, dtype=bool)
    lo = np.asarray(lo_zyx, dtype=np.float64).reshape(1, 3)
    hi = lo + (size - 1)
    R = segs[:, 7:8].astype(np.float64) / 2.0
    smin = np.minimum(segs[:, 0:3], segs[:, 3:6]).astype(np.float64) - R
    smax = np.maximum(segs[:, 0:3], segs[:, 3:6]).astype(np.float64) + R
    return np.all((smax >= lo) & (smin <= hi), axis=1)


def prepare_filament_labels(prep_cfg, dataset_dir, tomo_dir, class_table, tracking_csv: Optional[Path] = None,
                            apix_override: Optional[float] = None) -> Optional[Path]:
    """Filament labels of one ``prepare_data`` dataset.

    For every tomogram of ``tomo_dir`` with a source file of a filament class (masks:
    ``mask_to_centerlines``; coordinates: ``read_filament_coords``), every centreline is fitted
    with :func:`fit_filament_spline` (``prepData.filament_smoothing_ang``, points every
    ``prepData.filament_label_spacing_ang``, default 0.5 x voxel size) and written to
    ``<dataset_dir>/filaments/<tomo>.csv``; with ``write_filament_masks`` also a tube mask per
    class. The tracking CSV gets a ``filaments_path`` column and the filament classes in
    ``class_annotated``; tomograms with filaments but no particle rows get rows of their own.
    Returns the tracking CSV path (None when there is no filament class).
    """
    import mrcfile
    import pandas as pd

    from tomocpt.configManager.configManager import safe_getattr
    from tomocpt.dataManager.scaling import read_apix
    from tomocpt.labels.helpers import write_coords_csv
    from tomocpt.labels.multiclass import FILAMENT, filament_sources

    sources = filament_sources(prep_cfg)
    if not sources:
        return tracking_csv
    by_number = {int(r.number): r for r in class_table}
    for n in sources:
        if n not in by_number:
            raise ValueError(f"Filament class {n} has no diameter (add it to class_diameters)")
        if str(getattr(by_number[n].kind, "value", by_number[n].kind)) != FILAMENT:
            raise ValueError(f"Class {n} is listed as a filament but the class table has it as a particle")
    smoothing = float(safe_getattr(prep_cfg, "filament_smoothing_ang", 10.0))
    spacing_cfg = safe_getattr(prep_cfg, "filament_label_spacing_ang", None)
    write_masks = bool(safe_getattr(prep_cfg, "write_filament_masks", False))

    dataset_dir = Path(dataset_dir)
    tomos = _tomogram_files(Path(tomo_dir))
    per_class_files: Dict[int, Dict[str, Path]] = {}
    for n, (kind, d) in sorted(sources.items()):
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(f"Filament source directory of class {n} not found: {d}")
        suffixes = _MASK_SUFFIXES if kind == "masks" else _COORD_SUFFIXES
        files = [f for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in suffixes]
        per_class_files[n] = match_files_to_tomograms(files, list(tomos))

    rows_extra: Dict[str, dict] = {}
    for stem, tomo_file in tomos.items():
        have = [n for n in sorted(sources) if stem in per_class_files[n]]
        if not have:
            continue
        apix = float(read_apix(tomo_file, apix_override))
        with mrcfile.open(str(tomo_file), permissive=True, header_only=True) as m:
            shape = (int(m.header.nz), int(m.header.ny), int(m.header.nx))
        spacing = float(spacing_cfg) if spacing_cfg else 0.5 * apix
        entries = []
        for n in have:
            row = by_number[n]
            src = per_class_files[n][stem]
            kind = sources[n][0]
            if kind == "masks":
                with mrcfile.open(str(src), permissive=True) as m:
                    mask = np.asarray(m.data) > 0
                if tuple(mask.shape) != shape:
                    raise ValueError(f"Filament mask {src} has shape {mask.shape}, tomogram {tomo_file} {shape}")
                raw = mask_to_centerlines(mask, apix, row.diameter_ang, 0.5 * row.diameter_ang)
            else:
                raw = read_filament_coords(src, apix)
            polys = []
            for p in raw:
                if len(np.asarray(p).reshape(-1, 3)) < 2:
                    continue
                pts, _, _ = fit_filament_spline(p, apix, smoothing, spacing)
                if len(pts) >= 2:
                    polys.append(pts)
            entries.append((n, float(row.diameter_ang), polys))
            if write_masks:
                mpath = dataset_dir / FILAMENT_MASKS_DIR_NAME / row.name / f"{stem}.mrc"
                mpath.parent.mkdir(parents=True, exist_ok=True)
                tube = centerlines_to_mask(polys, shape, apix, row.diameter_ang)
                mrcfile.write(str(mpath), data=tube.astype(np.int8), voxel_size=apix, overwrite=True)
        csv_path = write_filament_csv(dataset_dir / FILAMENTS_DIR_NAME / f"{stem}.csv", entries)
        rows_extra[stem] = {"tomogram_path": str(tomo_file), "filaments_path": str(csv_path),
                            "filament_classes": have, "apix_native": apix,
                            "max_diameter": max(by_number[n].diameter_ang for n in have)}

    if tracking_csv is not None and Path(tracking_csv).is_file():
        df = pd.read_csv(tracking_csv)
    else:
        tracking_csv = dataset_dir / "filament_tracking_multiclass.csv"
        df = pd.DataFrame(columns=["tomogram_path", "label_path", "particle_diameter_angst", "num_particles",
                                   "coords_path", "apix_native", "class_annotated"])
    df = df.copy()
    if "filaments_path" not in df.columns:
        df["filaments_path"] = ""
    df["filaments_path"] = df["filaments_path"].astype(object)
    df["class_annotated"] = df["class_annotated"].astype(object) if "class_annotated" in df.columns else ""
    seen = set()
    for i, r in df.iterrows():
        stem = Path(str(r["tomogram_path"])).stem
        if stem not in rows_extra:
            continue
        seen.add(stem)
        e = rows_extra[stem]
        df.at[i, "filaments_path"] = e["filaments_path"]
        ann = str(r.get("class_annotated", "")) if not pd.isna(r.get("class_annotated", "")) else ""
        nums = [t for t in ann.replace(";", ",").split(",") if t.strip()]
        nums += [str(n) for n in e["filament_classes"] if str(n) not in nums]
        df.at[i, "class_annotated"] = ",".join(nums)
    new_rows = []
    for stem, e in rows_extra.items():
        if stem in seen:
            continue
        coords_path = write_coords_csv(dataset_dir / "coords" / f"{stem}.csv", np.zeros((0, 3)), [], [])
        new_rows.append({"tomogram_path": e["tomogram_path"], "label_path": e["tomogram_path"],
                         "particle_diameter_angst": e["max_diameter"], "num_particles": 0,
                         "coords_path": str(coords_path), "apix_native": e["apix_native"],
                         "class_annotated": ",".join(str(n) for n in e["filament_classes"]),
                         "filaments_path": e["filaments_path"]})
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(tracking_csv, index=False)
    return Path(tracking_csv)
