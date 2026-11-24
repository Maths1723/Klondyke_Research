"""Valley-aligned coordinates.

Placer gold is elongated along the valley and narrow across it, and the
valley curves. Distances measured in map coordinates mix the two directions,
so we re-express every location as

    s : arc length along a valley centreline, increasing downstream (m)
    n : signed perpendicular offset from the centreline (m),
        positive on the LEFT limit (left side looking downstream, the
        placer-mining convention)

The centreline can come from a mapped stream line, or - with no external
data - from the YCGC drill lines themselves: each line was laid across the
valley, so the sequence of line midpoints traces the valley axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep


def _densify(poly: np.ndarray, step: float) -> np.ndarray:
    seg = np.diff(poly, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(seglen)])
    t = np.arange(0, cum[-1] + step / 2, step)
    return np.column_stack([np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])])


def centreline_from_drill_lines(
    x: np.ndarray,
    y: np.ndarray,
    line_order: np.ndarray,
    smoothing: float | None = None,
    step: float = 5.0,
    extend: float = 300.0,
) -> np.ndarray:
    """Smooth centreline through the midpoints of cross-valley drill lines.

    line_order: a sortable key per hole giving downstream order of its line
    (e.g. claim number for claims numbered 'below discovery').
    The midpoint of a line is the midpoint of its extreme holes, not the
    centroid, so uneven hole spacing does not drag the axis sideways.
    """
    df = pd.DataFrame({"x": x, "y": y, "k": line_order})
    mids = []
    for k, g in df.groupby("k", sort=True):
        if len(g) == 1:
            mids.append((k, g.x.iloc[0], g.y.iloc[0]))
            continue
        pts = g[["x", "y"]].to_numpy()
        # extreme pair along the line's principal direction
        c = pts - pts.mean(0)
        u = np.linalg.svd(c, full_matrices=False)[2][0]
        proj = c @ u
        a, b = pts[proj.argmin()], pts[proj.argmax()]
        mids.append((k, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    m = np.array([(mx, my) for _, mx, my in mids])
    if len(m) < 4:
        poly = m
    else:
        s = smoothing if smoothing is not None else len(m) * (40.0 ** 2)
        tck, _ = splprep([m[:, 0], m[:, 1]], s=s, k=3)
        uu = np.linspace(0, 1, max(200, 20 * len(m)))
        sx, sy = splev(uu, tck)
        poly = np.column_stack([sx, sy])
    # extend both ends straight so edge holes project cleanly
    d0 = poly[0] - poly[1]
    d1 = poly[-1] - poly[-2]
    poly = np.vstack([poly[0] + d0 / np.linalg.norm(d0) * extend, poly,
                      poly[-1] + d1 / np.linalg.norm(d1) * extend])
    return _densify(poly, step)


def project_to_centreline(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (s, n) for each point. Exact projection onto the polyline segments."""
    P = np.column_stack([px, py]).astype(float)
    A = poly[:-1]
    B = poly[1:]
    AB = B - A
    L2 = (AB ** 2).sum(1)
    seglen = np.sqrt(L2)
    cum = np.concatenate([[0], np.cumsum(seglen)])[:-1]
    s_out = np.empty(len(P))
    n_out = np.empty(len(P))
    # chunk to bound memory: points x segments
    for i0 in range(0, len(P), 2000):
        Q = P[i0:i0 + 2000]
        AQ = Q[:, None, :] - A[None, :, :]
        t = np.clip((AQ * AB[None]).sum(2) / L2[None], 0, 1)
        C = A[None] + t[..., None] * AB[None]
        d2 = ((Q[:, None, :] - C) ** 2).sum(2)
        j = d2.argmin(1)
        idx = np.arange(len(Q))
        s_out[i0:i0 + 2000] = cum[j] + t[idx, j] * seglen[j]
        # sign: cross(AB, AQ) > 0 means point is to the left of the downstream direction
        cross = AB[j, 0] * AQ[idx, j, 1] - AB[j, 1] * AQ[idx, j, 0]
        n_out[i0:i0 + 2000] = np.sign(cross) * np.sqrt(d2[idx, j])
    return s_out, n_out


def centreline_from_points(x, y, bin_m: float = 200.0, iters: int = 4, smoothing_m: float = 40.0,
                           step: float = 5.0, extend: float = 300.0) -> np.ndarray:
    """Valley axis from an unlabelled cloud of holes (a simple principal curve).

    Start from the first principal axis; then repeatedly project the holes onto
    the current curve, take the median position of holes in bins along it, and
    refit a smoothing spline (Hastie & Stuetzle 1989, binned). Works when the
    valley is longer than wide and does not fold back on itself within one bin.
    """
    P = np.column_stack([x, y]).astype(float)
    c = P.mean(0)
    u = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ u
    poly = np.vstack([c + u * (t.min() - extend), c + u * (t.max() + extend)])
    poly = _densify(poly, step)
    for _ in range(iters):
        s, _ = project_to_centreline(P[:, 0], P[:, 1], poly)
        edges = np.arange(s.min(), s.max() + bin_m, bin_m)
        idx = np.digitize(s, edges)
        knots = [np.median(P[idx == k], axis=0) for k in np.unique(idx) if (idx == k).sum() >= 3]
        m = np.array(knots)
        if len(m) < 4:
            break
        tck, _ = splprep([m[:, 0], m[:, 1]], s=len(m) * smoothing_m ** 2, k=3)
        sx, sy = splev(np.linspace(0, 1, max(200, 20 * len(m))), tck)
        poly = np.column_stack([sx, sy])
        d0, d1 = poly[0] - poly[1], poly[-1] - poly[-2]
        poly = np.vstack([poly[0] + d0 / np.linalg.norm(d0) * extend, poly,
                          poly[-1] + d1 / np.linalg.norm(d1) * extend])
        poly = _densify(poly, step)
    return poly


def cluster_lines(s: np.ndarray, gap_m: float = 40.0) -> np.ndarray:
    """Group holes into cross-valley drill lines by gaps in down-valley position.

    YCGC lines are ~100-200 m apart and holes on one line share s within a few
    tens of metres, so a 1-D single-linkage split at `gap_m` recovers them.
    Returns an integer line label per hole, increasing downstream.
    """
    order = np.argsort(s)
    lab = np.empty(len(s), int)
    cur = 0
    lab[order[0]] = 0
    for a, b in zip(order[:-1], order[1:]):
        if s[b] - s[a] > gap_m:
            cur += 1
        lab[b] = cur
    return lab


def cluster_lines_graph(x, y, poly, radius: float = 60.0, max_angle_deg: float = 35.0) -> np.ndarray:
    """Drill lines as connected components of a 'cross-valley neighbour' graph.

    Two holes are linked if they are within `radius` m and the vector between
    them is within `max_angle_deg` of the local cross-valley direction.
    Holes on one line (tens of m apart, across the valley) connect; holes on
    neighbouring lines (100-200 m apart, along the valley) do not.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    P = np.column_stack([x, y]).astype(float)
    s, _ = project_to_centreline(P[:, 0], P[:, 1], poly)
    seg = np.hypot(*np.diff(poly, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    tx = np.interp(s + 2, cum, poly[:, 0]) - np.interp(s - 2, cum, poly[:, 0])
    ty = np.interp(s + 2, cum, poly[:, 1]) - np.interp(s - 2, cum, poly[:, 1])
    t = np.column_stack([tx, ty]) / (np.hypot(tx, ty)[:, None] + 1e-12)
    pairs = np.array(list(cKDTree(P).query_pairs(radius)))
    if len(pairs) == 0:
        return np.arange(len(P))
    d = P[pairs[:, 1]] - P[pairs[:, 0]]
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    tm = t[pairs[:, 0]] + t[pairs[:, 1]]
    tm /= np.linalg.norm(tm, axis=1, keepdims=True) + 1e-12
    along = np.abs((d * tm).sum(1))                      # |cos| with the valley direction
    keep = along <= np.sin(np.deg2rad(max_angle_deg))
    e = pairs[keep]
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(P), len(P)))
    _, lab = connected_components(A, directed=False)
    # relabel in downstream order
    order = pd.Series(s).groupby(lab).median().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in lab])


def reconstruct_frame(x, y, iters: int = 3, radius: float = 60.0, max_angle_deg: float = 35.0):
    """Unlabelled holes -> (centreline, line labels): axis from the cloud, lines from
    the cross-valley graph, then the axis refitted through line midpoints, repeated."""
    poly = centreline_from_points(x, y)
    lab = None
    for _ in range(iters):
        lab = cluster_lines_graph(x, y, poly, radius=radius, max_angle_deg=max_angle_deg)
        multi = pd.Series(lab).map(pd.Series(lab).value_counts()) >= 3
        if multi.sum() < 10:
            break
        poly = centreline_from_drill_lines(np.asarray(x)[multi], np.asarray(y)[multi], lab[multi])
    return poly, lab


def sn_to_xy(s: np.ndarray, n: np.ndarray, poly: np.ndarray, s_offset: float = 0.0):
    """Inverse of project_to_centreline: valley coordinates back to map x, y."""
    seg = np.hypot(*np.diff(poly, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    sq = np.asarray(s, float) + s_offset
    px, py = np.interp(sq, cum, poly[:, 0]), np.interp(sq, cum, poly[:, 1])
    tx = np.interp(sq + 2, cum, poly[:, 0]) - np.interp(sq - 2, cum, poly[:, 0])
    ty = np.interp(sq + 2, cum, poly[:, 1]) - np.interp(sq - 2, cum, poly[:, 1])
    tn = np.hypot(tx, ty) + 1e-12
    n = np.asarray(n, float)
    return px - ty / tn * n, py + tx / tn * n


def valley_halfwidth(s: np.ndarray, n: np.ndarray, line_key: np.ndarray, window: float = 500.0) -> np.ndarray:
    """Rough local half-width of the drilled valley floor at each hole.

    Taken as the median, over drill lines within +-window/2 along the valley,
    of each line's half extent. A proxy only: YCGC lines often did not span
    the whole valley.
    """
    df = pd.DataFrame({"s": s, "n": n, "k": line_key})
    per_line = df.groupby("k").agg(s=("s", "median"), hw=("n", lambda v: (v.max() - v.min()) / 2))
    out = np.empty(len(s))
    for i, si in enumerate(s):
        near = per_line[np.abs(per_line.s - si) <= window / 2]
        out[i] = np.median(near.hw) if len(near) else np.nan
    return out


def tributary_features(s: np.ndarray, n: np.ndarray, mouths: pd.DataFrame) -> pd.DataFrame:
    """Distance along the valley to the nearest tributary mouth on each limit.

    mouths: DataFrame with columns s (position of the mouth along the
    centreline) and side ('left' / 'right').
    Positive ds = the hole is downstream of the mouth.
    """
    out = {}
    for side in ("left", "right"):
        ms = mouths.loc[mouths.side == side, "s"].to_numpy()
        if len(ms) == 0:
            out[f"ds_trib_{side}"] = np.full(len(s), np.nan)
            continue
        d = s[:, None] - ms[None, :]
        j = np.abs(d).argmin(1)
        out[f"ds_trib_{side}"] = d[np.arange(len(s)), j]
    return pd.DataFrame(out)
