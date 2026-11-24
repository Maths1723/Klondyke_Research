"""Spatial cross-validation.

Random hold-out leaks information: holes on the same YCGC drill line are tens
of metres apart and strongly correlated, so a random test hole almost always
has a near-twin in training. Two honest schemes:

* leave-one-line-out (LOLO): hold out an entire cross-valley drill line.
  Tests "fill the gap between drilled lines" - the interpolation question.
* along-valley blocks: hold out a contiguous stretch of valley (e.g. 1 km)
  plus a buffer either side. Tests "predict ground further from data" -
  the extrapolation question an operator actually cares about.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def leave_one_line_out(line_key: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    keys = np.asarray(line_key)
    for k in np.unique(keys):
        test = np.where(keys == k)[0]
        train = np.where(keys != k)[0]
        yield train, test


def grouped_line_folds(line_key: np.ndarray, n_folds: int = 10, seed: int = 0):
    """Faster LOLO variant: lines assigned to folds in along-valley order,
    interleaved (line i -> fold i mod K) so each fold samples the whole reach."""
    keys = np.asarray(line_key)
    uniq = np.unique(keys)
    rng = np.random.default_rng(seed)
    offset = rng.integers(n_folds)
    fold_of = {k: (i + offset) % n_folds for i, k in enumerate(uniq)}
    f = np.array([fold_of[k] for k in keys])
    for i in range(n_folds):
        yield np.where(f != i)[0], np.where(f == i)[0]


def along_valley_blocks(s: np.ndarray, block: float = 1000.0, buffer: float = 150.0):
    """Contiguous blocks along s; training excludes holes within `buffer` of the block."""
    s = np.asarray(s, dtype=float)
    lo, hi = s.min(), s.max()
    edges = np.arange(lo, hi + block, block)
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.where((s >= a) & (s < b))[0]
        if len(test) == 0:
            continue
        train = np.where((s < a - buffer) | (s >= b + buffer))[0]
        yield train, test


def run_cv(model_factory, X, y, splits, **fit_kwargs):
    """Fit a fresh model per fold. Returns out-of-fold mean, sd and fold id.

    model_factory() -> object with fit(X, y) and predict(X) -> (mean, sd).
    sd may be None for deterministic models.
    """
    n = len(y)
    mu = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    fold = np.full(n, -1)
    for i, (tr, te) in enumerate(splits):
        m = model_factory()
        m.fit(X[tr], y[tr], **fit_kwargs)
        pm, ps = m.predict(X[te])
        mu[te] = pm
        if ps is not None:
            sd[te] = ps
        fold[te] = i
    return mu, sd, fold
