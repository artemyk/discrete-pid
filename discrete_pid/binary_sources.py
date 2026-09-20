"""Blackwell redundancy from a target prior and binary conditional channels."""

from __future__ import annotations

from numbers import Real
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from . import _source_scan
from ._validation import integer_weights, normalize_rows, validate_row_sums
from ._common import RedundancyResult, make_result, normalize_prior


def redundancy_binary_sources(
    prior: ArrayLike,
    channels: Iterable[ArrayLike],
    *,
    tolerance: float | None = None,
    return_channel: bool = False,
    return_garblings: bool = False,
) -> RedundancyResult:
    """Compute the binary-source meet in O(k*d) operations.

    ``prior`` contains nonnegative weights for P(Y), normalized internally.
    It may be floating point. Each channel has shape (d, 2), with rows for
    Y and columns for X_i=0,1. A packed (k,d,2) array is also accepted.

    By default, channel entries must be integers in [0, 2**32-1]. Every row
    on the positive-prior support must have the SAME positive sum L_i within
    a channel: P(X_i=x|Y=y) = channels[i][y,x] / L_i. Different sources may
    use different L_i. NumPy uint32 arrays are recommended; fitting integer
    arrays/lists are accepted. Arbitrary joint counts or unequal row totals
    do not satisfy this contract. Do not round probabilities into integers.

    Anchored channel differences give exact collinearity and endpoint
    decisions using uint64 products, without GCDs or arbitrary-size integers.
    Returned probabilities and information use floating point. Optional Numba
    compiles the integer scan at import; the same algorithm works without it.

    Set return_channel=True to return P(Q|Y) in result.channel, together with
    posteriors and posterior weights. Independently, return_garblings=True
    constructs P(Q|X_i). Both flags default to False; omitted outputs are None.
    Garbling construction and residual evaluation are skipped unless requested.

    Floating-point channels require a finite nonnegative ``tolerance``.
    Their rows may contain unnormalized weights. Collinearity is compared on
    anchored differences scaled to maximum absolute coordinate one, allowing
    sign reversal. The tolerance does NOT bound the error in redundancy;
    even tolerance=0 is approximate arithmetic. An explicit tolerance never
    weakens checks when all channels are integer arrays.

    All inputs must be finite and nonnegative. Rows with zero prior are
    ignored, and may be all zero. Inputs are not modified. Binary sources
    always have two columns; represent a constant source with a zero column.
    """
    if tolerance is not None and (
        not isinstance(tolerance, Real) or isinstance(tolerance, (bool, np.bool_))
        or not np.isfinite(tolerance) or tolerance < 0
    ):
        raise ValueError("tolerance must be None or a finite nonnegative number")
    p = normalize_prior(prior)
    support = p > 0
    raw = np.asarray(channels if isinstance(channels, np.ndarray) else list(channels))
    if raw.ndim != 3 or raw.shape[0] == 0 or raw.shape[1:] != (len(p), 2):
        raise ValueError("channels must have shape (number of sources, target states, 2)")
    if integer_weights(raw):
        counts = np.array(raw, dtype=np.uint32, order='C', copy=True)
        validate_row_sums(counts, support)
        probabilities = None
        if return_garblings:
            anchor = np.flatnonzero(support)[0]
            denominator = counts[:, anchor, 0].astype(np.uint64) + counts[:, anchor, 1]
            probabilities = counts / denominator[:, None, None]
        meet, kernels = _source_scan.geometry(counts, support, return_garblings)
    else:
        if tolerance is None:
            raise ValueError(
                "Floating-point channels require an explicit numerical tolerance "
                "(e.g. tolerance=1e-12): collinearity is sensitive to rounding. "
                "Supply uint32 conditional weights with constant row sums for exact checks."
            )
        probabilities = normalize_rows(raw, support)
        meet, kernels = _floating_geometry(probabilities, support, tolerance, return_garblings)

    # No marginal reconciliation is needed: every channel uses the same prior.
    tables = list(probabilities * p[None, :, None]) if return_garblings else None
    joint = p[:, None] * meet
    masses = joint.sum(axis=0)
    if np.any(masses <= 0):
        raise ArithmeticError("a positive probability is too small for floating-point output")
    return make_result(p, joint / masses, masses, tables=tables,
                       garblings=tuple(kernels) if return_garblings else None,
                       return_channel=return_channel)


def _floating_geometry(channels, support, tolerance, return_garblings=False):
    k, d, _ = channels.shape
    kernel_count = k if return_garblings else 0
    active = np.flatnonzero(support)
    anchor = active[np.argmin(channels[0, active, 1])]
    pivot = active[np.argmax(channels[0, active, 1])]
    differences = channels[:, active, 1] - channels[:, anchor, 1, None]
    scales = np.max(np.abs(differences), axis=1)
    spans = channels[:, pivot, 1] - channels[:, anchor, 1]
    if np.any(scales == 0) or np.any(spans == 0):
        return np.ones((d, 1)), np.ones((kernel_count, 2, 1))
    orientation = np.where(spans > 0, 1, -1)
    unit = differences / scales[:, None] * orientation[:, None]
    if np.any(np.abs(unit - unit[0]) > tolerance):
        return np.ones((d, 1)), np.ones((kernel_count, 2, 1))
    increasing = (spans > 0).astype(int)
    indices = np.arange(k)
    alpha = channels[indices, anchor, increasing] / np.abs(spans)
    beta = channels[indices, pivot, 1 - increasing] / np.abs(spans)
    a, b = alpha.max(), beta.max()
    normalizer = 1 + a + b
    z = np.zeros(d)
    z[active] = unit[0]
    meet = np.column_stack((b + (1 - z), a + z)) / normalizer
    if not return_garblings:
        return meet, np.empty((0, 2, 2))
    kernels = np.empty((k, 2, 2))
    flip0, flip1 = (a - alpha) / normalizer, (b - beta) / normalizer
    kernels[indices, 1 - increasing, 1] = flip0
    kernels[indices, 1 - increasing, 0] = 1 - flip0
    kernels[indices, increasing, 0] = flip1
    kernels[indices, increasing, 1] = 1 - flip1
    return meet, kernels
