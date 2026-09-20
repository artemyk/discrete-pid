"""Blackwell redundancy from a target prior and binary conditional channels."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from . import _source_scan
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
    integer = raw.dtype.kind in 'iu'
    if raw.dtype.kind == 'O':
        if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) for v in raw.flat):
            raise ValueError("channels must contain real integer or floating-point weights")
        integer = all(isinstance(v, Integral) for v in raw.flat)
    elif raw.dtype.kind not in 'iuf':
        raise ValueError("channels must contain real integer or floating-point weights")

    if integer:
        if np.any(raw < 0) or np.any(raw > np.iinfo(np.uint32).max):
            raise ValueError("integer channel weights must be in [0, 2**32-1] (uint32 range)")
        counts = np.array(raw, dtype=np.uint32, order='C', copy=True)
        totals = counts[:, :, 0].astype(np.uint64) + counts[:, :, 1]
        denominator = totals[:, np.flatnonzero(support)[0]]
        if np.any(denominator == 0) or np.any(totals[:, support] != denominator[:, None]):
            raise ValueError(
                "integer channels need the same positive row sum within each source "
                "on the positive-prior support; supply conditional weights, not joint counts"
            )
        probabilities = counts / denominator[:, None, None] if return_garblings else None
        meet, kernels = _source_scan.geometry(counts, support, return_garblings)
    else:
        if tolerance is None:
            raise ValueError(
                "Floating-point channels require an explicit numerical tolerance "
                "(e.g. tolerance=1e-12): collinearity is sensitive to rounding. "
                "Supply uint32 conditional weights with constant row sums for exact checks."
            )
        try:
            weights = np.asarray(raw, dtype=float)
        except (OverflowError, ValueError) as error:
            raise ValueError("channel weights must be finite and nonnegative") from error
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("channel weights must be finite and nonnegative")
        scales = weights.max(axis=2)
        if np.any(scales[:, support] == 0):
            raise ValueError("each channel row on the positive-prior support needs positive mass")
        # Scaling before summation permits even very large finite row weights.
        scaled = weights / np.where(scales > 0, scales, 1)[:, :, None]
        totals = scaled.sum(axis=2)
        probabilities = scaled / np.where(totals > 0, totals, 1)[:, :, None]
        meet, kernels = _floating_geometry(probabilities, support, tolerance, return_garblings)

    # No marginal reconciliation is needed: every channel uses the same prior.
    tables = list(probabilities * p[None, :, None]) if return_garblings else None
    joint = p[:, None] * meet
    masses = joint.sum(axis=0)
    if np.any(masses <= 0):
        raise ArithmeticError("a positive probability is too small for floating-point output")
    return make_result(tables, joint / masses, masses, 0.0,
                       tuple(kernels) if return_garblings else None,
                       return_channel=return_channel, prior=p)


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
