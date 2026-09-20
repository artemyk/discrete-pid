"""Shared channel validation and normalization."""

from numbers import Integral, Real
import warnings

import numpy as np


def integer_weights(raw):
    """Validate channel dtype/range and report whether all weights are integers."""
    integer = raw.dtype.kind in 'iu'
    if raw.dtype.kind == 'O':
        if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) for v in raw.flat):
            raise ValueError("channels must contain real integer or floating-point weights")
        integer = all(isinstance(v, Integral) for v in raw.flat)
    elif raw.dtype.kind not in 'iuf':
        raise ValueError("channels must contain real integer or floating-point weights")
    # uint32 already enforces both bounds, so no comparison arrays are needed.
    if integer and raw.dtype != np.dtype(np.uint32):
        if np.any(raw < 0) or np.any(raw > np.iinfo(np.uint32).max):
            raise ValueError("integer channel weights must be in [0, 2**32-1] (uint32 range)")
    return integer


def normalize_rows(raw, support, *, dtype=float):
    """Copy channel weights and normalize rows, checking active rows have mass."""
    try:
        weights = np.array(raw, dtype=dtype, copy=True)
    except (OverflowError, ValueError) as error:
        raise ValueError("channel weights must be finite and nonnegative") from error
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("channel weights must be finite and nonnegative")
    scales = weights.max(axis=2)
    if np.any(scales[:, support] == 0):
        raise ValueError("each channel row on the positive-prior support needs positive mass")
    # Scale before summation so very large finite weights cannot overflow.
    weights /= np.where(scales > 0, scales, 1)[:, :, None]
    totals = weights.sum(axis=2)
    weights /= np.where(totals > 0, totals, 1)[:, :, None]
    return weights


def _same_row_sums(counts, support):
    # Callers validate shape/range and guarantee nonempty prior support.
    # uint32 entries are widened before addition; inputs are never modified.
    anchor = 0
    while not support[anchor]:
        anchor += 1
    for i in range(counts.shape[0]):
        reference = np.uint64(0)
        for x in range(counts.shape[2]):
            reference += np.uint64(counts[i, anchor, x])
        if reference == 0:
            return False
        for y in range(counts.shape[1]):
            if support[y] and y != anchor:
                total = np.uint64(0)
                for x in range(counts.shape[2]):
                    total += np.uint64(counts[i, y, x])
                if total != reference:
                    return False
    return True


try:
    from numba import njit, types
except ImportError:
    _compiled_same_row_sums = None
    warnings.warn(
        "Numba is unavailable; discrete-pid is using slower Python loops. "
        "Install discrete-pid[speed] for faster computation.",
        RuntimeWarning,
        stacklevel=2,
    )
else:
    # Eager compilation, including support for read-only and strided inputs.
    _compiled_same_row_sums = njit(
        types.boolean(types.Array(types.uint32, 3, 'A', readonly=True),
                      types.Array(types.boolean, 1, 'A', readonly=True)),
        cache=True,
    )(_same_row_sums)


def validate_row_sums(counts, support):
    """Require a common positive sum on each source's active target rows.

    Counts must already be range-checked. uint32 arrays need no conversion;
    the target solver also calls this on bounded batches of other integer types.
    """
    counts = np.asarray(counts, dtype=np.uint32)
    if not (_compiled_same_row_sums or _same_row_sums)(counts, support):
        raise ValueError(
            "integer channels need the same positive row sum within each source "
            "on the positive-prior support; supply conditional weights, not joint counts"
        )
