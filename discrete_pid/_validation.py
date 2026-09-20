"""Integer channel row-sum checks without source-by-target temporaries."""

import warnings

import numpy as np


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
