"""Supporting-line scans and an upper envelope for binary-target joins.

Compiled kernels are optional. Wider longdouble inputs retain their precision
in the same Python kernels, as in the binary-target meet implementation.
"""

import numpy as np


def _lines(values, theta, order, totals, multipliers):
    sources, _, states = values.shape
    tail0 = np.zeros(sources, dtype=theta.dtype)
    tail1 = np.zeros(sources, dtype=theta.dtype)
    previous = np.full(sources, -1., dtype=theta.dtype)
    slopes = np.empty(len(theta), dtype=theta.dtype)
    intercepts = np.empty_like(slopes)
    used = 0
    for position in range(len(order) - 1, -1, -1):
        index = order[position]
        i, x = index // states, index % states
        a, b = values[i, 0, x], values[i, 1, x]
        if a == 0 and b == 0:
            continue
        t = theta[index]
        if previous[i] >= 0 and previous[i] != t:
            intercept = multipliers[1] * tail1[i] / totals[i]
            mass = multipliers[0] * tail0[i] / totals[i] + intercept
            slopes[used], intercepts[used] = -mass, intercept
            used += 1
        tail0[i] += a
        tail1[i] += b
        previous[i] = t
    return slopes[:used], intercepts[:used]


def _envelope(slopes, intercepts):
    """Indices and activation coordinates of slope-sorted distinct lines."""
    indices = np.empty(len(slopes), dtype=np.int64)
    starts = np.empty(len(slopes), dtype=slopes.dtype)
    used = 0
    for i in range(len(slopes)):
        start = -np.inf
        while used:
            previous = indices[used - 1]
            start = ((intercepts[previous] - intercepts[i])
                     / (slopes[i] - slopes[previous]))
            if start > starts[used - 1]:
                break
            used -= 1
            start = -np.inf
        indices[used], starts[used] = i, start
        used += 1
    return indices[:used], starts[:used]


try:
    from numba import njit, types
except ImportError:
    _compiled_lines = _compiled_envelope = None
else:
    counts_type = types.Array(types.uint32, 3, 'A', readonly=True)
    joint_type = types.Array(types.float64, 3, 'A', readonly=True)
    vector_type = types.Array(types.float64, 1, 'C', readonly=True)
    order_type = types.Array(types.int64, 1, 'C', readonly=True)
    totals_type = types.Array(types.uint64, 1, 'C', readonly=True)
    _compiled_lines = njit(cache=True)(_lines)
    for values_type in (counts_type, joint_type):
        _compiled_lines.compile((values_type, vector_type, order_type,
                                 totals_type, vector_type))
    _compiled_envelope = njit(cache=True)(_envelope)
    _compiled_envelope.compile((vector_type, vector_type))


def supporting_lines(values, theta, order, totals, multipliers):
    scan = _compiled_lines if theta.dtype == np.dtype(np.float64) else None
    return (scan or _lines)(values, theta, order, totals, multipliers)


def upper_envelope(slopes, intercepts):
    scan = _compiled_envelope if slopes.dtype == np.dtype(np.float64) else None
    return (scan or _envelope)(slopes, intercepts)
