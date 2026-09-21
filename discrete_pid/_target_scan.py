"""Call-function knots from one global posterior order per batch."""

import numpy as np


def _count_posteriors(counts, prior):
    sources, _, states = counts.shape
    theta = np.empty(sources * states, dtype=prior.dtype)
    totals = np.empty(sources, dtype=np.uint64)
    for i in range(sources):
        total0 = np.uint64(0)
        total1 = np.uint64(0)
        for x in range(states):
            a, b = counts[i, 0, x], counts[i, 1, x]
            total0 += np.uint64(a)
            total1 += np.uint64(b)
            numerator = prior[1] * float(b)
            denominator = prior[0] * float(a) + numerator
            theta[i * states + x] = numerator / denominator if denominator > 0 else 0.
        if total0 == 0 or total0 != total1:
            raise ValueError(
                "integer channels need the same positive row sum within each source "
                "on the positive-prior support; supply conditional weights, not joint counts"
            )
        totals[i] = total0
    return theta, totals


def _scan(values, theta, order, totals, prior):
    """Scan descending, writing ascending knots with separate source tails.

    For uint32 counts, use their row totals and the target prior. For already
    normalized joint tables, use unit totals and multipliers (1,1). Dtype is
    inherited from theta so the Python path also preserves wider longdouble.
    """
    sources, _, states = values.shape
    tail0 = np.zeros(sources, dtype=theta.dtype)
    tail1 = np.zeros(sources, dtype=theta.dtype)
    locations = np.empty(len(theta) + 2, dtype=theta.dtype)
    heights = np.empty_like(locations)
    used = len(theta) + 1
    locations[used], heights[used] = 1., 0.
    used -= 1
    for position in range(len(order) - 1, -1, -1):
        index = order[position]
        i, x = index // states, index % states
        tail0[i] += values[i, 0, x]
        tail1[i] += values[i, 1, x]
        t = theta[index]
        # Posterior-one atoms contribute to the tails, even though the common
        # endpoints are inserted only once. Zero-mass states have theta=0.
        if 0. < t < 1.:
            target_tail = prior[1] * tail1[i]
            total_tail = prior[0] * tail0[i] + target_tail
            call = (target_tail - t * total_tail) / totals[i]
            locations[used], heights[used] = t, max(call, 0.)
            used -= 1
    # The caller fills in the common endpoint height P(Y=1).
    locations[used], heights[used] = 0., 0.
    return locations[used:], heights[used:]


def _source_orders(order, sources, states):
    positions = np.zeros(sources, dtype=np.int64)
    result = np.empty((sources, states), dtype=np.int64)
    for index in order:
        source = index // states
        result[source, positions[source]] = index % states
        positions[source] += 1
    return result


try:
    from numba import njit, types
except ImportError:
    _compiled_posteriors = _compiled_scan = _compiled_orders = None
else:
    # Eager compilation; arbitrary layouts and read-only input arrays work.
    # Wider longdouble uses these same loops in Python, without downcasting.
    counts_type = types.Array(types.uint32, 3, 'A', readonly=True)
    joint_type = types.Array(types.float64, 3, 'A', readonly=True)
    vector_type = types.Array(types.float64, 1, 'C', readonly=True)
    order_type = types.Array(types.int64, 1, 'C', readonly=True)
    totals_type = types.Array(types.uint64, 1, 'C', readonly=True)
    _compiled_posteriors = njit(cache=True)(_count_posteriors)
    _compiled_posteriors.compile((counts_type, vector_type))
    _compiled_scan = njit(cache=True)(_scan)
    for values_type in (counts_type, joint_type):
        _compiled_scan.compile((values_type, vector_type, order_type, totals_type, vector_type))
    _compiled_orders = njit(cache=True)(_source_orders)
    _compiled_orders.compile((order_type, types.int64, types.int64))


def count_posteriors(counts, prior):
    scan = (_compiled_posteriors if prior.dtype == np.dtype(np.float64) else None)
    return (scan or _count_posteriors)(counts, prior)


def call_knots(values, theta, order, totals, prior):
    scan = (_compiled_scan if theta.dtype == np.dtype(np.float64) else None)
    return (scan or _scan)(values, theta, order, totals, prior)


def source_orders(order, sources, states):
    return (_compiled_orders or _source_orders)(order, sources, states)
