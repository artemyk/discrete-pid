"""Sparse inverse-transform martingale coupling of finite posterior laws.

This is the discrete construction of Jourdain and Margheriti (2020),
https://doi.org/10.1214/20-EJP543, Section 3. Starting from equal-quantile
matching, pair positive and negative moment discrepancies in quantile order.
Each exchange preserves both mass marginals and corrects the output means.
The sorted-input scans use O(m+q) arithmetic and memory; no optimizer is used.
"""

import numpy as np


def _couple(theta, mass, support, weights):
    """Return nonnegative COO contributions c[x,j] for sorted posterior laws.

    In exact arithmetic, the output law must be below the source in convex
    order. Floating-point discrepancies can leave unmatched moment or slightly
    overdraw a cell; cap exchanges at its remaining mass and leave unmatched
    mass in place. The caller must check the resulting joint-distribution
    residual, which also detects genuinely infeasible input laws.
    """
    m, q = len(mass), len(weights)
    capacity = m + q
    source = np.empty(capacity, dtype=np.int64)
    output = np.empty(capacity, dtype=np.int64)
    remaining = np.empty(capacity, dtype=mass.dtype)
    discrepancy = np.empty(capacity, dtype=mass.dtype)
    plus = np.empty(capacity, dtype=np.int64)
    minus = np.empty(capacity, dtype=np.int64)
    n, nplus, nminus = 0, 0, 0
    x, j = 0, 0
    source_left, output_left = mass[0], weights[0]
    # Residual masses avoid subtracting nearly equal cumulative probabilities,
    # which would erase tiny source states near the right end of the CDF.
    while x < m:
        take = min(source_left, output_left) if j < q - 1 else source_left
        if take > 0:
            source[n], output[n], remaining[n] = x, j, take
            delta = take * (support[j] - theta[x])
            discrepancy[n] = abs(delta)
            if delta > 0:
                plus[nplus] = n
                nplus += 1
            elif delta < 0:
                minus[nminus] = n
                nminus += 1
            n += 1
        source_left -= take
        output_left -= take
        if source_left <= 0:
            x += 1
            if x < m:
                source_left = mass[x]
        if output_left <= 0 and j < q - 1:
            j += 1
            output_left = weights[j]

    rows = np.empty(3 * capacity, dtype=np.int64)
    cols = np.empty(3 * capacity, dtype=np.int64)
    data = np.empty(3 * capacity, dtype=mass.dtype)
    count, a, b = 0, 0, 0
    while a < nplus and b < nminus:
        u, v = plus[a], minus[b]
        moment = min(discrepancy[u], discrepancy[v])
        gap = theta[source[v]] - theta[source[u]]
        if gap > 0:
            transfer = min(moment / gap, remaining[u], remaining[v])
            if transfer > 0:
                remaining[u] -= transfer
                remaining[v] -= transfer
                rows[count], cols[count], data[count] = source[v], output[u], transfer
                rows[count + 1], cols[count + 1], data[count + 1] = source[u], output[v], transfer
                count += 2
        discrepancy[u] -= moment
        discrepancy[v] -= moment
        if discrepancy[u] <= 0:
            a += 1
        if discrepancy[v] <= 0:
            b += 1
    for cell in range(n):
        if remaining[cell] > 0:
            rows[count], cols[count], data[count] = source[cell], output[cell], remaining[cell]
            count += 1
    return rows[:count], cols[:count], data[:count]


def _rows(rows, cols, data, nrows):
    """Group COO contributions into CSR rows in linear time.

    Column indices need not be sorted; duplicate positive contributions are
    valid CSR entries. Avoid sorting or a dense intermediate matrix.
    """
    indptr = np.zeros(nrows + 1, dtype=np.int64)
    for row in rows:
        indptr[row + 1] += 1
    for row in range(nrows):
        indptr[row + 1] += indptr[row]
    cursor = indptr[:-1].copy()
    indices = np.empty(len(data), dtype=np.int64)
    values = np.empty(len(data), dtype=data.dtype)
    for i in range(len(data)):
        position = cursor[rows[i]]
        indices[position], values[position] = cols[i], data[i]
        cursor[rows[i]] += 1
    return values, indices, indptr


try:
    from numba import float64, int64, njit
except ImportError:
    _compiled_couple = _compiled_rows = None
else:
    # Compile or load cached code at import. Wider longdouble keeps the Python
    # coupling scan; only final float64 kernel entries use the CSR row scan.
    _compiled_couple = njit(cache=True)(_couple)
    _compiled_couple.compile((float64[::1],) * 4)
    _compiled_rows = njit(cache=True)(_rows)
    _compiled_rows.compile((int64[::1], int64[::1], float64[::1], int64))


def inverse_transform(theta, mass, support, weights):
    if _compiled_couple is not None and mass.dtype == np.dtype(np.float64):
        return _compiled_couple(theta, mass, support, weights)
    return _couple(theta, mass, support, weights)


def sparse_rows(rows, cols, data, nrows):
    if _compiled_rows is not None and data.dtype == np.dtype(np.float64):
        return _compiled_rows(rows, cols, data, nrows)
    return _rows(rows, cols, data, nrows)
