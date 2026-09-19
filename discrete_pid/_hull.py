"""Optional compilation of the lower-hull scan, without reduced precision."""

import numpy as np


def _scan(x, y):
    hull = []
    for index in range(len(x)):
        point = (x[index], y[index], index)
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = ((b[0] - a[0]) * (point[1] - b[1])
                     - (b[1] - a[1]) * (point[0] - b[0]))
            if cross > 0:
                break
            hull.pop()
        hull.append(point)
    indices = np.empty(len(hull), dtype=np.int64)
    for index, point in enumerate(hull):
        indices[index] = point[2]
    return indices


# Numba does not support extended-precision NumPy floats. Never downcast the
# hull to obtain a speedup: platforms with wider longdouble use the same scan
# in Python. No fastmath or parallel reductions are used.
try:
    from numba import njit
except ImportError:
    _compiled_scan = None
else:
    _compiled_scan = njit(cache=True)(_scan)


def lower_hull(x, y):
    if _compiled_scan is not None and x.dtype == np.dtype(np.float64):
        return _compiled_scan(x, y)
    return _scan(x, y)
