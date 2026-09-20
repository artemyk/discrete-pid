"""Exact binary-channel geometry: uint32 weights, uint64 products."""

import numpy as np


def _scan(counts, support, return_garblings):
    k, d, _ = counts.shape
    kernel_count = k if return_garblings else 0
    anchor = 0
    while not support[anchor]:
        anchor += 1
    pivot = anchor
    for y in range(d):
        if support[y]:
            if counts[0, y, 1] < counts[0, anchor, 1]:
                anchor = y
            if counts[0, y, 1] > counts[0, pivot, 1]:
                pivot = y
    reference_span = np.uint64(counts[0, pivot, 1]) - np.uint64(counts[0, anchor, 1])
    if reference_span == 0:
        return np.ones((d, 1)), np.ones((kernel_count, 2, 1))

    # After relabeling, each channel is (a_i + h_i*z)/(a_i + b_i + h_i),
    # where z ranges from zero at anchor to one at pivot. The meet uses
    # alpha=max(a_i/h_i), beta=max(b_i/h_i). All three integers are <= 2**32-1.
    parameters = np.empty((k, 3), dtype=np.uint64)
    increasing = np.empty(k, dtype=np.int64)
    left = right = 0
    for i in range(k):
        span = np.int64(counts[i, pivot, 1]) - np.int64(counts[i, anchor, 1])
        if span == 0:
            return np.ones((d, 1)), np.ones((kernel_count, 2, 1))
        x = 1 if span > 0 else 0
        h = np.uint64(abs(span))
        for y in range(d):
            if support[y]:
                difference = np.int64(counts[i, y, x]) - np.int64(counts[i, anchor, x])
                if difference < 0:
                    return np.ones((d, 1)), np.ones((kernel_count, 2, 1))
                reference = np.uint64(counts[0, y, 1]) - np.uint64(counts[0, anchor, 1])
                # Compare products directly; their difference need not fit int64.
                if np.uint64(difference) * reference_span != reference * h:
                    return np.ones((d, 1)), np.ones((kernel_count, 2, 1))
        a = np.uint64(counts[i, anchor, x])
        b = np.uint64(counts[i, pivot, 1 - x])
        parameters[i, 0], parameters[i, 1], parameters[i, 2] = a, b, h
        increasing[i] = x
        if a * parameters[left, 2] > parameters[left, 0] * h:
            left = i
        if b * parameters[right, 2] > parameters[right, 1] * h:
            right = i

    a, ha = parameters[left, 0], parameters[left, 2]
    b, hb = parameters[right, 1], parameters[right, 2]
    alpha, beta = float(a) / float(ha), float(b) / float(hb)
    normalizer = 1.0 + alpha + beta
    meet = np.empty((d, 2))
    for y in range(d):
        z = 0.0
        if support[y]:
            z = float(np.uint64(counts[0, y, 1]) - np.uint64(counts[0, anchor, 1])) / float(reference_span)
        meet[y, 0] = (beta + (1.0 - z)) / normalizer
        meet[y, 1] = (alpha + z) / normalizer

    kernels = np.empty((kernel_count, 2, 2))
    for i in range(kernel_count):
        ai, bi, h = parameters[i, 0], parameters[i, 1], parameters[i, 2]
        x = increasing[i]
        # Exact nonnegative differences preserve tiny garbling probabilities.
        # Every product and difference magnitude is <= (2**32-1)**2.
        flip0 = (float(a * h - ai * ha) / float(ha * h)) / normalizer
        flip1 = (float(b * h - bi * hb) / float(hb * h)) / normalizer
        kernels[i, 1 - x, 1] = flip0
        kernels[i, 1 - x, 0] = 1.0 - flip0
        kernels[i, x, 0] = flip1
        kernels[i, x, 1] = 1.0 - flip1
    return meet, kernels


try:
    from numba import njit
except ImportError:
    _compiled_scan = None
else:
    # Compile/load at import, matching the hull scan. No fastmath is used.
    _compiled_scan = njit(
        "Tuple((float64[:,::1], float64[:,:,::1]))(uint32[:,:,::1], boolean[::1], boolean)",
        cache=True,
    )(_scan)


def geometry(counts, support, return_garblings=False):
    return (_compiled_scan or _scan)(counts, support, return_garblings)
