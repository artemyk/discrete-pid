"""Blackwell union information and source decoding, with no posterior grid."""
import numpy as np
from discrete_pid import union_binary_target, union_binary_sources

bsc = np.array([[9, 1], [1, 9]], dtype=np.uint32)
bec = np.array([[5, 0, 5], [0, 5, 5]], dtype=np.uint32)
joined = union_binary_target([1, 1], [bsc, bec],
                             return_channel=True, return_garblings=True)
assert abs(joined.union_bits - 0.6390359525563188) < 1e-10
for raw, decoder in zip([bsc, bec], joined.garblings):
    np.testing.assert_allclose(joined.channel @ decoder,
                               raw / raw.sum(axis=1, keepdims=True), atol=1e-10)
print(f"Binary-target union: {joined.union_bits:.9f} bits")

# These binary channels with a three-state target need not have a join.
ones = np.array([[.5, .5], [.5, 0], [0, .5]])
channels = np.stack((1 - ones.T, ones.T), axis=2)
result = union_binary_sources([1, 1, 1], channels, tolerance=1e-7,
                               return_channel=True, return_garblings=True)
for raw, decoder in zip(channels, result.garblings):
    np.testing.assert_allclose(result.channel @ decoder, raw, atol=1e-7)
print(f"Binary-source union: {result.union_bits:.9f} bits; "
      f"numerical gap {result.gap_nats:.3g} nats")
