"""Fast special-case algorithms for discrete Blackwell redundancy."""

from ._common import RedundancyResult
from .binary_target import redundancy_binary_target
from .binary_sources import redundancy_binary_sources

__all__ = [
    "RedundancyResult",
    "redundancy_binary_target",
    "redundancy_binary_sources",
]
