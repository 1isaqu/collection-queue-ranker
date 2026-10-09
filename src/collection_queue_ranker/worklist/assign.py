"""Share a ranked queue between collectors."""

from __future__ import annotations

import numpy as np


def assign_round_robin(n_rows: int, collectors: int) -> tuple[np.ndarray, np.ndarray]:
    """Collector number (1-based) and position inside that collector's list, row by row.

    Rows are dealt out like cards, not cut into contiguous blocks. The advantage of a model over
    the balance rule is concentrated at the top of the queue, and it only materialises if the queue
    is worked in order. With blocks, collector 1 would get ranks 1..k (all of the value) and the last
    collector the tail (close to no prioritisation at all). Dealing gives every collector a list
    that is already ordered and representative of the whole ranking, so if everyone works from the
    top down, the set actually worked approximates the global top.
    """
    if collectors < 1:
        raise ValueError("collectors must be at least 1")
    index = np.arange(n_rows)
    return index % collectors + 1, index // collectors + 1
