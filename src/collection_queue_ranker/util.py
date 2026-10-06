"""Small helpers shared by several modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def text_id(value: Any) -> str:
    """An id as text: no `12.0`, no scientific notation, empty string for missing.

    Ids leave the package as text so a spreadsheet cannot turn a long number into 1.2E+10 or drop
    leading zeros, and so ids read back from a CSV compare equal to ids read from a database.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def debtor_key(frame: pd.DataFrame) -> pd.Series:
    """One string per (product_line, debtor_id), stable across int / float / text ids."""
    return frame["product_line"].astype(str) + "|" + frame["debtor_id"].map(text_id)
