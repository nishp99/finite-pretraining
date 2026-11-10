"""Strict, portable JSON; infinity is represented by a string."""

import json
from pathlib import Path
import numpy as np


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [portable(v) for v in value]
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(portable(value), indent=2, allow_nan=False) + "\n")
