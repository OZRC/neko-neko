# cattorch/backend/utils.py
from __future__ import annotations

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

import numpy as np

from . import _array as ap
from . import dtypes


def detect_xp_from_obj(obj):
    # try to detect xp (numpy or cupy)

    if hasattr(obj, "data"):
        if isinstance(obj.data, ap.Array):
            return obj.data.xp
    if isinstance(obj, ap.Array):
        return obj.xp
    try:
        a = getattr(obj, "_array", None)
        if a is not None:
            # cupy array has module 'cupy'
            mod = a.__class__.__module__
            if "cupy" in mod:
                return cp or np
    except Exception:
        pass
    return np


def get_xp_for(*objs):
    """Return the backend module (np or cupy) that matches the provided objects.
    Raises if more than one backend is present simultaneously.
    """
    xps = set()
    for o in objs:
        xps.add(detect_xp_from_obj(o))
    if len(xps) == 0:
        return np
    if len(xps) > 1:
        raise RuntimeError(
            "Mixed backends (numpy and cupy) in einsum operands not supported"
        )
    return xps.pop()


def _ensure_array(raw, like: ap.Array):
    if raw is None:
        return None
    if isinstance(raw, ap.Array):
        return raw
    xp = getattr(like, "xp", None) or __import__("numpy")
    try:
        raw_arr = xp.asarray(raw)
    except Exception:
        import numpy as _np

        raw_arr = _np.asarray(raw)
    return ap.Array(raw_arr, device=getattr(like, "device", None))


def to_ap(x):
    if hasattr(x, "data") and isinstance(x.data, ap.Array):
        return x.data
    if isinstance(x, ap.Array):
        return x
    return ap.Array(x)


def to_raw(x):
    """Return raw ndarray (numpy/cupy) for ap.Array / Tensor / numpy/cupy array."""
    if hasattr(x, "data"):
        if isinstance(x.data, ap.Array):  # Tensor
            d = x.data
            return getattr(d, "_array", d.asnumpy())
    if isinstance(x, ap.Array):
        return getattr(x, "_array", x.asnumpy())
    return x

# ---------------------------
# expose dtypes mapping helpers
# ---------------------------
def to_numpy_dtype(dt):
    return (
        np.dtype(dtypes.to_numpy(dt)) if hasattr(dtypes, "to_numpy") else np.dtype(dt)
    )
