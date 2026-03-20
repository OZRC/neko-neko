# cattorch/_constants.py

from __future__ import annotations

from typing import Any, Optional

from .tensor import Tensor

# -------------------------
# Backend selection
# -------------------------
_backend_name: Optional[str] = None
try:
    import cupy as _xp  # type: ignore
    import numpy as _np

    _backend_name = "cupy"
except Exception:
    try:
        import numpy as _xp  # type: ignore

        _np = _xp  # type: ignore
        _backend_name = "numpy"
    except Exception as exc:
        raise ImportError(
            "cattorch._constants: numpy is required (cupy optional). "
            "Could not import cupy or numpy."
        ) from exc


# -------------------------
# Helpers: dtype normalization
# -------------------------
def _normalize_dtype_for_backend(dtype: Any):
    """Return a dtype object acceptable to backend.finfo / backend.array.

    Accepts:
      - canonical strings: 'float16','float32','float64'
      - aliases: 'f16','f32','f64','half','float','double'
      - numpy/cupy dtype objects or objects with .dtype (e.g., Tensor)
    """
    # string aliases
    if isinstance(dtype, str):
        s = dtype.lower()
        aliases = {
            "f16": "float16",
            "f32": "float32",
            "f64": "float64",
            "half": "float16",
            "float": "float64",
            "double": "float64",
        }
        if s in aliases:
            s = aliases[s]
        if s in ("float16", "float32", "float64"):
            try:
                return _xp.dtype(s)
            except Exception:
                return s  # some backends accept strings in finfo
        try:
            nd = _np.dtype(s)
            try:
                return _xp.dtype(nd)
            except Exception:
                return nd
        except Exception:
            raise ValueError(f"Unsupported dtype string: {dtype!r}")

    # object with .dtype (e.g., Tensor)
    if hasattr(dtype, "dtype"):
        return _normalize_dtype_for_backend(getattr(dtype, "dtype"))

    # numpy dtype or dtype-like
    try:
        nd = _np.dtype(dtype)
        try:
            return _xp.dtype(nd)
        except Exception:
            return nd
    except Exception:
        pass

    # final attempt: let backend parse
    try:
        return _xp.dtype(dtype)
    except Exception as exc:
        raise ValueError(f"Could not interpret dtype: {dtype!r}") from exc


# -------------------------
# Tensor scalar constructor (robust)
# -------------------------
def _make_tensor_scalar(value: Any, dtype: Any = None):
    """Construct a cattorch.Tensor scalar from a Python/cupy/numpy value.

    Tries multiple strategies and preserves dtype if possible.
    """
    # Try simplest constructor forms commonly used in lightweight frameworks.
    # 1) Tensor(value, requires_grad=False)
    try:
        return Tensor(value, requires_grad=False)
    except Exception:
        pass

    # 2) Tensor(value, dtype=..., requires_grad=False)
    if dtype is not None:
        try:
            return Tensor(value, dtype=dtype, requires_grad=False)
        except Exception:
            pass

    # 3) Build backend array then pass it to Tensor
    try:
        bdt = None
        try:
            bdt = _normalize_dtype_for_backend(dtype) if dtype is not None else None
        except Exception:
            bdt = None
        if bdt is not None:
            arr = _xp.array(value, dtype=bdt)
        else:
            arr = _xp.array(value)
        # Many Tensor constructors accept array-like inputs
        try:
            return Tensor(arr, requires_grad=False)
        except Exception:
            # If that fails, try passing Python scalar
            return Tensor(arr.item(), requires_grad=False)
    except Exception:
        pass

    # 4) Fallback: convert to Python float then make Tensor
    try:
        return Tensor(float(value), requires_grad=False)
    except Exception as exc:
        raise RuntimeError("Failed to create Tensor scalar for constant.") from exc


# -------------------------
# Tensor-typed constants (take from backend, then wrap)
# -------------------------
# Use backend-provided names (no hardcoding)
# backend modules expose pi,e,inf,nan
try:
    pi = _make_tensor_scalar(_xp.pi)
except Exception:
    pi = _make_tensor_scalar(float(_xp.pi))

try:
    e = _make_tensor_scalar(_xp.e)
except Exception:
    e = _make_tensor_scalar(float(_xp.e))

try:
    inf = _make_tensor_scalar(_xp.inf)
except Exception:
    inf = _make_tensor_scalar(float(_xp.inf))

try:
    nan = _make_tensor_scalar(_xp.nan)
except Exception:
    nan = _make_tensor_scalar(float(_xp.nan))


# -------------------------
# Tensor-typed finfo
# -------------------------
class _TensorFinfo:
    __slots__ = ("eps", "tiny", "max", "min", "bits", "dtype")

    def __init__(self, dtype: Any):
        # get backend finfo
        bdt = _normalize_dtype_for_backend(dtype)
        b_finfo = _xp.finfo(bdt)
        self.dtype = dtype
        # bits -> make as integer Tensor if supported, else Python int
        try:
            self.bits = _make_tensor_scalar(int(getattr(b_finfo, "bits", 0)))
        except Exception:
            self.bits = int(getattr(b_finfo, "bits", 0))

        # numeric fields as Tensor scalars (preserve dtype where possible)
        # convert backend values to Python floats to ensure compatibility
        self.eps = _make_tensor_scalar(float(b_finfo.eps), dtype=bdt)
        self.tiny = _make_tensor_scalar(float(b_finfo.tiny), dtype=bdt)
        self.max = _make_tensor_scalar(float(b_finfo.max), dtype=bdt)
        self.min = _make_tensor_scalar(float(b_finfo.min), dtype=bdt)

    def __repr__(self):
        def _val_to_str(v):
            # If it's a Tensor-like object with .item(), prefer the Python scalar
            try:
                if hasattr(v, "item") and callable(getattr(v, "item")):
                    try:
                        return repr(v.item())
                    except Exception:
                        # Fall through to repr(v) if .item() fails
                        return repr(v)
                # Try numpy/cupy scalar conversion if available
                try:
                    import numpy as _np  # safe to import; will be present in your environment

                    if isinstance(v, (_np.floating, _np.integer)):
                        # show float for floating types, int for integer types
                        return (
                            repr(float(v))
                            if isinstance(v, _np.floating)
                            else repr(int(v))
                        )
                except Exception:
                    pass
            except Exception:
                pass

            # Fallback: use the object's repr
            try:
                return repr(v)
            except Exception:
                return "<value>"

        return (
            f"_TensorFinfo(dtype={self.dtype!r}, "
            f"bits={_val_to_str(self.bits)}, "
            f"eps={_val_to_str(self.eps)}, "
            f"tiny={_val_to_str(self.tiny)}, "
            f"max={_val_to_str(self.max)}, "
            f"min={_val_to_str(self.min)})"
        )


def finfo(dtype: Any) -> _TensorFinfo:
    """Return finfo-like object whose numeric attributes are cattorch.Tensor scalars.

    dtype: string, numpy/cupy dtype, or object with .dtype.
    """
    return _TensorFinfo(dtype)


def tensor_eps_of(dtype: Any):
    """Return dtype-machine-epsilon as a cattorch.Tensor scalar."""
    return finfo(dtype).eps


# -------------------------
# Convenience: backend name
# -------------------------
backend = _backend_name

# -------------------------
# Public API
# -------------------------
__all__ = [
    "pi",
    "e",
    "inf",
    "nan",
    "finfo",
    "tensor_eps_of",
    "backend",
]
