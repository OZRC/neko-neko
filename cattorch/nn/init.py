# cattorch/nn/init.py
import numpy as np

from ..backend.utils import to_ap as _to_array
from ..tensor import Tensor

# -----------------------------
# Initializers (in-place)
# -----------------------------

def zeros_(tensor: Tensor):
    """In-place zero initializer."""
    # Use Array's in-place zeros_ (or zero_) if available
    tensor.data.zeros_()
    return tensor


def ones_(tensor: Tensor):
    """In-place ones initializer."""
    tensor.data.fill_(1)
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0):
    """In-place normal initializer."""
    # Array.normal_ should respect tensor dtype/device
    tensor.data.normal_(mean=mean, std=std)
    return tensor


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0):
    """In-place uniform initializer on [a, b)."""
    tensor.data.uniform_(from_=a, to=b)
    return tensor


def xavier_uniform_(tensor: Tensor, gain: float = 1.0):
    """
    Xavier / Glorot uniform initializer.
    For 2D weight (out, in), uses fan_in, fan_out. For conv kernels, it approximates using product of dims.
    """
    arr0 = _to_array(tensor)
    shape = arr0.shape
    if arr0.ndim < 1:
        fan_in = fan_out = 1
    elif arr0.ndim == 1:
        fan_in = fan_out = shape[0]
    elif arr0.ndim == 2:
        fan_in, fan_out = shape[1], shape[0]
    else:
        # For conv: (out_channels, in_channels, k1, k2, ...)
        fan_out = shape[0]
        fan_in = int(np.prod(shape[1:]))
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    a = np.sqrt(3.0) * std
    # use Array.uniform_ in-place
    tensor.data.uniform_(from_=-a, to=a)
    return tensor


def xavier_normal_(tensor: Tensor, gain: float = 1.0):
    arr0 = _to_array(tensor)
    shape = arr0.shape
    if arr0.ndim < 1:
        fan_in = fan_out = 1
    elif arr0.ndim == 1:
        fan_in = fan_out = shape[0]
    elif arr0.ndim == 2:
        fan_in, fan_out = shape[1], shape[0]
    else:
        fan_out = shape[0]
        fan_in = int(np.prod(shape[1:]))
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    tensor.data.normal_(mean=0.0, std=std)
    return tensor


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
):
    """
    Kaiming / He uniform initializer.
    mode: 'fan_in' or 'fan_out'
    nonlinearity: currently unused except to mimic signature (we won't compute gain from nonlinearity).
    """
    arr0 = _to_array(tensor)
    shape = arr0.shape
    if arr0.ndim < 1:
        fan_in = fan_out = 1
    elif arr0.ndim == 1:
        fan_in = fan_out = shape[0]
    elif arr0.ndim == 2:
        fan_in, fan_out = shape[1], shape[0]
    else:
        fan_out = shape[0]
        fan_in = int(np.prod(shape[1:]))
    fan = fan_in if mode == "fan_in" else fan_out
    bound = np.sqrt(3.0) * np.sqrt(2.0 / fan)
    tensor.data.uniform_(from_=-bound, to=bound)
    return tensor


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
):
    arr0 = _to_array(tensor)
    shape = arr0.shape
    if arr0.ndim < 1:
        fan_in = fan_out = 1
    elif arr0.ndim == 1:
        fan_in = fan_out = shape[0]
    elif arr0.ndim == 2:
        fan_in, fan_out = shape[1], shape[0]
    else:
        fan_out = shape[0]
        fan_in = int(np.prod(shape[1:]))
    fan = fan_in if mode == "fan_in" else fan_out
    std = np.sqrt(2.0 / fan)
    tensor.data.normal_(mean=0.0, std=std)
    return tensor