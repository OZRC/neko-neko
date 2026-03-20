# cattorch/autograd/_helpers.py
from ..backend import _array as _ap
from ..backend.utils import to_raw

try:
    from ..tensor import Tensor as _Tensor
except Exception:
    _Tensor = None  # will import lazily in functions


# ---------- small helpers ----------
def _ensure_tensor_like(x, like_tensor=None):
    """
    Wrap raw array-like x into a Tensor (requires_grad=False). If x already a Tensor, return it.
    like_tensor: optional Tensor to copy device/dtype from.
    """
    global _Tensor
    if _Tensor is None:
        from ..tensor import Tensor as _Tensor
    if isinstance(x, _Tensor):
        return x
    # x may be ap.Array or numpy array or scalar
    try:
        if isinstance(x, _ap.Array):
            arr = x
        else:
            arr = _ap.Array(x)
    except Exception:
        # fallback: try convert via numpy
        arr = _ap.Array(x)
    device = (
        getattr(like_tensor, "device", None)
        if like_tensor is not None
        else getattr(arr, "device", None)
    )
    return _Tensor(arr, requires_grad=False, device=device)


def unbroadcast(grad, target_shape, xp):
    """cattorch
    Reduce `grad` (ndarray / array-like) so kết quả có shape == target_shape.
    Works for target_shape being scalar () as well.
    """
    grad = xp.asarray(to_raw(grad))
    # ensure tuples
    if isinstance(target_shape, int):
        target_shape = (target_shape,)
    target_shape = tuple(target_shape)

    # case scalar target
    if target_shape == () or len(target_shape) == 0:
        # sum all elements -> scalar (wrap into backend Array to keep types consistent)
        return _ap.Array(grad.sum())

    grad_ndim = grad.ndim
    target_ndim = len(target_shape)

    # 1) reduce leading dimensions if grad has more dims than target
    leading = grad_ndim - target_ndim
    axes_to_sum = list(range(leading)) if leading > 0 else []

    # 2) for each target dim that equals 1, we must sum over that axis in grad
    # note: after removing leading dims, the positions shift by `leading`.
    for i, size in enumerate(target_shape):
        if size == 1:
            axes_to_sum.append(i + max(0, leading))

    if len(axes_to_sum) == 0:
        # shapes already align; but still ensure shape equals target_shape
        # e.g., grad.shape == target_shape
        if grad.shape != target_shape:
            return grad.reshape(target_shape)
        return grad

    # Use axis tuple and keepdims=True so we can reshape directly after
    axes_tuple = tuple(sorted(set(axes_to_sum)))
    reduced = grad.sum(axis=axes_tuple, keepdims=True)

    # Now squeeze/reshape to target_shape
    # reduced might have extra leading dims of size 1 (keepdims), so reshape:
    return reduced.reshape(target_shape)


def _tensor_unbroadcast(grad_t, target_shape):
    """
    Sum-reduce grad_t (a Tensor) to match target_shape (tuple).
    Mirrors backend.utils.unbroadcast but using Tensor.sum.
    """
    global _Tensor
    if _Tensor is None:
        from ..tensor import Tensor as _Tensor
    if not isinstance(target_shape, tuple):
        target_shape = tuple(target_shape)
    g = grad_t
    g_shape = tuple(getattr(g, "shape", ()))
    # quick path
    if g_shape == target_shape:
        return g.reshape(target_shape)
    # axes to sum: where target dim ==1 but grad dim !=1, plus extra leading dims
    axes = []
    for i, (gd, td) in enumerate(zip(g_shape[::-1], target_shape[::-1])):
        if td == 1 and gd != 1:
            axes.append(len(g_shape) - 1 - i)
    if len(g_shape) > len(target_shape):
        extra_axes = list(range(len(g_shape) - len(target_shape)))
        axes = extra_axes + axes
    if not axes:
        return g.reshape(target_shape)
    # sum along axes, then reshape
    axes_tuple = tuple(sorted(set(axes)))
    # Tensor.sum supports keepdims; ensure we keep dims then reshape
    g = g.sum(dim=axes_tuple, keepdims=True)
    return g.reshape(target_shape)


# Helper: broadcast a Tensor `t` to target_shape using unsqueeze + addition with zeros-like.
def _tensor_broadcast_to(t, target_shape):
    """
    Expand and broadcast Tensor `t` to target_shape.
    Implementation uses reshape/unsqueeze to add leading singleton dims when needed,
    then arithmetic with zeros-like(AP array) to force broadcasting without xp.broadcast_to.
    """
    # current shape
    cur_shape = tuple(getattr(t, "shape", ()))
    nd_cur = len(cur_shape)
    nd_target = len(target_shape)
    # if target has more dims, prepend singleton dims
    if nd_target > nd_cur:
        new_shape = (1,) * (nd_target - nd_cur) + cur_shape
        t = t.reshape(new_shape)
    # Now t may have fewer dims but broadcasting rules will handle remaining axes;
    # to ensure full broadcastability, create a zeros-like tensor of target_shape and add 0*zeros
    zeros_arr = _ap.Array.zeros(target_shape)
    zeros_t = _ensure_tensor_like(zeros_arr, like_tensor=t)
    return t + zeros_t * 0.0


def _maybe_cast_rhs_to_lhs_dtype(lhs_raw, rhs_raw, xp):
    """Cast rhs_raw to dtype of lhs_raw if possible (to mimic torch scalar casting)."""
    try:
        lhs_dtype = getattr(lhs_raw, "dtype", None)
        if lhs_dtype is not None:
            return xp.asarray(rhs_raw, dtype=lhs_dtype)
    except Exception:
        pass
    # fallback
    return xp.asarray(rhs_raw)


def _as_float_if_needed(arr, xp):
    """
    Ensure `arr` is floating dtype in the given backend xp.
    If arr.dtype is integer, return arr.astype(xp.float32).
    This avoids xp.finfo failing on integer dtypes.
    """
    try:
        dt = getattr(arr, "dtype", None)
        if dt is None:
            return xp.asarray(arr, dtype=xp.float32)
        # xp.issubdtype works for numpy/cupy
        if not xp.issubdtype(dt, xp.floating):
            return xp.asarray(arr, dtype=xp.float32)
        return arr
    except Exception:
        return xp.asarray(arr, dtype=xp.float32)
