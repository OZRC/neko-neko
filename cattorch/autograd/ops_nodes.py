# cattorch/autograd/ops_nodes.py
"""
OpNode implementations used by cattorch.

This is a self-contained, drop-in file. It implements:
 - helper utilities (detect_xp_from_obj, to_raw, to_ap, unbroadcast)
 - OpNode base class and many ops (Add, Sub, Mul, Div, Pow, Exp, Log, Sqrt, Abs, Trig, Tanh, Sigmoid, etc.)
 - CopyFromOp for mapping gradients in in-place semantics
 - Inplace ops (IAdd/ISub/IMul/IDiv/IMod) with version checks
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..backend import _array as _ap
from ..backend.utils import detect_xp_from_obj, to_ap, to_raw

# Tensor-mode helpers to support create_graph=True in engine.
# These helpers are lightweight wrappers to allow OpNode.backward to accept
# either raw arrays (legacy) or Tensor arguments (tensor-mode).
try:
    from ..tensor import Tensor as _Tensor
except Exception:
    _Tensor = None  # will import lazily in functions
from ._helpers import (
    _ensure_tensor_like,
    unbroadcast,
    _tensor_unbroadcast,
    _tensor_broadcast_to,
    _maybe_cast_rhs_to_lhs_dtype,
    _as_float_if_needed,
)


# ---------- base class ----------
class OpNode:
    __slots__ = ("inputs", "output", "metadata")

    def __init__(self, inputs: List, output, metadata: Optional[Dict[str, Any]] = None):
        self.inputs = inputs
        self.output = output
        self.metadata = metadata or {}
        try:
            output.op = self
        except Exception:
            pass

    def backward(self, grad_out_raw):
        raise NotImplementedError()

    def describe(self) -> str:
        inp_descr = ", ".join(
            f"{getattr(t, 'name', 'T')}{tuple(getattr(t, 'shape', ()))}"
            for t in self.inputs
        )
        out_descr = f"{getattr(self.output, 'name', 'out')}{tuple(getattr(self.output, 'shape', ()))}"
        return f"{type(self).__name__}: inputs=[{inp_descr}] -> out={out_descr}"

# -------------------------------
class ScatterAddOp(OpNode):
    @classmethod
    def create(cls, template, idx, src, dim: Optional[int] = None):
        """
        template: Tensor whose shape defines the output shape (no gradient wrt template by default)
        idx: indexing object (Tensor or raw), used to place src into output
        src: Tensor or array-like to add into output at idx
        dim: optional dim for 1D-style scatter (if your index format expects a dim)
        """
        from ..tensor import Tensor

        xp = detect_xp_from_obj(template)
        # Prepare raw arrays for numeric forward value
        template_raw = to_raw(template.data)
        # get idx_raw (can be tuple, array, or Tensor)
        if hasattr(idx, "data"):
            try:
                idx_raw = to_raw(idx.data)
            except Exception:
                idx_raw = idx
        else:
            idx_raw = idx
        # get src raw
        if isinstance(src, Tensor):
            src_raw = to_raw(src.data)
        else:
            src_raw = to_raw(getattr(src, "data", src))
        # compute numeric out by scattering src into zeros of template shape
        out_raw = xp.zeros_like(template_raw)
        try:
            # Prefer numpy/cupy.add.at (works for many cases)
            # Note: when idx is tuple-of-arrays for advanced indexing, numpy.add.at handles it.
            xp.add.at(out_raw, idx_raw, src_raw)
        except Exception:
            # Fallback: try simple gather semantics for 1D index arrays.
            try:
                out_raw[idx_raw] += src_raw
            except Exception:
                # Generic fallback: iterate (slow)
                try:
                    flat = out_raw.ravel()
                    flat_src = src_raw.ravel()
                    flat_idx = idx_raw.ravel()
                    for i, v in enumerate(flat_src):
                        flat[flat_idx[i]] += v
                    out_raw = flat.reshape(template_raw.shape)
                except Exception:
                    # give up: raise
                    raise

        out_tensor = Tensor(
            to_ap(out_raw),
            requires_grad=(
                getattr(template, "requires_grad", False)
                or (isinstance(src, Tensor) and getattr(src, "requires_grad", False))
            ),
            device=getattr(template, "device", None),
        )
        # create node and metadata
        meta = {"idx": idx_raw, "dim": dim, "xp": xp}
        node = cls([template, idx, src], out_tensor, metadata=meta)
        return node

    def backward(self, grad_out):
        """
        grad_out: either raw array (legacy) or Tensor (tensor-mode).
        returns gradients for [template, idx, src] respectively.
        - grad wrt src is gather(grad_out, idx)
        - template and idx have no gradient (None)
        """
        from ..tensor import Tensor

        xp = self.metadata.get("xp", __import__("numpy"))
        idx = self.metadata.get("idx")
        # If grad_out is a Tensor -> perform Tensor gather using idx (preserves graph)
        if isinstance(grad_out, Tensor):
            # if idx was passed as a Tensor input, prefer that existing Tensor object
            # but metadata stored idx_raw; attempt to recover Tensor input from self.inputs[1] if available
            idx_input = None
            if len(self.inputs) >= 2:
                cand = self.inputs[1]
                try:
                    if isinstance(cand, Tensor):
                        idx_input = cand
                except Exception:
                    idx_input = None
            # If idx_input exists (Tensor) use it for indexing; else try to wrap idx as Tensor-like
            if idx_input is not None:
                idx_for_index = idx_input
            else:
                # wrap raw idx into Tensor (requires_grad=False)
                try:
                    idx_for_index = _ensure_tensor_like(idx, like_tensor=grad_out)
                except Exception:
                    idx_for_index = idx  # fallback: raw index, expect backend to accept
            # gather gradient for src by fancy-indexing grad_out with idx
            try:
                # For many tensor implementations fancy indexing with a Tensor works: grad_out[idx_for_index]
                # If idx_for_index is actually raw, using grad_out[idx] may also work.
                src_grad = grad_out[idx_for_index]
            except Exception:
                try:
                    src_grad = grad_out[idx]  # raw idx fallback
                except Exception:
                    # Last fallback: try take (1D)
                    try:
                        # If idx is 1D integer array: use take
                        src_grad = grad_out.take(idx, axis=0)
                    except Exception:
                        # unsupported indexing style in tensor-mode: return None to signal failure
                        return [None, None, None]
            # return grads: no grad for template, no grad for index, grad for src
            return [None, None, src_grad]

        # legacy raw-array path:
        g_raw = xp.asarray(grad_out)
        idx_raw = idx
        try:
            src_grad_raw = g_raw[idx_raw]
        except Exception:
            # try take
            try:
                src_grad_raw = xp.take(g_raw, idx_raw, axis=0)
            except Exception:
                # generic fallback: zeros
                src_grad_raw = xp.zeros_like(to_raw(self.inputs[2].data))
        return [None, None, src_grad_raw]


# ---------- identity ----------
class IdentityOp(OpNode):
    @classmethod
    def create(cls, inp, out_name: str = "out"):
        from ..tensor import Tensor

        out_tensor = Tensor(
            inp.data, requires_grad=getattr(inp, "requires_grad", False)
        )
        out_tensor.name = out_name
        try:
            out_tensor.device = getattr(inp, "device", None)
        except Exception:
            pass
        node = cls(inputs=[inp], output=out_tensor, metadata={"indices": None})
        try:
            out_tensor.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out):
        """
        Identity: gradient flows unchanged.
        Works for both Tensor-mode and legacy raw-mode.
        """
        # if Tensor: return Tensor (keeps grad_fn)
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        if Tensor is not None and isinstance(grad_out, Tensor):
            return [grad_out]
        # legacy raw
        xp = self.metadata.get("xp", __import__("numpy"))
        g_raw = xp.asarray(grad_out)
        return [g_raw]


# -------------------------
# Helpers for backend detection / conversions
# -------------------------


def diagonal_backward(
    grad_out, input_shape: Tuple[int, ...], ax0: int, ax1: int, out_axis: int, xp
):
    """
    Transpose of selecting diagonal: scatter `grad_out` back into the input shape.
    This is a simple and explicit implementation; not optimized for speed.
    """
    grad_in = xp.zeros(input_shape, dtype=grad_out.dtype)
    N = input_shape[ax0]
    for k in range(N):
        in_sl = [slice(None)] * len(input_shape)
        in_sl[ax0] = k
        in_sl[ax1] = k
        out_sl = [slice(None)] * grad_out.ndim
        out_sl[out_axis] = k
        grad_in[tuple(in_sl)] += grad_out[tuple(out_sl)]
    return grad_in


class DiagonalOp(OpNode):
    @classmethod
    def create(cls, inp, ax0: int, ax1: int, index):
        """
        Args:
            inp (cattorch.Tensor):
            ax0 (int):
            ax1 (int):
            index (cattorch.einsum.Index)  # kept for compatibility with your code
        """
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        a_raw = to_raw(inp.data)
        # forward diagonal, then move last axis to ax0 (match your original)
        out_val = xp.diagonal(a_raw, axis1=ax0, axis2=ax1)
        out_val = xp.moveaxis(out_val, -1, ax0)
        out_tensor = Tensor(
            to_ap(out_val), requires_grad=getattr(inp, "requires_grad", False)
        )
        out_tensor.name = f"{getattr(inp, 'name', 'T')}_diag"

        # Precompute simple diag_idx for common 2D-like case.
        # If input is 2D, diag_idx is tuple (rows, cols).
        # For higher dims we keep diag_idx = None (fallback to raw-loop backward).
        in_shape = tuple(getattr(inp, "shape", ()))
        diag_idx = None
        try:
            import numpy as _np

            nd = len(in_shape)
            if nd == 2:
                # For 2D, diagonal positions (k,k)
                # limit diag length to min dimensions to avoid out-of-bounds
                diag_len = min(in_shape[ax0], in_shape[ax1])
                rows = _np.arange(diag_len, dtype=_np.int64)
                cols = _np.arange(diag_len, dtype=_np.int64)
                diag_idx = (rows, cols)
            else:
                # For ndim > 2, building a fully general diag_idx is trickier:
                # we set diag_idx = None and fallback to diagonal_backward on raw arrays
                diag_idx = None
        except Exception:
            diag_idx = None

        node = cls(
            inputs=[inp],
            output=out_tensor,
            metadata={
                "ax0": ax0,
                "ax1": ax1,
                "input_shape": in_shape,
                "out_axis": ax0,
                "index": index,
                "indices": diag_idx,
                "xp": xp,
            },
        )
        try:
            out_tensor.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out_raw):
        """
        Backward supports two modes:
         - legacy/raw: use diagonal_backward on raw arrays (exactly like your original).
         - tensor-mode (if grad_out_raw is a Tensor): try to build a ScatterAddOp (non-inplace)
           for the 2D-case (diag indices precomputed). If ScatterAddOp not available or diag_idx
           is None (complex case), fallback to performing diagonal_backward on raw and wrapping result.
        """
        from ..tensor import Tensor

        xp = self.metadata.get("xp", np)
        ax0 = int(self.metadata["ax0"])
        ax1 = int(self.metadata["ax1"])
        input_shape = tuple(self.metadata["input_shape"])
        out_axis = int(self.metadata.get("out_axis", ax0))

        # If grad_out_raw is a Tensor (tensor-mode), attempt Tensor-aware backward
        if isinstance(grad_out_raw, Tensor):
            # First attempt: if we have simple precomputed diag indices (2D case) and ScatterAddOp available,
            # create a ScatterAddOp to scatter grad_out back into input positions along the diagonal.
            diag_idx = self.metadata.get("indices", None)
            if diag_idx is not None:
                try:
                    # Try to use ScatterAddOp (must be defined in this module)
                    # Note: we pass `self.inputs[0]` as template, diag_idx as index, and grad_out_raw as src
                    node = ScatterAddOp.create(self.inputs[0], diag_idx, grad_out_raw)
                    out_t = node.output
                    # attach node metadata so engine can traverse newly created op during higher-order backward
                    try:
                        out_t.grad_fn = node
                        out_t.grad_fn_name = node.__class__.__name__
                        # Attempt to add parents; if method not present, ignore (engine traverses via node)
                        try:
                            out_t._add_parents(self.inputs[0], None, grad_out_raw)
                        except Exception:
                            pass
                        out_t._is_leaf = False
                    except Exception:
                        pass
                    return [out_t]
                except Exception:
                    # if ScatterAddOp creation failed, we fall back to raw-based diagonal_backward
                    pass

            # Fallback: compute via raw arrays then wrap into Tensor.
            try:
                raw = to_raw(grad_out_raw.data)
            except Exception:
                raw = to_raw(grad_out_raw)
            gin_raw = diagonal_backward(raw, input_shape, ax0, ax1, out_axis, xp)
            # wrap as Tensor (requires_grad=False) to avoid accidental graph mixing
            gin_t = _ensure_tensor_like(gin_raw, like_tensor=grad_out_raw)
            return [gin_t]

        # Legacy raw path (exactly original behavior)
        g_raw = xp.asarray(grad_out_raw)
        gin = diagonal_backward(g_raw, input_shape, ax0, ax1, out_axis, xp)
        return [gin]


class ExpandDimsOp(OpNode):
    @classmethod
    def create(cls, inp, dim: int, indices_in=None, indices_out=None):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        a_raw = to_raw(inp.data)
        out_val = xp.expand_dims(a_raw, axis=dim)
        out_tensor = Tensor(
            to_ap(out_val), requires_grad=getattr(inp, "requires_grad", False)
        )
        out_tensor.name = getattr(inp, "name", "T") + "_expand"
        node = cls(
            inputs=[inp],
            output=out_tensor,
            metadata={
                "dim": dim,
                "in_shape": tuple(getattr(inp, "shape", ())),
                "out_shape": tuple(getattr(out_tensor, "shape", ())),
                "indices": {"in": indices_in, "out": indices_out},
                "xp": xp,
            },
        )
        try:
            out_tensor.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out_raw):
        """
        Backward for ExpandDims:
        - legacy/raw: sum along expanded axis then unbroadcast (original behavior).
        - tensor-mode: if grad_out_raw is a Tensor, perform Tensor.sum(axis) then _tensor_unbroadcast
          so the returned gradient is a Tensor and keeps grad_fn (supports create_graph).
        """
        # lazy import Tensor for detection
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        xp = self.metadata.get("xp", np)
        dim = self.metadata["dim"]
        target_shape = tuple(self.inputs[0].shape)

        # Tensor-mode branch: grad_out_raw is a Tensor
        if Tensor is not None and isinstance(grad_out_raw, Tensor):
            g_t = grad_out_raw
            # use Tensor.sum to collapse the expanded axis (keepdims=False by default)
            g_summed = g_t.sum(dim=dim)
            # ensure we reduce to the exact target shape (use helper that sums extra dims if needed)
            return [_tensor_unbroadcast(g_summed, target_shape)]

        # Legacy/raw branch: exact original behavior
        g_raw = xp.asarray(grad_out_raw)
        # sum over the expanded axis to get gradient for original shape
        gin = xp.sum(g_raw, axis=dim)
        # ensure broadcast consistency (use your existing unbroadcast helper)
        return [unbroadcast(gin, target_shape, xp)]


# ---------- reshape/transpose/broadcast/view ----------
class ReshapeOp(OpNode):
    @classmethod
    def create(
        cls,
        inp,
        new_shape: Tuple[int, ...],
        indices_in=None,
        indices_out=None,
    ):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        a_raw = to_raw(inp.data)
        out_val = xp.reshape(a_raw, new_shape)
        out_tensor = Tensor(
            to_ap(out_val), requires_grad=getattr(inp, "requires_grad", False)
        )
        out_tensor.name = f"{getattr(inp, 'name', 'T')}_reshape"
        node = cls(
            inputs=[inp],
            output=out_tensor,
            metadata={
                "in_shape": tuple(getattr(inp, "shape", ())),
                "out_shape": new_shape,
                "indices": {"in": indices_in, "out": indices_out},
                "xp": xp,
            },
        )
        try:
            out_tensor.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out):
        """Tensor-aware backward: reshape gradient back to input shape."""
        # lazy import tensor class to avoid circulars
        from ..tensor import Tensor

        in_shape = tuple(self.metadata["in_shape"])
        # If grad_out is a Tensor, use Tensor reshape to preserve graph
        if isinstance(grad_out, Tensor):
            return [grad_out.reshape(in_shape)]
        # legacy raw-array path
        xp = self.metadata.get("xp", __import__("numpy"))
        g_raw = xp.asarray(grad_out)
        gin = xp.reshape(g_raw, in_shape)
        return [gin]


class SwapaxesOp(OpNode):
    """
    SwapaxesOp: swap two axes ax0 and ax1 of the input.
    - Forward: out = xp.swapaxes(a, ax0, ax1)  (or xp.transpose with perm)
    - Backward: gradient is swapped back: grad_in = swapaxes(grad_out, ax0, ax1)
    """

    @classmethod
    def create(cls, inp, ax0: int, ax1: int, indices_in=None, indices_out=None):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        a_raw = to_raw(inp.data)

        # normalize negative axes relative to input ndim
        in_shape = tuple(getattr(inp, "shape", ()))
        nd = len(in_shape)
        if ax0 < 0:
            ax0n = nd + ax0
        else:
            ax0n = ax0
        if ax1 < 0:
            ax1n = nd + ax1
        else:
            ax1n = ax1

        # perform swap on raw array
        try:
            out_raw = xp.swapaxes(a_raw, ax0n, ax1n)
        except Exception:
            # fallback: construct permutation and use transpose
            perm = list(range(nd))
            perm[ax0n], perm[ax1n] = perm[ax1n], perm[ax0n]
            out_raw = xp.transpose(a_raw, axes=perm)

        out_tensor = Tensor(
            to_ap(out_raw),
            requires_grad=getattr(inp, "requires_grad", False),
            device=getattr(inp, "device", None),
        )
        node = cls(
            inputs=[inp],
            output=out_tensor,
            metadata={
                "ax0": int(ax0n),
                "ax1": int(ax1n),
                "in_shape": in_shape,
                "out_shape": tuple(getattr(out_tensor, "shape", ())),
                "indices": {"in": indices_in, "out": indices_out},
                "xp": xp,
            },
        )
        try:
            out_tensor.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out):
        """
        Backward: swap axes of grad_out back to input layout.
        Supports Tensor-mode (returns Tensor) and legacy raw-mode (returns raw array).
        """
        # lazy import Tensor class
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        xp = self.metadata.get("xp", __import__("numpy"))
        ax0 = int(self.metadata["ax0"])
        ax1 = int(self.metadata["ax1"])
        in_shape = tuple(self.metadata.get("in_shape", ()))

        # Tensor-mode branch
        if Tensor is not None and isinstance(grad_out, Tensor):
            g_t = grad_out
            # prefer a Tensor API: try transpose(ax0, ax1) then swapaxes
            try:
                # many Tensor impls provide transpose with two args for swapping axes
                g_in = g_t.transpose(ax0, ax1)
            except Exception:
                try:
                    g_in = g_t.swapaxes(ax0, ax1)
                except Exception:
                    # fallback: build permutation and call transpose
                    try:
                        rank = len(getattr(g_t, "shape", ()))
                        perm = list(range(rank))
                        perm[ax0], perm[ax1] = perm[ax1], perm[ax0]
                        g_in = g_t.transpose(tuple(perm))
                    except Exception:
                        # last fallback: convert to raw, do swapaxes, wrap back (loses graph)
                        raw = to_raw(g_t.data)
                        raw_swapped = xp.swapaxes(raw, ax0, ax1)
                        return [_ensure_tensor_like(raw_swapped, like_tensor=g_t)]
            # unbroadcast to input shape if necessary
            try:
                g_in = _tensor_unbroadcast(g_in, in_shape)
            except Exception:
                try:
                    g_in = g_in.reshape(in_shape)
                except Exception:
                    pass
            return [g_in]

        # Legacy raw branch
        g_raw = xp.asarray(grad_out)
        try:
            gin_raw = xp.swapaxes(g_raw, ax0, ax1)
        except Exception:
            # fallback to transpose with perm
            rank = len(in_shape)
            perm = list(range(rank))
            perm[ax0], perm[ax1] = perm[ax1], perm[ax0]
            gin_raw = xp.transpose(g_raw, axes=perm)
        gin_raw = unbroadcast(gin_raw, in_shape, xp)
        return [gin_raw]


class TransposeOp(OpNode):
    @classmethod
    def create(cls, inp, perm: List[int], indices_in=None, indices_out=None):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        a_raw = to_raw(inp.data)
        out_val = xp.transpose(a_raw, axes=perm)
        out_tensor = Tensor(
            to_ap(out_val), requires_grad=getattr(inp, "requires_grad", False)
        )
        out_tensor.name = f"{getattr(inp, 'name', 'T')}_T"
        node = cls(
            inputs=[inp],
            output=out_tensor,
            metadata={
                "perm": perm,
                "indices": {"in": indices_in, "out": indices_out},
                "xp": xp,
            },
        )
        try:
            out_tensor.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out):
        from ..tensor import Tensor

        perm = self.metadata["perm"]
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        # If grad_out is a Tensor, use Tensor reshape to preserve graph
        if isinstance(grad_out, Tensor):
            return [grad_out.transpose(inv)]
        # legacy raw-array path
        xp = self.metadata.get("xp", np)
        g = xp.asarray(grad_out)
        gin = xp.transpose(g, axes=inv)
        return [gin]


class BroadcastOp(OpNode):
    @classmethod
    def create(
        cls,
        inp,
        target_shape: Tuple[int, ...],
        indices_in=None,
        indices_out=None,
    ):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        a_raw = to_raw(inp.data)
        out_val = xp.broadcast_to(a_raw, target_shape)
        out_val = xp.copy(out_val)  # ensure writable
        out_tensor = Tensor(
            to_ap(out_val), requires_grad=getattr(inp, "requires_grad", False)
        )
        if getattr(inp, "name", "T") is not None:
            out_tensor.name = getattr(inp, "name", "T") + "_bcast"
        node = cls(
            inputs=[inp],
            output=out_tensor,
            metadata={
                "in_shape": tuple(getattr(inp, "shape", ())),
                "out_shape": tuple(target_shape),
                "indices": {"in": indices_in, "out": indices_out},
                "xp": xp,
            },
        )
        try:
            out_tensor.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out):
        from ..tensor import Tensor

        in_shape = self.metadata["in_shape"]
        if isinstance(grad_out, Tensor):
            # use tensor unbroadcast helper
            return [_tensor_unbroadcast(grad_out, in_shape)]
        # legacy raw-array path
        xp = self.metadata.get("xp", np)
        g_raw = xp.asarray(grad_out)
        return [unbroadcast(g_raw, in_shape, xp)]


class IndexOp(OpNode):
    @classmethod
    def create(cls, inp, idx):
        from ..tensor import Tensor  # lazy

        xp = (
            detect_xp_from_obj(inp)
            if "detect_xp_from_obj" in globals()
            else (inp.data.xp if hasattr(inp.data, "xp") else __import__("numpy"))
        )
        raw = to_raw(inp.data)
        sliced = raw[idx]
        out = Tensor(
            to_ap(sliced),
            requires_grad=inp.requires_grad,
            device=inp.device,
        )
        return cls([inp], out, {"idx": idx, "in_shape": tuple(inp.shape), "xp": xp})

    def backward(self, grad_out):
        from ..tensor import Tensor

        if isinstance(grad_out, Tensor):
            # create a ScatterAddOp which produces grad for the input by scattering grad_out
            # create node
            node = ScatterAddOp.create(self.inputs[0], self.metadata["idx"], grad_out)
            out_t = node.output
            # attach grad_fn and parents so that engine can traverse this new node when computing higher derivatives
            out_t.grad_fn = node
            out_t.grad_fn_name = node.__class__.__name__
            out_t._add_parents(
                self.inputs[0],
                self.inputs[1] if len(self.inputs) > 1 else None,
                grad_out,
            )
            out_t._is_leaf = False
            # return gradient for the input (the scatter result tensor), no grads for other things
            return [out_t]

        xp = self.metadata.get("xp", __import__("numpy"))
        inp = self.inputs[0]
        g = xp.asarray(grad_out)
        grad_in = xp.zeros(to_raw(inp.data).shape, dtype=g.dtype)
        xp.add.at(grad_in, self.metadata["idx"], g)
        return [grad_in]


class MaskedFillOp(OpNode):
    """
    masked_fill: out = where(mask, value, a)
    Inputs in node.inputs are [a, mask, value] (mask/value may be scalars or Tensors).
    Gradients:
      - grad_a = grad_out * (~mask)
      - grad_mask = zeros (mask is treated non-differentiable)
      - grad_value = grad_out * mask  (and unbroadcast to value.shape if value is a tensor)
    """

    @classmethod
    def create(cls, a, mask, value):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)

        # mask can be Tensor or array/scalar
        mask_raw = to_raw(mask.data) if hasattr(mask, "data") else to_raw(mask)
        mask_arr = xp.asarray(mask_raw)
        # convert to boolean mask (treat non-zero as True)
        mask_bool = mask_arr.astype(bool)

        # value may be Tensor or scalar/array
        value_raw = to_raw(value.data) if hasattr(value, "data") else to_raw(value)
        # cast value to a's dtype where it makes sense (mimic torch scalar casting)
        value_raw = _maybe_cast_rhs_to_lhs_dtype(a_raw, value_raw, xp)

        out_val = xp.where(mask_bool, value_raw, a_raw)
        out = Tensor(
            to_ap(out_val),
            requires_grad=(a.requires_grad or getattr(value, "requires_grad", False)),
            device=a.device,
        )

        # include inputs list in same order as create args for consistent backward outputs
        return cls(
            [a, mask, value],
            out,
            {
                "xp": xp,
                # store boolean mask as backend array to reuse in backward (no grads for mask)
                "mask_bool": xp.asarray(mask_bool),
                "a_shape": tuple(a.shape),
                "value_shape": tuple(getattr(value, "shape", ())),
            },
        )

    def backward(self, grad_out_raw):
        """
        Tensor-aware backward + legacy raw:
         - If grad_out_raw is a Tensor -> compute using Tensor ops and return Tensors
         - Else -> use raw xp arrays and return raw arrays
        Returns: [grad_a, grad_mask, grad_value]
        """
        # lazy import Tensor class
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        xp = self.metadata["xp"]
        mask_bool = self.metadata["mask_bool"]

        a = self.inputs[0]
        mask = self.inputs[1]
        value = self.inputs[2]

        # TARGET SHAPES
        a_shape = tuple(getattr(a, "shape", ()))
        value_shape = tuple(getattr(value, "shape", ()))

        # -----------------------
        # Tensor-mode branch
        # -----------------------
        if Tensor is not None and isinstance(grad_out_raw, Tensor):
            g = grad_out_raw  # Tensor

            # get mask as Tensor (prefer mask input if it is a Tensor)
            if hasattr(mask, "data") and isinstance(mask, Tensor):
                mask_t = mask
            else:
                # wrap stored boolean mask into a Tensor (requires_grad=False)
                mask_t = _ensure_tensor_like(mask_bool, like_tensor=g)

            # inv mask numeric cast (same dtype as grad)
            inv_mask_num = (~mask_t).astype(g.dtype)
            # grad w.r.t a: g * (~mask)
            ga_t = g * inv_mask_num
            ga_t = _tensor_unbroadcast(ga_t, a_shape)

            # grad w.r.t mask: zeros shaped like mask (mask non-diff)
            if hasattr(mask, "data"):
                # create zeros with xp then wrap as Tensor (requires_grad=False)
                mb_raw = to_raw(mask.data)
                z_raw = xp.zeros_like(mb_raw, dtype=getattr(g, "dtype", None))
                g_mask_t = _ensure_tensor_like(z_raw, like_tensor=g)
            else:
                g_mask_t = None

            # grad w.r.t value: only positions where mask is True
            if hasattr(value, "data") and getattr(value, "requires_grad", False):
                mask_num = mask_t.astype(g.dtype)
                gv_t = g * mask_num
                gv_t = _tensor_unbroadcast(gv_t, value_shape)
            else:
                # if value exists but doesn't require grad, return zeros of its shape
                if hasattr(value, "data"):
                    vr = to_raw(value.data)
                    z_raw = xp.zeros_like(vr, dtype=getattr(g, "dtype", None))
                    gv_t = _ensure_tensor_like(z_raw, like_tensor=g)
                else:
                    gv_t = None

            return [ga_t, g_mask_t, gv_t]

        # -----------------------
        # Legacy raw branch
        # -----------------------
        g_raw = xp.asarray(grad_out_raw)

        # grad w.r.t a
        inv_mask = (~mask_bool).astype(g_raw.dtype)
        ga_raw = g_raw * inv_mask
        ga = unbroadcast(ga_raw, a_shape, xp)

        # grad w.r.t mask: zeros like mask (if mask has data), else None
        if hasattr(mask, "data"):
            mb_raw = to_raw(mask.data)
            g_mask = xp.zeros_like(mb_raw, dtype=g_raw.dtype)
            g_mask = unbroadcast(g_mask, tuple(getattr(mask, "shape", ())), xp)
        else:
            g_mask = None

        # grad w.r.t value
        if hasattr(value, "data") and getattr(value, "requires_grad", False):
            mask_as_num = mask_bool.astype(g_raw.dtype)
            gv_raw = g_raw * mask_as_num
            gv = unbroadcast(gv_raw, value_shape, xp)
        else:
            if hasattr(value, "data"):
                vr = to_raw(value.data)
                gv_raw = xp.zeros_like(vr, dtype=g_raw.dtype)
                gv = unbroadcast(gv_raw, value_shape, xp)
            else:
                gv = None

        return [ga, g_mask, gv]


# ---------- elementwise binary ----------
class AddOp(OpNode):
    @classmethod
    def create(cls, a, b):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        # cast rhs to lhs dtype when possible to mimic torch scalar/tensor casting
        b_raw = _maybe_cast_rhs_to_lhs_dtype(a_raw, b_raw, xp)
        out_val = xp.add(a_raw, b_raw)
        out = Tensor(
            to_ap(out_val),
            requires_grad=(a.requires_grad or getattr(b, "requires_grad", False)),
            device=a.device,
        )
        return cls(
            [a, b],
            out,
            {
                "a_shape": tuple(a.shape),
                "b_shape": tuple(getattr(b, "shape", ())),
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        """Elementwise add: gradient flows unchanged to both inputs (with unbroadcast)."""
        from ..tensor import Tensor

        a_shape = tuple(self.metadata.get("a_shape", ()))
        b_shape = tuple(self.metadata.get("b_shape", ()))
        if isinstance(grad_out, Tensor):
            ga = _tensor_unbroadcast(grad_out, a_shape)
            gb = _tensor_unbroadcast(grad_out, b_shape)
            return [ga, gb]

        xp = self.metadata.get("xp", __import__("numpy"))
        g_raw = xp.asarray(grad_out)
        return [unbroadcast(g_raw, a_shape, xp), unbroadcast(g_raw, b_shape, xp)]


class SubOp(OpNode):
    @classmethod
    def create(cls, a, b):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        b_raw = _maybe_cast_rhs_to_lhs_dtype(a_raw, b_raw, xp)
        out_val = xp.subtract(a_raw, b_raw)
        out = Tensor(
            to_ap(out_val),
            requires_grad=(a.requires_grad or getattr(b, "requires_grad", False)),
            device=a.device,
        )
        return cls(
            [a, b],
            out,
            {
                "a_shape": tuple(a.shape),
                "b_shape": tuple(getattr(b, "shape", ())),
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        """Elementwise add: gradient flows unchanged to both inputs (with unbroadcast)."""
        from ..tensor import Tensor

        a_shape = tuple(self.metadata.get("a_shape", ()))
        b_shape = tuple(self.metadata.get("b_shape", ()))
        if isinstance(grad_out, Tensor):
            ga = _tensor_unbroadcast(grad_out, a_shape)
            gb = _tensor_unbroadcast(-grad_out, b_shape)
            return [ga, gb]

        xp = self.metadata.get("xp", __import__("numpy"))
        g_raw = xp.asarray(grad_out)
        return [unbroadcast(g_raw, a_shape, xp), unbroadcast(-g_raw, b_shape, xp)]


class MulOp(OpNode):
    @classmethod
    def create(cls, a, b):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        b_raw = _maybe_cast_rhs_to_lhs_dtype(a_raw, b_raw, xp)
        out_val = xp.multiply(a_raw, b_raw)
        out = Tensor(
            to_ap(out_val),
            requires_grad=(a.requires_grad or getattr(b, "requires_grad", False)),
            device=a.device,
        )
        return cls([a, b], out, {"xp": xp})

    def backward(self, grad_out):
        from ..tensor import Tensor

        a, b = self.inputs
        if isinstance(grad_out, Tensor):
            # ensure operands are tensors (wrap raw as Tensor with requires_grad=False)
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            b_t = (
                b
                if isinstance(b, Tensor)
                else _ensure_tensor_like(getattr(b, "data", b), like_tensor=grad_out)
            )
            ga = grad_out * b_t
            gb = grad_out * a_t
            return [
                _tensor_unbroadcast(ga, tuple(a.shape)),
                _tensor_unbroadcast(gb, tuple(b.shape)),
            ]

        xp = self.metadata["xp"]
        g = xp.asarray(grad_out)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data)
        ga = g * b_raw
        gb = g * a_raw
        return [
            unbroadcast(ga, tuple(a.shape), xp),
            unbroadcast(gb, tuple(b.shape), xp),
        ]


class DivOp(OpNode):
    @classmethod
    def create(cls, a, b):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        # ensure we have raw arrays
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        # cast rhs to lhs dtype when possible to mimic torch scalar/tensor casting
        b_raw = _maybe_cast_rhs_to_lhs_dtype(a_raw, b_raw, xp)

        # --- stable denominator ---
        # ensure floating dtype for finfo and safe thresholding
        a_raw_float = _as_float_if_needed(a_raw, xp)
        b_raw_float = xp.asarray(b_raw, dtype=a_raw_float.dtype)

        finfo = xp.finfo(a_raw_float.dtype)
        eps_dtype = finfo.eps
        # threshold scales with magnitude so we preserve relative scale
        threshold = eps_dtype * xp.maximum(1.0, xp.abs(b_raw_float))
        sign = xp.sign(b_raw_float)
        sign = xp.where(sign == 0, 1.0, sign)  # make sign(0)=>+1
        denom = xp.where(xp.abs(b_raw_float) < threshold, sign * threshold, b_raw_float)
        # ensure denom is finite (replace NaN/inf by safe threshold)
        denom = xp.where(xp.isfinite(denom), denom, sign * threshold)
        # ------------------------------------------------

        out_val = a_raw_float / denom  # safe division with denom clamped
        out = Tensor(
            to_ap(out_val),
            requires_grad=(a.requires_grad or getattr(b, "requires_grad", False)),
            device=a.device,
        )
        return cls([a, b], out, {"xp": xp})

    def backward(self, grad_out):
        """
        Tensor-aware backward: if grad_out is a Tensor -> compute using Tensor ops and
        return Tensor gradients (keeping grad_fn so create_graph works).
        Otherwise fallback to legacy raw array path (clamped denom logic present in forward).
        """
        # lazy import Tensor
        try:
            from .. import finfo as ctfinfo
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        a, b = self.inputs

        # ---- Tensor-mode branch ----
        if Tensor is not None and isinstance(grad_out, Tensor):
            # Wrap inputs as Tensors if needed so arithmetic builds graph
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            b_t = (
                b
                if isinstance(b, Tensor)
                else _ensure_tensor_like(getattr(b, "data", b), like_tensor=grad_out)
            )
            eps_scalar = ctfinfo(b_t.dtype).eps
            sign = b_t.sign()
            abs_b = b_t.abs()
            sign = sign.where(sign != 0, 1.0)
            threshold = eps_scalar * b_t.abs().maximum(1.0)
            denom = b_t.where(abs_b >= threshold, sign * threshold)
            denom = (sign * threshold).where(1 - denom.isfinite(), denom)

            # Straightforward derivative of a/b:
            # d/da (a/b) = 1/b
            # d/db (a/b) = -a / b^2
            # Note: this tensor-branch does NOT re-implement denom clamping used in forward.
            # If you need exact same clamping in tensor-mode, more work is needed to express clamp as Tensor ops.
            recip = 1.0 / denom
            ga = grad_out * recip
            gb = -grad_out * a_t * (recip * recip)

            # Unbroadcast each gradient to match original input shapes
            ga = _tensor_unbroadcast(ga, a.shape)
            gb = _tensor_unbroadcast(gb, b.shape)
            return [ga, gb]

        # ---- Legacy raw branch (exactly as original file) ----
        xp = self.metadata.get("xp", __import__("numpy"))
        g = xp.asarray(grad_out)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data)

        # ensure float versions same as forward
        a_raw_float = _as_float_if_needed(a_raw, xp)
        b_raw_float = xp.asarray(b_raw, dtype=a_raw_float.dtype)

        finfo = xp.finfo(a_raw_float.dtype)
        eps_scalar = finfo.eps

        abs_b = xp.abs(b_raw_float)
        threshold = eps_scalar * xp.maximum(1.0, abs_b)
        sign = xp.sign(b_raw_float)
        sign = xp.where(sign == 0, 1.0, sign)
        denom = xp.where(abs_b < threshold, sign * threshold, b_raw_float)
        denom = xp.where(xp.isfinite(denom), denom, sign * threshold)

        # use reciprocal to avoid denom**2 overflow
        recip = 1.0 / denom  # safe: if denom huge, recip -> 0 (no overflow)
        recip2 = recip * recip  # reciprocal squared (very safe)
        ga = g * recip  # = g / denom
        gb = -g * a_raw_float * recip2  # = -g * a / denom^2, computed via recip^2

        return [
            unbroadcast(ga, tuple(a.shape), xp),
            unbroadcast(gb, tuple(b.shape), xp),
        ]


class FloorDivOp(OpNode):
    @classmethod
    def create(cls, a, b):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        b_raw = _maybe_cast_rhs_to_lhs_dtype(a_raw, b_raw, xp)
        out_val = xp.floor_divide(a_raw, b_raw)
        out = Tensor(to_ap(out_val), requires_grad=False, device=a.device)
        return cls([a, b], out, {"xp": xp})

    def backward(self, grad_out):
        try:
            from ..tensor import Tensor, zeros
        except Exception:
            Tensor = None
        # zero gradient almost everywhere
        # ---- Tensor-mode branch ----
        if Tensor is not None and isinstance(grad_out, Tensor):
            return [
                zeros(tuple(inp.shape), dtype=grad_out.dtype) for inp in self.inputs
            ]
        xp = self.metadata["xp"]
        return [
            xp.zeros(tuple(inp.shape), dtype=to_raw(grad_out).dtype)
            for inp in self.inputs
        ]


class ModOp(OpNode):
    @classmethod
    def create(cls, a, b):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        b_raw = _maybe_cast_rhs_to_lhs_dtype(a_raw, b_raw, xp)
        out_val = xp.mod(a_raw, b_raw)
        out = Tensor(
            to_ap(out_val),
            requires_grad=(a.requires_grad or getattr(b, "requires_grad", False)),
            device=a.device,
        )
        return cls([a, b], out, {"xp": xp})

    def backward(self, grad_out):
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None
        a, b = self.inputs
        if Tensor is not None and isinstance(grad_out, Tensor):
            # Wrap inputs as Tensors if needed so arithmetic builds graph
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            b_t = (
                b
                if isinstance(b, Tensor)
                else _ensure_tensor_like(getattr(b, "data", b), like_tensor=grad_out)
            )
            ga = _tensor_unbroadcast(grad_out, tuple(a_t.shape))
            gb = -grad_out * a_t.floor() / b_t
            gb = _tensor_unbroadcast(gb, tuple(b_t.shape))
            return [ga, gb]

        xp = self.metadata["xp"]
        g = xp.asarray(grad_out)
        ga = unbroadcast(g, tuple(a.shape), xp)
        # Use xp.floor (backend-safe) and unbroadcast
        gb = -g * xp.floor(to_raw(a.data) / to_raw(b.data))
        gb = unbroadcast(gb, tuple(b.shape), xp)
        return [ga, gb]


class PowOp(OpNode):
    @classmethod
    def create(cls, a, b):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        # cast exponent to lhs dtype (important for scalar floats)
        b_raw = _maybe_cast_rhs_to_lhs_dtype(a_raw, b_raw, xp)
        out_val = xp.power(a_raw, b_raw)
        out = Tensor(
            to_ap(out_val),
            requires_grad=(a.requires_grad or getattr(b, "requires_grad", False)),
            device=a.device,
        )
        return cls([a, b], out, {"xp": xp})

    def backward(self, grad_out):
        # lazy import Tensor
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        base, exp = self.inputs
        if Tensor is not None and isinstance(grad_out, Tensor):
            # ensure Tensor representations for base & exp
            base_t = (
                base
                if isinstance(base, Tensor)
                else _ensure_tensor_like(
                    getattr(base, "data", base), like_tensor=grad_out
                )
            )
            exp_t = (
                exp
                if isinstance(exp, Tensor)
                else _ensure_tensor_like(
                    getattr(exp, "data", exp), like_tensor=grad_out
                )
            )
            tiny_t = _ensure_tensor_like(1e-30, base_t)
            # handle possible broadcasting of exponent/base
            # gradient wrt a: b * a^(b-1)
            # gradient wrt b: a^b * log(|a|)
            ga = grad_out * (exp_t * (base_t.power(exp_t - 1.0)))
            gb = grad_out * (base_t.power(exp_t) * (base_t.abs() + tiny_t).log())
            return [
                _tensor_unbroadcast(ga, tuple(base.shape)),
                _tensor_unbroadcast(gb, tuple(exp.shape)),
            ]

        xp = self.metadata["xp"]
        g = xp.asarray(grad_out)
        base_raw = to_raw(base.data)
        exp_raw = to_raw(exp.data)
        tiny = xp.asarray(1e-30, dtype=base_raw.dtype)

        ga = g * (exp_raw * (xp.power(base_raw, (exp_raw - 1.0))))
        gb = g * (xp.power(base_raw, exp_raw) * xp.log(xp.abs(base_raw) + tiny))
        return [
            unbroadcast(ga, tuple(base.shape), xp),
            unbroadcast(gb, tuple(exp.shape), xp),
        ]


class MultiMulOp(OpNode):
    @classmethod
    def create(cls, inputs, indices: List[List[str]] = None):
        from ..tensor import Tensor

        if len(inputs) == 0:
            raise ValueError("MultiMulOp.create requires at least one input")
        # choose xp based on first input (assumes all inputs use same backend)
        xp = detect_xp_from_obj(inputs[0])
        # compute forward product numerically (raw arrays)
        out_val = to_raw(inputs[0].data)
        for t in inputs[1:]:
            out_val = out_val * to_raw(t.data)
        req = any(getattr(t, "requires_grad", False) for t in inputs)
        name = "(" + "*".join(getattr(t, "name", "") or "T" for t in inputs) + ")"
        out_tensor = Tensor(
            to_ap(out_val), requires_grad=req, device=getattr(inputs[0], "device", None)
        )
        out_tensor.name = name
        node = cls(
            inputs=inputs,
            output=out_tensor,
            metadata={
                "shapes": [tuple(getattr(t, "shape", ())) for t in inputs],
                "indices": indices,
                "xp": xp,
            },
        )
        try:
            out_tensor.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out):
        """
        Legacy/raw: compute gradient = grad_out * (product of all other inputs)
        Tensor-mode: if grad_out is a Tensor, compute each grad as Tensor expression to preserve graph.
        Returns list of gradients (Tensor in tensor-mode, raw arrays in legacy).
        """
        # lazy import Tensor class
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        xp = self.metadata.get("xp", __import__("numpy"))
        shapes = list(self.metadata.get("shapes", []))
        n = len(self.inputs)

        # Prepare raw values for legacy branch
        raw_vals = [to_raw(t.data) for t in self.inputs]

        # -----------------------
        # Tensor-mode branch
        # -----------------------
        if Tensor is not None and isinstance(grad_out, Tensor):
            # Ensure we have Tensor wrappers for all inputs (wrap raw when necessary)
            tens_inputs = []
            for inp in self.inputs:
                if isinstance(inp, Tensor):
                    tens_inputs.append(inp)
                else:
                    tens_inputs.append(
                        _ensure_tensor_like(
                            getattr(inp, "data", inp), like_tensor=grad_out
                        )
                    )

            # Compute prefix and suffix products as Tensors
            prefix = [None] * n
            suffix = [None] * n
            for i in range(n):
                prefix[i] = tens_inputs[i] if i == 0 else prefix[i - 1] * tens_inputs[i]
            for i in range(n - 1, -1, -1):
                suffix[i] = (
                    tens_inputs[i] if i == n - 1 else suffix[i + 1] * tens_inputs[i]
                )

            grads = []
            g = grad_out
            for i in range(n):
                if n == 1:
                    prod_others = _ensure_tensor_like(1.0, like_tensor=g)
                else:
                    if i == 0:
                        prod_others = suffix[1]
                    elif i == n - 1:
                        prod_others = prefix[n - 2]
                    else:
                        prod_others = prefix[i - 1] * suffix[i + 1]
                gi = g * prod_others
                # unbroadcast to input shape using tensor helper
                gi = _tensor_unbroadcast(gi, tuple(shapes[i]))
                grads.append(gi)
            return grads

        # -----------------------
        # Legacy raw branch
        # -----------------------
        g_raw = xp.asarray(grad_out)
        # compute prefix/suffix on raw arrays
        prefix_raw = [None] * n
        suffix_raw = [None] * n
        for i in range(n):
            prefix_raw[i] = raw_vals[i] if i == 0 else prefix_raw[i - 1] * raw_vals[i]
        for i in range(n - 1, -1, -1):
            suffix_raw[i] = (
                raw_vals[i] if i == n - 1 else suffix_raw[i + 1] * raw_vals[i]
            )

        grads = []
        for i in range(n):
            if n == 1:
                prod_others = 1.0
            else:
                if i == 0:
                    prod_others = suffix_raw[1]
                elif i == n - 1:
                    prod_others = prefix_raw[n - 2]
                else:
                    prod_others = prefix_raw[i - 1] * suffix_raw[i + 1]
            gi = g_raw * prod_others
            gi = unbroadcast(gi, tuple(getattr(self.inputs[i], "shape", ())), xp)
            grads.append(gi)
        return grads


# ---------- unary ----------
class ExpOp(OpNode):
    @classmethod
    def create(cls, a):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        x = to_raw(a.data)

        # Ensure floating dtype for exp and finfo
        x_float = _as_float_if_needed(x, xp)

        finfo = xp.finfo(x_float.dtype)
        # choose margin so exp(max_log) won't overflow
        max_log = xp.log(finfo.max) - 2.0  # -2.0 an extra safety margin
        x_clamped = xp.minimum(x_float, max_log)

        out_val = xp.exp(x_clamped)
        out = Tensor(
            to_ap(out_val),
            requires_grad=a.requires_grad,
            device=a.device,
        )
        # store clamped input maybe useful for backward/consistency (not strictly necessary)
        return cls([a], out, {"xp": xp})

    def backward(self, grad_out):
        # lazy import Tensor class
        try:
            from .. import finfo as ctfinfo
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        a = self.inputs[0]
        # -----------------------
        # Tensor-mode branch
        # -----------------------
        if Tensor is not None and isinstance(grad_out, Tensor):
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            finfo = ctfinfo(a_t.dtype)
            max_log = finfo.max.log() - 2.0
            _clamped = a_t.minimum(max_log)
            exp_x = _clamped.exp()
            return [_tensor_unbroadcast(grad_out * exp_x, tuple(a_t.shape))]

        x = to_raw(a.data)
        xp = self.metadata["xp"]
        g = xp.asarray(grad_out)

        x_float = _as_float_if_needed(x, xp)
        finfo = xp.finfo(x_float.dtype)
        max_log = xp.log(finfo.max) - 2.0
        x_clamped = xp.minimum(x_float, max_log)

        # use the same clamped exp to compute gradient
        exp_x = xp.exp(x_clamped)
        ga = g * exp_x

        return [unbroadcast(ga, tuple(a.shape), xp)]


class LogOp(OpNode):
    @classmethod
    def create(cls, a):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        out_val = xp.log(to_raw(a.data))
        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        return cls([a], out, {"xp": xp})

    def backward(self, grad_out):
        # lazy import Tensor class
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        a = self.inputs[0]
        if Tensor is not None and isinstance(grad_out, Tensor):
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )

            return [_tensor_unbroadcast(grad_out / a_t, a_t.shape)]

        xp = self.metadata["xp"]
        g = xp.asarray(grad_out)
        return [unbroadcast(g / to_raw(a.data), tuple(a.shape), xp)]


class SqrtOp(OpNode):
    @classmethod
    def create(cls, a):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        out_val = xp.sqrt(to_raw(a.data))
        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        return cls([a], out, {"xp": xp})

    def backward(self, grad_out):
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        a = self.inputs[0]

        if Tensor is not None and isinstance(grad_out, Tensor):
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            return [_tensor_unbroadcast(grad_out * (0.5 / a_t.sqrt()), a_t.shape)]

        xp = self.metadata["xp"]
        g = xp.asarray(grad_out)
        return [unbroadcast(g * (0.5 / xp.sqrt(to_raw(a.data))), tuple(a.shape), xp)]


class AbsOp(OpNode):
    @classmethod
    def create(cls, a):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        out_val = xp.abs(to_raw(a.data))
        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        return cls([a], out, {"xp": xp})

    def backward(self, grad_out):
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None
        a = self.inputs[0]

        if Tensor is not None and isinstance(grad_out, Tensor):
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            return [_tensor_unbroadcast(grad_out * a_t.sign(), a_t.shape)]

        xp = self.metadata["xp"]
        g = xp.asarray(grad_out)
        return [unbroadcast(g * xp.sign(to_raw(a.data)), tuple(a.shape), xp)]


class SignOp(OpNode):
    @classmethod
    def create(cls, inp):
        """
        Forward: out = sign(inp)  (values in {-1, 0, 1})
        """
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        out_raw = xp.sign(to_raw(inp.data))
        out = Tensor(
            to_ap(out_raw),
            requires_grad=getattr(inp, "requires_grad", False),
            device=getattr(inp, "device", None),
        )
        return cls([inp], out, {"xp": xp, "in_shape": tuple(getattr(inp, "shape", ()))})

    def backward(self, grad_out):
        """
        Backward of sign: zero almost everywhere.
        - If grad_out is Tensor: return grad_out * 0 (Tensor) and unbroadcast to input shape.
        - Else: return raw zeros array unbroadcasted to input shape.
        """
        # lazy import Tensor
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        in_shape = tuple(self.metadata.get("in_shape", ()))
        # Tensor-mode: produce Tensor of zeros (keeps grad_fn if grad_out is Tensor)
        if Tensor is not None and isinstance(grad_out, Tensor):
            # create zero scalar Tensor compatible with grad_out (device/dtype)
            zero_scalar = _ensure_tensor_like(0.0, like_tensor=grad_out)
            gz = grad_out * zero_scalar  # yields a Tensor of zeros
            return [_tensor_unbroadcast(gz, in_shape)]

        # Legacy raw branch: return zeros array shaped like input
        # We create zeros same dtype/device as input.data
        xp = self.metadata.get("xp", __import__("numpy"))
        zeros_raw = xp.zeros_like(to_raw(self.inputs[0].data))
        zeros_unb = unbroadcast(zeros_raw, in_shape, xp)
        return [zeros_unb]


class TrigOp(OpNode):
    @classmethod
    def create(cls, a, op):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        raw = to_raw(a.data)
        if op == "sin":
            out_val = xp.sin(raw)
        elif op == "cos":
            out_val = xp.cos(raw)
        elif op == "tan":
            out_val = xp.tan(raw)
        else:
            raise ValueError("unsupported trig")
        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        return cls([a], out, {"xp": xp, "op": op})

    def backward(self, grad_out):
        # lazy import Tensor
        try:
            from .. import finfo as ctfinfo
            from ..tensor import Tensor
        except Exception:
            Tensor = None
        op = self.metadata["op"]
        a = self.inputs[0]
        if Tensor is not None and isinstance(grad_out, Tensor):
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            if op == "sin":
                ga = grad_out * a_t.cos()
            elif op == "cos":
                ga = -grad_out * a_t.sin()
            elif op == "tan":
                # avoid direct square overflow by computing 1/(cos**2) as recip**2 if needed
                cosv = a_t.cos()
                # small epsilon to avoid division by zero
                try:
                    eps = ctfinfo(cosv.dtype).eps
                except Exception:
                    eps = _ensure_tensor_like(1e-7, cosv)
                ga = grad_out * (1.0 / (cosv**2 + eps))
            return [_tensor_unbroadcast(ga, a.shape)]

        xp = self.metadata["xp"]

        g = xp.asarray(grad_out)
        raw = to_raw(a.data)
        if op == "sin":
            ga = g * xp.cos(raw)
        elif op == "cos":
            ga = -g * xp.sin(raw)
        elif op == "tan":
            # avoid direct square overflow by computing 1/(cos**2) as recip**2 if needed
            cosv = xp.cos(raw)
            # small epsilon to avoid division by zero
            eps = (
                xp.finfo(cosv.dtype).eps
                if xp.issubdtype(cosv.dtype, xp.floating)
                else 1e-7
            )
            ga = g * (1.0 / (cosv**2 + eps))
        return [unbroadcast(ga, tuple(a.shape), xp)]


# ---------- logical ops (no grad) ----------
class LogicalOp(OpNode):
    @classmethod
    def create(cls, a, b, op):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        ar = to_raw(a.data)
        br = to_raw(b.data)
        if op == "and":
            out_val = ar & br
        elif op == "or":
            out_val = ar | br
        elif op == "xor":
            out_val = ar ^ br
        else:
            raise ValueError("logical")
        out = Tensor(to_ap(out_val), requires_grad=False, device=a.device)
        return cls([a, b], out, {"xp": xp})

    def backward(self, grad_out):
        try:
            from ..tensor import zeros
        except Exception:
            Tensor = None

        if Tensor is not None and isinstance(grad_out, Tensor):
            a_t = (
                self.inputs[0]
                if isinstance(self.inputs[0], Tensor)
                else _ensure_tensor_like(
                    getattr(self.inputs[0], "data", self.inputs[0]),
                    like_tensor=grad_out,
                )
            )
            b_t = (
                self.inputs[1]
                if isinstance(self.inputs[1], Tensor)
                else _ensure_tensor_like(
                    getattr(self.inputs[1], "data", self.inputs[1]),
                    like_tensor=grad_out,
                )
            )
            return [
                zeros(a_t.shape, device=grad_out.device),
                zeros(b_t.shape, device=grad_out.device),
            ]
        xp = self.metadata["xp"]
        return [
            xp.zeros_like(to_raw(self.inputs[0].data)),
            xp.zeros_like(to_raw(self.inputs[1].data)),
        ]


# ---------- activations ----------
class ReluOp(OpNode):
    @classmethod
    def create(cls, a):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        raw = to_raw(a.data)
        out_val = xp.maximum(raw, 0)
        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        return cls([a], out, {"xp": xp})

    def backward(self, grad_out):
        # lazy import Tensor
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        a = self.inputs[0]
        if Tensor is not None and isinstance(grad_out, Tensor):
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            mask = (a_t > 0).to(dtype=grad_out.dtype)
            return [_tensor_unbroadcast(grad_out * mask, a.shape)]

        xp = self.metadata["xp"]

        raw = to_raw(a.data)
        mask = (raw > 0).astype(grad_out.dtype)
        return [unbroadcast(xp.asarray(grad_out) * mask, tuple(a.shape), xp)]


class LeakyReluOp(OpNode):
    @classmethod
    def create(cls, a, negative_slope=1e-2):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        raw = to_raw(a.data)
        out_val = xp.where(raw > 0, raw, raw * negative_slope)
        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        return cls([a], out, {"xp": xp, "neg": negative_slope})

    def backward(self, grad_out):
        # lazy import Tensor
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None
        xp = self.metadata["xp"]
        neg = self.metadata["neg"]
        a = self.inputs[0]
        if Tensor is not None and isinstance(grad_out, Tensor):
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            neg = _ensure_tensor_like(neg, like_tensor=a_t)
            mask = neg.where(a_t <= 0, 1.0)
            return [_tensor_unbroadcast(grad_out * mask, a.shape)]

        raw = to_raw(a.data)
        g = xp.asarray(grad_out)
        mask = xp.where(raw > 0, 1.0, neg)
        return [unbroadcast(g * mask, tuple(a.shape), xp)]


class SigmoidOp(OpNode):
    @classmethod
    def create(cls, a):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        raw = to_raw(a.data)
        # stable sigmoid computation: avoid exp overflow
        raw_f = _as_float_if_needed(raw, xp)
        # use elementwise stable formula
        pos_mask = raw_f >= 0
        neg_mask = ~pos_mask
        out_val = xp.empty_like(raw_f)
        # for positive x: 1 / (1 + exp(-x))
        if xp.any(pos_mask):
            xv = raw_f[pos_mask]
            out_val[pos_mask] = 1.0 / (1.0 + xp.exp(-xv))
        # for negative x: exp(x) / (1 + exp(x))
        if xp.any(neg_mask):
            xv = raw_f[neg_mask]
            ex = xp.exp(xv)
            out_val[neg_mask] = ex / (1.0 + ex)

        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        return cls([a], out, {"xp": xp})

    def backward(self, grad_out):
        # lazy import Tensor
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        if Tensor is not None and isinstance(grad_out, Tensor):
            return [
                unbroadcast(
                    grad_out * self.output * (self.output), self.inputs[0].shape
                )
            ]

        xp = self.metadata["xp"]
        a = self.inputs[0]
        # use cached output safely
        s = to_raw(self.output.data)
        g = xp.asarray(grad_out)
        return [unbroadcast(g * s * (1 - s), tuple(a.shape), xp)]


class TanhOp(OpNode):
    @classmethod
    def create(cls, a):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        x = to_raw(a.data)
        out_val = xp.tanh(x)

        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        node = cls([a], out, {"xp": xp})
        return node

    def backward(self, grad_out):
        # lazy import Tensor
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        if Tensor is not None and isinstance(grad_out, Tensor):
            dx = grad_out * (1.0 - self.output * self.output)
            return [_tensor_unbroadcast(dx, self.inputs[0].shape)]

        xp = self.metadata["xp"]
        g = xp.asarray(grad_out)
        tanh_x = to_raw(self.output.data)
        dx = g * (1.0 - tanh_x * tanh_x)

        return [unbroadcast(dx, self.inputs[0].shape, xp)]


# GELU (approx) — this op may be unused if user builds gelu from primitives.
# Keep a correct backward for direct GeluOp calls; for composed-built gelu the primitives must also be numerically consistent.
class GeluOp(OpNode):
    @classmethod
    def create(cls, a):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        x = to_raw(a.data)
        out_val = (
            0.5 * x * (1 + xp.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        )
        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        return cls([a], out, {"xp": xp})

    def backward(self, grad_out):
        try:
            from ..tensor import Tensor
            from .. import pi
        except Exception:
            Tensor = None

        if Tensor is not None and isinstance(grad_out, Tensor):
            a = (2.0 / pi).sqrt()
            b = _ensure_tensor_like(0.044715, like_tensor=grad_out)
            x_t = (
                self.inputs[0]
                if isinstance(self.inputs[0], Tensor)
                else _ensure_tensor_like(
                    getattr(self.inputs[0], "data", self.inputs[0]),
                    like_tensor=grad_out,
                )
            )
            tanh_t = (a * (x_t + b * x_t**3)).tanh()
            dt_dx = a * (1.0 + 3.0 * b * x_t * x_t)
            dx = 0.5 * (1.0 + tanh_t) + 0.5 * x_t * (1.0 - tanh_t * tanh_t) * dt_dx
            return [_tensor_unbroadcast(grad_out * dx, self.inputs[0].shape)]

        xp = self.metadata.get("xp", __import__("numpy"))
        g = xp.asarray(grad_out)
        a = xp.sqrt(2.0 / xp.pi)
        b = 0.044715
        x = to_raw(self.inputs[0].data)
        t = a * (x + b * x**3)
        tanh_t = xp.tanh(t)
        dt_dx = a * (1.0 + 3.0 * b * x * x)
        dx = 0.5 * (1.0 + tanh_t) + 0.5 * x * (1.0 - tanh_t * tanh_t) * dt_dx
        return [unbroadcast(g * dx, tuple(self.inputs[0].shape), xp)]


# ---------- softmax / log_softmax ----------
class SoftmaxOp(OpNode):
    @classmethod
    def create(cls, inp, dim=-1):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        x = to_raw(inp.data)
        shift = x - xp.max(x, axis=dim, keepdims=True)
        ex = xp.exp(shift)
        s = ex / xp.sum(ex, axis=dim, keepdims=True)
        out = Tensor(to_ap(s), requires_grad=inp.requires_grad, device=inp.device)
        return cls([inp], out, {"dim": int(dim), "xp": xp})

    def backward(self, grad_out):
        """
        Tensor-mode: grad_out is Tensor -> use Tensor ops, keep graph.
        Legacy: grad_out is raw array -> use xp (numpy/cupy).
        Formula: grad_input = soft * (grad_out - sum(grad_out * soft, axis=dim, keepdims=True))
        """
        # lazy import Tensor
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        xp = self.metadata["xp"]
        dim = int(self.metadata["dim"])
        a = self.inputs[0]

        # Tensor-mode
        if Tensor is not None and isinstance(grad_out, Tensor):
            g = grad_out
            # try to use cached output Tensor
            soft_t = self.output if isinstance(self.output, Tensor) else None
            if soft_t is None:
                # fallback: compute soft from input tensor
                inp = (
                    a
                    if isinstance(a, Tensor)
                    else _ensure_tensor_like(getattr(a, "data", a), like_tensor=g)
                )
                shift = inp - inp.max(dim=dim, keepdim=True)
                ex = shift.exp()
                denom = ex.sum(dim=dim, keepdims=True)
                soft_t = ex / denom
            # compute dot = sum(g * soft, axis=dim, keepdims=True)
            dot = (g * soft_t).sum(dim=dim, keepdims=True)
            ga = soft_t * (g - dot)
            # unbroadcast to input shape
            ga = _tensor_unbroadcast(ga, tuple(a.shape))
            return [ga]

        # Legacy raw branch (unchanged numeric behavior)
        g_raw = xp.asarray(grad_out)
        soft_raw = to_raw(self.output.data)
        dot = xp.sum(g_raw * soft_raw, axis=dim, keepdims=True)
        ga_raw = soft_raw * (g_raw - dot)
        return [unbroadcast(ga_raw, tuple(a.shape), xp)]


class LogSoftmaxOp(OpNode):
    @classmethod
    def create(cls, inp, dim=-1):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        x = to_raw(inp.data)
        shift = x - xp.max(x, axis=dim, keepdims=True)
        logsum = xp.log(xp.sum(xp.exp(shift), axis=dim, keepdims=True))
        out_val = shift - logsum
        out = Tensor(to_ap(out_val), requires_grad=inp.requires_grad, device=inp.device)
        return cls([inp], out, {"dim": int(dim), "xp": xp})

    def backward(self, grad_out):
        """
        Tensor-mode: grad_out is Tensor -> use Tensor ops and keep graph.
        Legacy: use raw xp arrays.
        Formula: for logsoftmax l = x - logsumexp(x),
          soft = exp(l), ga = grad_out - soft * sum(grad_out, axis=dim, keepdims=True)
        """
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        xp = self.metadata["xp"]
        dim = int(self.metadata["dim"])
        a = self.inputs[0]

        if Tensor is not None and isinstance(grad_out, Tensor):
            g = grad_out
            logsoft_t = self.output if isinstance(self.output, Tensor) else None
            if logsoft_t is None:
                # reconstruct logsoft from input tensor
                inp = (
                    a
                    if isinstance(a, Tensor)
                    else _ensure_tensor_like(getattr(a, "data", a), like_tensor=g)
                )

                shift = inp - inp.max(dim=dim, keepdim=True)
                logsum = (shift.exp()).sum(dim=dim, keepdims=True).log()
                logsoft_t = shift - logsum
            soft_t = logsoft_t.exp()

            dot = g.sum(axis=dim, keepdims=True)
            ga = g - soft_t * dot
            ga = _tensor_unbroadcast(ga, tuple(a.shape))
            return [ga]

        # legacy raw branch
        g_raw = xp.asarray(grad_out)
        logsoft_raw = to_raw(self.output.data)
        soft_raw = xp.exp(logsoft_raw)
        dot = xp.sum(g_raw, axis=dim, keepdims=True)
        ga_raw = g_raw - soft_raw * dot
        return [unbroadcast(ga_raw, tuple(a.shape), xp)]


# ---------- matmul / linear ----------


class MatMulOp(OpNode):
    @classmethod
    def create(
        cls,
        a,
        b,
        left_indices=None,
        right_indices=None,
        out_indices=None,
    ):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data)
        out_val = xp.matmul(a_raw, b_raw)
        out_tensor = Tensor(
            to_ap(out_val),
            requires_grad=(
                getattr(a, "requires_grad", False) or getattr(b, "requires_grad", False)
            ),
            device=a.device,
        )
        out_tensor.name = f"matmul({getattr(a, 'name', '')},{getattr(b, 'name', '')})"
        node = cls(
            inputs=[a, b],
            output=out_tensor,
            metadata={
                "a_shape": tuple(getattr(a, "shape", ())),
                "b_shape": tuple(getattr(b, "shape", ())),
                "left_indices": left_indices,
                "right_indices": right_indices,
                "out_indices": out_indices,
                "xp": xp,
            },
        )
        try:
            out_tensor.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out):
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        a, b = self.inputs
        xp = self.metadata.get("xp", np)
        if Tensor is not None and isinstance(grad_out, Tensor):
            # Wrap inputs as Tensors if needed so arithmetic builds graph
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            b_t = (
                b
                if isinstance(b, Tensor)
                else _ensure_tensor_like(getattr(b, "data", b), like_tensor=grad_out)
            )
            ga = grad_out.matmul(b_t.transpose(-1, -2))
            gb = a_t.transpose(-1, -2).matmul(grad_out)
            ga = _tensor_unbroadcast(ga, a_t.shape)
            gb = _tensor_unbroadcast(gb, b_t.shape)
            return [ga, gb]

        g = xp.asarray(grad_out)
        b_raw = to_raw(b.data)
        a_raw = to_raw(a.data)
        ga = xp.matmul(g, xp.swapaxes(b_raw, -1, -2))
        gb = xp.matmul(xp.swapaxes(a_raw, -1, -2), g)
        ga = unbroadcast(ga, tuple(getattr(a, "shape", ())), xp)
        gb = unbroadcast(gb, tuple(getattr(b, "shape", ())), xp)
        return [ga, gb]


# ---------- reductions ----------
class ReduceSumOp(OpNode):
    @classmethod
    def create(
        cls,
        inp,
        dim: Tuple[int, ...],
        keepdims: bool = False,
        indices_in=None,
        indices_out=None,
    ):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        a_raw = to_raw(inp.data)
        out_val = a_raw.sum(dim, keepdims=keepdims)
        out_tensor = Tensor(
            to_ap(out_val), requires_grad=getattr(inp, "requires_grad", False)
        )
        out_tensor.name = f"{getattr(inp, 'name', 'T')}_sum"
        node = cls(
            inputs=[inp],
            output=out_tensor,
            metadata={
                "dim": dim,
                "in_shape": tuple(getattr(inp, "shape", ())),
                "keepdims": keepdims,
                "indices": {"in": indices_in, "out": indices_out},
                "xp": xp,
            },
        )
        try:
            out_tensor.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out):
        from ..tensor import Tensor

        xp = self.metadata.get("xp", __import__("numpy"))
        dim = self.metadata.get("dim", None)
        in_shape = tuple(self.metadata.get("in_shape", ()))
        keepdims = self.metadata.get("keepdims", False)

        if isinstance(grad_out, Tensor):
            g = grad_out
            if (not keepdims) and dim is not None:
                dim_iter = (dim,) if isinstance(dim, int) else tuple(dim)
                for ax in sorted(dim_iter):
                    g = g.unsqueeze(ax)
            # broadcast to in_shape using helper
            g_expanded = _tensor_broadcast_to(g, in_shape)
            return [g_expanded]

        # legacy raw path
        g_raw = xp.asarray(grad_out)
        if not keepdims and dim is not None:
            dim_iter = (dim,) if isinstance(dim, int) else tuple(dim)
            for ax in sorted(dim_iter):
                g_raw = xp.expand_dims(g_raw, axis=ax)
        g_raw = xp.broadcast_to(g_raw, in_shape)
        return [g_raw]


class ReduceMeanOp(OpNode):
    @classmethod
    def create(cls, inp, dim=None, keepdims=False):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        out_val = to_raw(inp.data).mean(dim, keepdims=keepdims)
        out = Tensor(to_ap(out_val), requires_grad=inp.requires_grad, device=inp.device)
        if dim is None:
            N = int(xp.prod(inp.shape))
        else:
            axes_iter = (dim,) if isinstance(dim, int) else tuple(dim)
            N = 1
            for ax in axes_iter:
                N *= inp.shape[ax]
        return cls(
            [inp],
            out,
            {
                "dim": dim,
                "in_shape": tuple(inp.shape),
                "keepdims": keepdims,
                "N": N,
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        from ..tensor import Tensor

        xp = self.metadata["xp"]
        dim = self.metadata["dim"]
        in_shape = self.metadata["in_shape"]
        N = float(self.metadata["N"])
        keep = self.metadata["keepdims"]

        if isinstance(grad_out, Tensor):
            g = grad_out
            if not keep and dim is not None:
                dim_iter = (dim,) if isinstance(dim, int) else tuple(dim)
                for ax in sorted(dim_iter):
                    g = g.unsqueeze(ax)
            g_expanded = _tensor_broadcast_to(g, in_shape)
            return [g_expanded / N]

        # legacy raw
        g_raw = xp.asarray(grad_out)
        if not keep and dim is not None:
            dim_iter = (dim,) if isinstance(dim, int) else tuple(dim)
            for ax in sorted(dim_iter):
                g_raw = xp.expand_dims(g_raw, axis=ax)
        return [xp.broadcast_to(g_raw, in_shape) / N]


class ReduceVarOp(OpNode):
    @classmethod
    def create(cls, inp, dim=None, correction=None, keepdims=False):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        a = to_raw(inp.data)
        if dim is None:
            dim_use = tuple(range(inp.ndim))
        else:
            dim_use = dim if isinstance(dim, tuple) else (dim,)
        mean_vals = a.mean(dim_use, keepdims=True)
        centered = a - mean_vals
        sumsq = (centered**2).sum(dim_use, keepdims=True)
        N = 1
        for d in dim_use:
            N *= inp.shape[d]
        corr = 1.0 if correction is None else float(correction)
        denom = float(N) - corr
        var = sumsq / denom
        if not keepdims:
            for ax in sorted(dim_use, reverse=True):
                var = xp.squeeze(var, axis=ax)
        out = Tensor(to_ap(var), requires_grad=inp.requires_grad, device=inp.device)
        return cls(
            [inp],
            out,
            {
                "dim": dim_use,
                "in_shape": tuple(inp.shape),
                "mean_vals": mean_vals,
                "denom": denom,
                "keepdims": keepdims,
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        from ..tensor import Tensor

        xp = self.metadata["xp"]
        dim = self.metadata["dim"]
        denom = float(self.metadata["denom"])
        in_shape = tuple(self.metadata["in_shape"])
        keepdims = bool(self.metadata.get("keepdims", False))
        a = self.inputs[0]

        if isinstance(grad_out, Tensor):
            g = grad_out
            # ensure mean_vals computed via Tensor
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=g)
            )
            dim_iter = dim
            mean_vals_t = a_t.mean(dim_iter, keepdims=True)
            # reshape g
            if not keepdims:
                dim_iter_local = (dim,) if isinstance(dim, int) else tuple(dim)
                for ax in sorted(dim_iter_local):
                    g = g.unsqueeze(ax)
            upstream = _tensor_broadcast_to(g, in_shape)
            self_grad = (2.0 / denom) * (a_t - mean_vals_t) * upstream
            return [_tensor_unbroadcast(self_grad, in_shape)]

        # legacy raw path
        g_raw = xp.asarray(grad_out)
        a_raw = to_raw(self.inputs[0].data)
        mean_vals = self.metadata["mean_vals"]
        grad_shape = list(self.inputs[0].shape)
        for d in dim:
            grad_shape[d] = 1
        g_reshaped = xp.reshape(g_raw, tuple(grad_shape))
        upstream = xp.broadcast_to(g_reshaped, self.inputs[0].shape)
        self_grad = (2.0 / denom) * (a_raw - mean_vals) * upstream
        return [unbroadcast(self_grad, tuple(self.inputs[0].shape), xp)]


class ReduceMaxOp(OpNode):
    @classmethod
    def create(cls, inp, dim=None, keepdims=False):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        out_val = to_raw(inp.data).max(dim, keepdims=keepdims)
        out = Tensor(to_ap(out_val), requires_grad=inp.requires_grad, device=inp.device)
        return cls(
            [inp],
            out,
            {
                "dim": dim,
                "in_shape": tuple(inp.shape),
                "keepdims": keepdims,
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        from ..tensor import Tensor

        xp = self.metadata["xp"]
        dim = self.metadata["dim"]
        in_shape = self.metadata["in_shape"]
        keep = self.metadata["keepdims"]
        a = self.inputs[0]

        if isinstance(grad_out, Tensor):
            g = grad_out
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=g)
            )
            # compute max with keepdims=True
            dim_iter = dim
            max_vals = a_t.max(dim_iter, keepdims=True)
            mask = (a_t == max_vals).astype(getattr(g, "dtype", None))
            count = mask.sum(dim_iter, keepdims=True)
            # prepare g_expanded
            if not keep:
                if dim is None:
                    nd = len(in_shape)
                    for _ in range(nd):
                        g = g.unsqueeze(0)
                else:
                    dim_iter_local = (dim,) if isinstance(dim, int) else tuple(dim)
                    for ax in sorted(dim_iter_local):
                        g = g.unsqueeze(ax)
            g_exp = _tensor_broadcast_to(g, in_shape)
            g_over_count = g_exp / count
            ga = mask * g_over_count
            return [_tensor_unbroadcast(ga, in_shape)]

        # legacy raw
        g_raw = xp.asarray(grad_out)
        a_raw = to_raw(self.inputs[0].data)
        if dim is None:
            maxv = a_raw.max()
            mask = (a_raw == maxv).astype(g_raw.dtype)
            cnt = mask.sum()
            ga = mask * (g_raw / cnt)
            return [unbroadcast(ga, tuple(self.inputs[0].shape), xp)]
        max_vals = a_raw.max(dim, keepdims=True)
        mask = (a_raw == max_vals).astype(g_raw.dtype)
        cnt = mask.sum(dim, keepdims=True)
        if not keep:
            if isinstance(dim, int):
                dim = (dim,)
            for ax in sorted(dim):
                g_raw = xp.expand_dims(g_raw, axis=ax)
        g_exp = xp.broadcast_to(g_raw, in_shape)
        ga = mask * (g_exp / cnt)
        return [unbroadcast(ga, tuple(self.inputs[0].shape), xp)]


class ReduceMinOp(OpNode):
    @classmethod
    def create(cls, inp, dim=None, keepdims=False):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        out_val = to_raw(inp.data).min(dim, keepdims=keepdims)
        out = Tensor(to_ap(out_val), requires_grad=inp.requires_grad, device=inp.device)
        return cls(
            [inp],
            out,
            {
                "dim": dim,
                "in_shape": tuple(inp.shape),
                "keepdims": keepdims,
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        from ..tensor import Tensor

        xp = self.metadata["xp"]
        dim = self.metadata["dim"]
        in_shape = self.metadata["in_shape"]
        keep = self.metadata["keepdims"]
        a = self.inputs[0]

        if isinstance(grad_out, Tensor):
            g = grad_out
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=g)
            )
            dim_iter = dim
            min_vals = a_t.min(dim_iter, keepdims=True)
            mask = (a_t == min_vals).astype(getattr(g, "dtype", None))
            cnt = mask.sum(dim_iter, keepdims=True)
            if not keep:
                if isinstance(dim, int):
                    dim = (dim,)
                for ax in sorted(dim):
                    g = g.unsqueeze(ax)
            g_exp = _tensor_broadcast_to(g, in_shape)
            ga = mask * (g_exp / cnt)
            return [_tensor_unbroadcast(ga, in_shape)]

        # legacy raw
        g_raw = xp.asarray(grad_out)
        a_raw = to_raw(self.inputs[0].data)
        if dim is None:
            minv = a_raw.min()
            mask = (a_raw == minv).astype(g_raw.dtype)
            cnt = mask.sum()
            ga = mask * (g_raw / cnt)
            return [unbroadcast(ga, tuple(self.inputs[0].shape), xp)]
        min_vals = a_raw.min(dim, keepdims=True)
        mask = (a_raw == min_vals).astype(g_raw.dtype)
        cnt = mask.sum(dim, keepdims=True)
        if not keep:
            if isinstance(dim, int):
                dim = (dim,)
            for ax in sorted(dim):
                g_raw = xp.expand_dims(g_raw, ax)
        g_exp = xp.broadcast_to(g_raw, in_shape)
        ga = mask * (g_exp / cnt)
        return [unbroadcast(ga, tuple(self.inputs[0].shape), xp)]


class ReduceProdOp(OpNode):
    @classmethod
    def create(cls, inp, dim=None, keepdims=False):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        out_val = to_raw(inp.data).prod(dim, keepdims=keepdims)
        out = Tensor(to_ap(out_val), requires_grad=inp.requires_grad, device=inp.device)
        return cls(
            [inp],
            out,
            {
                "dim": dim,
                "in_shape": tuple(inp.shape),
                "keepdims": keepdims,
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        """
        Exact backward handling zeros:
        For each element a_k in the reduced slice:
          - if count_zero == 0: grad = g * (prod_nozeros / a_k)
          - if count_zero == 1: grad = g * prod_nozeros if a_k == 0 else 0
          - if count_zero >=2: grad = 0
        """
        # lazy import Tensor
        from ..tensor import Tensor

        xp = self.metadata["xp"]
        dim = self.metadata["dim"]
        in_shape = tuple(self.metadata["in_shape"])
        keepdims = bool(self.metadata["keepdims"])
        a_input = self.inputs[0]

        # ---------- tensor-mode ----------
        if isinstance(grad_out, Tensor):
            g = grad_out  # Tensor
            # ensure a_t is Tensor
            a_t = (
                a_input
                if isinstance(a_input, Tensor)
                else _ensure_tensor_like(
                    getattr(a_input, "data", a_input), like_tensor=g
                )
            )

            # normalize axes for reductions: keepdims=True for convenience
            # compute mask_zero (boolean), numeric mask, and count_zero (keepdims=True)
            mask_zero = (a_t == _ensure_tensor_like(0.0, like_tensor=a_t)).astype(
                getattr(a_t, "dtype", None)
            )
            count_zero = mask_zero.sum(dim, keepdims=True)

            # a_nozero where zeros replaced by 1
            one_t = _ensure_tensor_like(1.0, like_tensor=a_t)
            mask_zero_num = mask_zero
            a_nozero = mask_zero_num * one_t + (1 - mask_zero_num) * a_t

            # product of non-zero-replaced array
            prod_nozeros = a_nozero.prod(dim, keepdims=True)

            # prepare grad g: unsqueeze axes if needed (when keepdims=False)
            if not keepdims and dim is not None:
                dim_iter = (dim,) if isinstance(dim, int) else tuple(dim)
                for ax in sorted(dim_iter):
                    g = g.unsqueeze(ax)

            # broadcast g to input shape
            g_b = _tensor_broadcast_to(g, in_shape)

            # numeric masks: count_zero == 0 or == 1
            zero_count_eq0 = (count_zero == 0).astype(getattr(a_t, "dtype", None))
            zero_count_eq1 = (count_zero == 1).astype(getattr(a_t, "dtype", None))

            # prod_div: valid when count_zero == 0 (no zeros in slice)
            prod_div = (
                prod_nozeros / a_t
            )  # safe because where count_zero==0 there is no zero in a_t

            # prod_from_zero: for positions where a_k==0 when exactly one zero
            prod_from_zero = mask_zero_num * prod_nozeros

            # combine:
            prod_others = zero_count_eq0 * prod_div + zero_count_eq1 * prod_from_zero

            ga = g_b * prod_others

            # unbroadcast to original input shape
            return [_tensor_unbroadcast(ga, in_shape)]

        # ---------- legacy raw ----------
        g_raw = xp.asarray(grad_out)
        a_raw = to_raw(self.inputs[0].data)

        # axes normalization: use keepdims=True for intermediate arrays
        dim_arg = dim

        # compute mask_zero and count_zero with keepdims=True
        mask_zero_raw = (a_raw == 0).astype(a_raw.dtype)
        count_zero_raw = mask_zero_raw.sum(dim_arg, keepdims=True)

        # replace zeros by 1 to compute product of non-zero entries
        a_nozero_raw = xp.where(mask_zero_raw != 0, xp.ones_like(a_raw), a_raw)
        prod_nozeros_raw = xp.prod(a_nozero_raw, dim_arg, keepdims=True)

        # prepare g_raw expanded if necessary
        if not keepdims and dim is not None:
            dim_iter = (dim,) if isinstance(dim, int) else tuple(dim)
            for ax in sorted(dim_iter):
                g_raw = xp.expand_dims(g_raw, axis=ax)
        g_b_raw = xp.broadcast_to(g_raw, in_shape)

        # masks numeric
        zero_count_eq0_raw = (count_zero_raw == 0).astype(a_raw.dtype)
        zero_count_eq1_raw = (count_zero_raw == 1).astype(a_raw.dtype)

        # prod_div raw (safe under zero_count==0)
        prod_div_raw = prod_nozeros_raw / a_raw

        prod_from_zero_raw = mask_zero_raw * prod_nozeros_raw

        prod_others_raw = (
            zero_count_eq0_raw * prod_div_raw + zero_count_eq1_raw * prod_from_zero_raw
        )

        ga_raw = g_b_raw * prod_others_raw

        return [unbroadcast(ga_raw, tuple(self.inputs[0].shape), xp)]


class MaximumOp(OpNode):
    """
    Elementwise maximum: out = maximum(a, b)
    - Tensor-mode (create_graph=True): uses Tensor comparisons (>, <, ==) and arithmetic masks
      so backward returns Tensor and preserves graph as much as possible.
    - Legacy raw-mode: uses xp.maximum and numeric masks.
    Tie handling: split ties 50/50.
    """

    @classmethod
    def create(cls, a, b):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        # cast rhs to lhs dtype when appropriate
        b_raw = _maybe_cast_rhs_to_lhs_dtype(a_raw, b_raw, xp)

        out_val = xp.maximum(a_raw, b_raw)
        out = Tensor(
            to_ap(out_val),
            requires_grad=(
                getattr(a, "requires_grad", False) or getattr(b, "requires_grad", False)
            ),
            device=getattr(a, "device", None),
        )
        return cls(
            [a, b],
            out,
            {
                "a_shape": tuple(getattr(a, "shape", ())),
                "b_shape": tuple(getattr(b, "shape", ())),
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        # lazy import Tensor
        from ..tensor import Tensor

        a, b = self.inputs
        if isinstance(grad_out, Tensor):
            # --- TENSOR branch (create_graph=True) ---
            g = grad_out  # Tensor
            # ensure Tensor wrappers for a and b (wrap raw inputs into Tensors without grad if needed)
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=g)
            )
            b_t = (
                b
                if isinstance(b, Tensor)
                else _ensure_tensor_like(getattr(b, "data", b), like_tensor=g)
            )

            # build numeric masks via comparisons then cast to gradient dtype
            mdtype = getattr(g, "dtype", None)
            gt = (a_t > b_t).astype(mdtype)
            lt = (a_t < b_t).astype(mdtype)
            eq = (a_t == b_t).astype(mdtype)

            # masks distributing ties equally
            mask_a = gt + 0.5 * eq
            mask_b = lt + 0.5 * eq

            ga = g * mask_a
            gb = g * mask_b

            # unbroadcast back to original input shapes (tensor helper)
            ga = _tensor_unbroadcast(ga, tuple(getattr(a, "shape", ())))
            gb = _tensor_unbroadcast(gb, tuple(getattr(b, "shape", ())))
            return [ga, gb]

        xp = self.metadata["xp"]
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        g_raw = xp.asarray(grad_out)
        gt = (a_raw > b_raw).astype(g_raw.dtype)
        lt = (a_raw < b_raw).astype(g_raw.dtype)
        eq = (a_raw == b_raw).astype(g_raw.dtype)
        mask_a = gt + 0.5 * eq
        mask_b = lt + 0.5 * eq
        ga = unbroadcast(g_raw * mask_a, tuple(getattr(a, "shape", ())), xp)
        gb = unbroadcast(g_raw * mask_b, tuple(getattr(b, "shape", ())), xp)
        return [ga, gb]


class MinimumOp(OpNode):
    """
    Elementwise minimum: out = minimum(a, b)
    Implementation mirrors MaximumOp but comparisons flipped.
    """

    @classmethod
    def create(cls, a, b):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        b_raw = _maybe_cast_rhs_to_lhs_dtype(a_raw, b_raw, xp)

        out_val = xp.minimum(a_raw, b_raw)
        out = Tensor(
            to_ap(out_val),
            requires_grad=(
                getattr(a, "requires_grad", False) or getattr(b, "requires_grad", False)
            ),
            device=getattr(a, "device", None),
        )
        return cls(
            [a, b],
            out,
            {
                "a_shape": tuple(getattr(a, "shape", ())),
                "b_shape": tuple(getattr(b, "shape", ())),
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        from ..tensor import Tensor

        a, b = self.inputs

        if isinstance(grad_out, Tensor):
            # --- TENSOR branch ---
            g = grad_out
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=g)
            )
            b_t = (
                b
                if isinstance(b, Tensor)
                else _ensure_tensor_like(getattr(b, "data", b), like_tensor=g)
            )

            mdtype = getattr(g, "dtype", None)
            lt = (a_t < b_t).astype(mdtype)
            gt = (a_t > b_t).astype(mdtype)
            eq = (a_t == b_t).astype(mdtype)

            mask_a = lt + 0.5 * eq
            mask_b = gt + 0.5 * eq

            ga = g * mask_a
            gb = g * mask_b

            ga = _tensor_unbroadcast(ga, tuple(getattr(a, "shape", ())))
            gb = _tensor_unbroadcast(gb, tuple(getattr(b, "shape", ())))
            return [ga, gb]

        xp = self.metadata["xp"]
        a_raw = to_raw(a.data)
        b_raw = to_raw(b.data) if hasattr(b, "data") else to_raw(b)
        g_raw = xp.asarray(grad_out)
        lt = (a_raw < b_raw).astype(g_raw.dtype)
        gt = (a_raw > b_raw).astype(g_raw.dtype)
        eq = (a_raw == b_raw).astype(g_raw.dtype)
        mask_a = lt + 0.5 * eq
        mask_b = gt + 0.5 * eq
        ga = unbroadcast(g_raw * mask_a, tuple(getattr(a, "shape", ())), xp)
        gb = unbroadcast(g_raw * mask_b, tuple(getattr(b, "shape", ())), xp)
        return [ga, gb]


# ---------- argmin/argmax (non-diff) ----------
class ArgMaxOp(OpNode):
    @classmethod
    def create(cls, inp, axis=None):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        out_val = to_raw(inp.data).argmax(axis=axis)
        out = Tensor(to_ap(out_val), requires_grad=False, device=inp.device)
        return cls([inp], out, {"axis": axis, "xp": xp})

    def backward(self, grad_out):
        from ..tensor import Tensor, zeros_like

        a = self.inputs[0]
        if isinstance(grad_out, Tensor):
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            return [zeros_like(a_t)]

        xp = self.metadata["xp"]
        return [xp.zeros_like(to_raw(a.data))]


class ArgMinOp(OpNode):
    @classmethod
    def create(cls, inp, axis=None):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        out_val = to_raw(inp.data).argmin(axis=axis)
        out = Tensor(to_ap(out_val), requires_grad=False, device=inp.device)
        return cls([inp], out, {"axis": axis, "xp": xp})

    def backward(self, grad_out):
        from ..tensor import Tensor, zeros_like

        a = self.inputs[0]
        if isinstance(grad_out, Tensor):
            a_t = (
                a
                if isinstance(a, Tensor)
                else _ensure_tensor_like(getattr(a, "data", a), like_tensor=grad_out)
            )
            return [zeros_like(a_t)]

        xp = self.metadata["xp"]
        a_raw = to_raw(a.data)
        return [xp.zeros_like(a_raw)]


class FloorOp(OpNode):
    @classmethod
    def create(cls, inp):
        """
        Forward: out = floor(inp)
        """
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        a_raw = to_raw(inp.data)
        out_raw = xp.floor(a_raw)
        out = Tensor(
            to_ap(out_raw),
            requires_grad=getattr(inp, "requires_grad", False),
            device=getattr(inp, "device", None),
        )
        return cls([inp], out, {"xp": xp, "in_shape": tuple(getattr(inp, "shape", ()))})

    def backward(self, grad_out):
        """
        Backward for floor:
        - tensor-mode: return grad_out * 0 (a Tensor) and unbroadcast to input shape
        - legacy raw: return zeros array (raw) and unbroadcast to input shape
        """
        # lazy import Tensor
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        xp = self.metadata.get("xp", __import__("numpy"))
        in_shape = tuple(self.metadata.get("in_shape", ()))
        inp = self.inputs[0]

        # Tensor-mode: return grad_out * 0 (keeps a Tensor, but value zero)
        if Tensor is not None and isinstance(grad_out, Tensor):
            # create zero scalar as Tensor-like for device/dtype
            zero_scalar = _ensure_tensor_like(0.0, like_tensor=grad_out)
            gz = grad_out * zero_scalar
            try:
                gz = _tensor_unbroadcast(gz, in_shape)
            except Exception:
                try:
                    gz = gz.reshape(in_shape)
                except Exception:
                    pass
            return [gz]

        # Legacy raw branch: zeros array same shape as input
        zeros_raw = xp.zeros_like(to_raw(inp.data))
        zeros_unb = unbroadcast(zeros_raw, in_shape, xp)
        return [zeros_unb]


# ---------- concat / stack ----------
class ConcatOp(OpNode):
    @classmethod
    def create(cls, tensors, dim=0):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(tensors[0])
        # normalize dim relative to first tensor
        in_ndim = len(getattr(tensors[0], "shape", ()))
        if dim < 0:
            dim_norm = dim + in_ndim
        else:
            dim_norm = dim

        arrs = [to_raw(t.data) for t in tensors]
        out_val = xp.concatenate(arrs, axis=dim_norm)
        out = Tensor(
            to_ap(out_val),
            requires_grad=any(getattr(t, "requires_grad", False) for t in tensors),
            device=getattr(tensors[0], "device", None),
        )
        sizes = [int(t.shape[dim_norm]) for t in tensors]
        in_shapes = [tuple(getattr(t, "shape", ())) for t in tensors]
        return cls(
            list(tensors),
            out,
            {"dim": int(dim_norm), "sizes": sizes, "in_shapes": in_shapes, "xp": xp},
        )

    def backward(self, grad_out):
        dim = int(self.metadata["dim"])
        sizes = list(self.metadata["sizes"])
        in_shapes = list(self.metadata["in_shapes"])

        from ..tensor import Tensor

        # Tensor-mode branch
        if isinstance(grad_out, Tensor):
            g = grad_out
            grads = []
            offset = 0
            for s, inp_shape, inp in zip(sizes, in_shapes, self.inputs):
                # slice grad tensor
                idx = [slice(None)] * len(getattr(g, "shape", ()))
                idx[dim] = slice(offset, offset + s)
                chunk = g[tuple(idx)]
                # unbroadcast chunk -> match original input shape
                chunk_unb = _tensor_unbroadcast(chunk, tuple(inp_shape))
                if getattr(inp, "requires_grad", False):
                    grads.append(chunk_unb)
                else:
                    # create zero Tensor same device/dtype as g and shape = inp_shape
                    zeros_arr = _ap.Array.zeros(tuple(inp_shape))
                    zero_t = _ensure_tensor_like(zeros_arr, like_tensor=g)
                    grads.append(zero_t)
                offset += s
            return grads

        # Legacy raw branch
        xp = self.metadata["xp"]
        g_raw = xp.asarray(grad_out)
        grads = []
        offset = 0
        for s, inp in zip(sizes, self.inputs):
            idx = [slice(None)] * g_raw.ndim
            idx[dim] = slice(offset, offset + s)
            chunk = g_raw[tuple(idx)]
            grads.append(unbroadcast(chunk, tuple(getattr(inp, "shape", ())), xp))
            offset += s
        return grads


class StackOp(OpNode):
    @classmethod
    def create(cls, tensors, dim=0):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(tensors[0])
        in_ndim = len(getattr(tensors[0], "shape", ()))
        out_ndim = in_ndim + 1
        if dim < 0:
            dim_norm = dim + out_ndim
        else:
            dim_norm = dim

        arrs = [xp.expand_dims(to_raw(t.data), axis=dim_norm) for t in tensors]
        out_val = xp.concatenate(arrs, axis=dim_norm)
        out = Tensor(
            to_ap(out_val),
            requires_grad=any(getattr(t, "requires_grad", False) for t in tensors),
            device=getattr(tensors[0], "device", None),
        )
        sizes = [1] * len(tensors)
        in_shapes = [tuple(getattr(t, "shape", ())) for t in tensors]
        return cls(
            list(tensors),
            out,
            {"dim": int(dim_norm), "sizes": sizes, "in_shapes": in_shapes, "xp": xp},
        )

    def backward(self, grad_out):
        dim = int(self.metadata["dim"])
        in_shapes = list(self.metadata["in_shapes"])

        from ..tensor import Tensor

        # Tensor-mode branch
        if isinstance(grad_out, Tensor):
            g = grad_out
            grads = []
            for i, inp in enumerate(self.inputs):
                idx = [slice(None)] * len(getattr(g, "shape", ()))
                idx[dim] = slice(i, i + 1)
                chunk = g[tuple(idx)]
                # squeeze stacked axis
                chunk_squeezed = chunk.squeeze(dim=dim)
                out_grad = _tensor_unbroadcast(chunk_squeezed, tuple(in_shapes[i]))
                if getattr(inp, "requires_grad", False):
                    grads.append(out_grad)
                else:
                    zeros_arr = _ap.Array.zeros(tuple(in_shapes[i]))
                    zero_t = _ensure_tensor_like(zeros_arr, like_tensor=g)
                    grads.append(zero_t)
            return grads

        # Legacy raw branch
        xp = self.metadata["xp"]
        g_raw = xp.asarray(grad_out)
        grads = []
        for i, inp in enumerate(self.inputs):
            idx = [slice(None)] * g_raw.ndim
            idx[dim] = slice(i, i + 1)
            chunk = g_raw[tuple(idx)]
            chunk = xp.squeeze(chunk, axis=dim)
            grads.append(unbroadcast(chunk, tuple(getattr(inp, "shape", ())), xp))
        return grads


# ---------- cumsum / repeat_interleave ----------
class CumsumOp(OpNode):
    @classmethod
    def create(cls, inp, dim=0):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        out_val = xp.cumsum(to_raw(inp.data), axis=dim)
        out = Tensor(
            to_ap(out_val),
            requires_grad=getattr(inp, "requires_grad", False),
            device=getattr(inp, "device", None),
        )
        return cls(
            [inp],
            out,
            {"dim": int(dim), "in_shape": tuple(getattr(inp, "shape", ())), "xp": xp},
        )

    def backward(self, grad_out):
        """
        grad_input = reverse( cumsum( reverse(grad_out, axis=dim), axis=dim) )
        Tensor-mode: implement via mask-matmul so graph is preserved.
        """
        xp = self.metadata["xp"]
        dim = int(self.metadata["dim"])
        in_shape = tuple(self.metadata["in_shape"])

        # lazy import Tensor
        from ..tensor import Tensor

        if isinstance(grad_out, Tensor):
            # tensor-mode branch: build lower-triangular mask and matmul
            g_t = grad_out  # Tensor
            rank = len(getattr(g_t, "shape", ()))
            # normalize negative dim
            if dim < 0:
                dim_norm = dim + rank
            else:
                dim_norm = dim

            # bring target axis to front
            g_perm = g_t.transpose(
                tuple([dim_norm] + [i for i in range(rank) if i != dim_norm])
            )
            # g_perm shape: (N, *rest)
            N = int(getattr(g_perm, "shape", ())[0])

            # flatten trailing dims to produce 2D (N, K)
            rest_shape = tuple(getattr(g_perm, "shape", ())[1:])
            K = 1
            for s in rest_shape:
                K *= int(s)
            if K == 0:
                # empty tensor
                out_grad = g_t * 0
                out_grad = _tensor_unbroadcast(out_grad, in_shape)
                return [out_grad]

            g_2d = g_perm.reshape((N, K))  # Tensor (N, K)

            # build lower-triangular mask L where L[i,j] = 1 if j >= i (for reverse-cumsum behavior)
            # Create index vectors as Tensor (constants) so comparisons are Tensor ops (part of graph)
            xp_backend = self.metadata["xp"]
            idx_i = _ensure_tensor_like(xp_backend.arange(N), like_tensor=g_t).reshape(
                (N, 1)
            )  # (N,1)
            idx_j = _ensure_tensor_like(xp_backend.arange(N), like_tensor=g_t).reshape(
                (1, N)
            )  # (1,N)
            # mask M shape (N, N): M[i,j] = 1 if j >= i
            M = (idx_j >= idx_i).astype(getattr(g_t, "dtype", None))  # (N,N) Tensor

            # matmul: (N,N) @ (N,K) -> (N,K)
            mat = M.matmul(g_2d)

            # reshape back to (N, *rest_shape)
            mat_rs = mat.reshape((N,) + rest_shape)

            # permute back to original axis order
            # inverse perm:
            inv_perm = [0] * rank
            perm = [dim_norm] + [i for i in range(rank) if i != dim_norm]
            for i, p in enumerate(perm):
                inv_perm[p] = i
            out_perm = mat_rs.transpose(tuple(inv_perm))

            # ensure shape matches input and unbroadcast if necessary
            out_final = _tensor_unbroadcast(out_perm, in_shape)
            return [out_final]

        g = xp.asarray(grad_out)
        grad_reversed = xp.flip(g, axis=dim)
        grad_input = xp.cumsum(grad_reversed, axis=dim)
        grad_input = xp.flip(grad_input, axis=dim)
        return [unbroadcast(grad_input, tuple(self.inputs[0].shape), xp)]


class RepeatInterleaveOp(OpNode):
    @classmethod
    def create(cls, inp, repeats, dim=0):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(inp)
        out_val = xp.repeat(to_raw(inp.data), repeats, axis=dim)
        out = Tensor(
            to_ap(out_val),
            requires_grad=getattr(inp, "requires_grad", False),
            device=getattr(inp, "device", None),
        )
        return cls(
            [inp],
            out,
            {
                "repeats": repeats,
                "dim": int(dim),
                "in_shape": tuple(getattr(inp, "shape", ())),
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        """
        General tensor-mode solution:
          - If repeats is integer r: reshape trick -> view (N, r, ...) then sum axis=1.
          - If repeats is per-element (sequence/array): build boolean mapping matrix M (N_in x N_out)
            using comparisons on arange indices, then apply M @ grad_flat to aggregate.
        Legacy raw-mode keeps original implementation.
        """
        xp = self.metadata["xp"]
        repeats = self.metadata["repeats"]
        dim = int(self.metadata["dim"])
        in_shape = tuple(self.metadata["in_shape"])

        from ..tensor import Tensor

        if isinstance(grad_out, Tensor):
            # tensor-mode branch
            g_t = grad_out  # Tensor
            rank = len(getattr(g_t, "shape", ()))
            # normalize dim
            dim_norm = dim if dim >= 0 else dim + rank

            # bring axis to front
            perm = [dim_norm] + [i for i in range(rank) if i != dim_norm]
            perm_inv = [0] * rank
            for i, p in enumerate(perm):
                perm_inv[p] = i

            g_perm = g_t.permute(*perm)  # shape (N_out, *rest)
            N_out = int(getattr(g_perm, "shape", ())[0])
            rest_shape = tuple(getattr(g_perm, "shape", ())[1:])
            K = 1
            for s in rest_shape:
                K *= int(s)

            # flatten trailing dims: shape (N_out, K)
            if K == 0:
                # empty
                zero_res = _ensure_tensor_like(xp.zeros(in_shape), like_tensor=g_t)
                return [_tensor_unbroadcast(zero_res, in_shape)]

            g_2d = g_perm.reshape((N_out, K))

            N_in = in_shape[dim_norm]

            # Fast path: repeats is integer scalar
            if isinstance(repeats, int):
                r = int(repeats)
                # expect N_out == N_in * r
                # reshape g_2d to (N_in, r, K) then sum over axis=1
                g_rs = g_2d.reshape((N_in, r, K))  # (N_in, r, K)
                summed = g_rs.sum(dim=1)  # (N_in, K)
                # reshape back to (N_in, *rest_shape) and permute back
                summed_rs = summed.reshape((N_in,) + rest_shape)
                out_perm = summed_rs.transpose(tuple(perm_inv))
                out_final = _tensor_unbroadcast(out_perm, in_shape)
                return [out_final]

            # General path: repeats is sequence-like (list/1D-array)
            # Build idx array of length N_out: idx[j] = original index i corresponding to output position j
            xp_backend = self.metadata["xp"]
            # Build idx_raw using xp so we can then wrap as Tensor constant
            idx_raw = xp_backend.arange(N_in).repeat(repeats)
            # wrap as Tensor constant (no grad)
            idx_t = _ensure_tensor_like(idx_raw, like_tensor=g_t).reshape(
                (1, N_out)
            )  # shape (1, N_out)
            arange_in_t = _ensure_tensor_like(
                xp_backend.arange(N_in), like_tensor=g_t
            ).reshape((N_in, 1))  # (N_in,1)

            # mask M (N_in, N_out): M[i,j] = 1 if idx[j] == i
            M = (arange_in_t == idx_t).astype(
                getattr(g_t, "dtype", None)
            )  # (N_in, N_out) Tensor

            # compute aggregated gradients: (N_in, N_out) @ (N_out, K) -> (N_in, K)
            agg = M.matmul(g_2d)  # (N_in, K)

            agg_rs = agg.reshape((N_in,) + rest_shape)
            out_perm = agg_rs.transpose(tuple(perm_inv))
            out_final = _tensor_unbroadcast(out_perm, in_shape)
            return [out_final]

        g = xp.asarray(grad_out)
        N = in_shape[dim]
        idx = xp.arange(N).repeat(repeats)
        grad = xp.moveaxis(g, dim, 0)
        grad_x = xp.zeros((N,) + grad.shape[1:], dtype=grad.dtype)
        for i in range(N):
            grad_x[i] = grad[idx == i].sum(axis=0)
        grad_x = xp.moveaxis(grad_x, 0, dim)
        return [unbroadcast(grad_x, tuple(self.inputs[0].shape), xp)]


class SqueezeOp(OpNode):
    """
    Squeeze: remove axes of size 1.
    axis: None (remove all dims==1) or int or tuple/list of axes.
    """

    @classmethod
    def create(cls, a, axis=None):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        orig_shape = tuple(a_raw.shape)
        ndim = a_raw.ndim

        # normalize axis -> tuple of ints relative to original array
        if axis is None:
            axes = None
        else:
            if isinstance(axis, int):
                axes = (axis,)
            else:
                axes = tuple(axis)
            # normalize negative axes
            axes = tuple(ax if ax >= 0 else ax + ndim for ax in axes)
            # validate
            for ax in axes:
                if not (0 <= ax < ndim):
                    raise IndexError(
                        f"squeeze axis {ax} out of range for array of ndim {ndim}"
                    )

        # perform squeeze
        if axes is None:
            out_val = xp.squeeze(a_raw)
            removed_axes = tuple(i for i, s in enumerate(orig_shape) if s == 1)
        else:
            out_val = xp.squeeze(a_raw, axis=axes)
            removed_axes = tuple(sorted(axes))

        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        return cls(
            [a], out, {"xp": xp, "orig_shape": orig_shape, "removed_axes": removed_axes}
        )

    def backward(self, grad_out):
        """
        Supports:
         - Tensor-mode (grad_out is Tensor): reconstruct original shape using Tensor.reshape
           (insert singleton dims where removed) and _tensor_unbroadcast.
         - Legacy raw-mode: same as original (numpy/cupy based).
        """
        # lazy import Tensor
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        xp = self.metadata["xp"]
        orig_shape = tuple(self.metadata["orig_shape"])
        removed_axes = tuple(self.metadata["removed_axes"])
        a = self.inputs[0]

        # Tensor-mode
        if Tensor is not None and isinstance(grad_out, Tensor):
            g_t = grad_out
            # If no axes removed, just reshape back to orig_shape
            if len(removed_axes) == 0:
                gin = g_t.reshape(orig_shape)
                return [_tensor_unbroadcast(gin, tuple(a.shape))]

            # Build target shape by walking orig_shape, consuming dims from g_t.shape
            g_shape_iter = iter(getattr(g_t, "shape", ()))
            target_shape = []
            for i, s in enumerate(orig_shape):
                if s == 1 and i in removed_axes:
                    target_shape.append(1)
                else:
                    next_dim = next(g_shape_iter)
                    target_shape.append(int(next_dim))
            # reshape using Tensor.reshape
            gin = g_t.reshape(tuple(target_shape))
            # ensure final shape equals orig_shape
            gin = gin.reshape(orig_shape)
            return [_tensor_unbroadcast(gin, tuple(a.shape))]

        # Legacy raw branch (original behavior)
        g_raw = xp.asarray(grad_out)
        if len(removed_axes) == 0:
            ga = g_raw.reshape(orig_shape)
            return [unbroadcast(ga, tuple(a.shape), xp)]

        g_shape_iter = iter(g_raw.shape)
        target_shape = []
        for i, s in enumerate(orig_shape):
            if s == 1 and i in removed_axes:
                target_shape.append(1)
            else:
                next_dim = next(g_shape_iter)
                target_shape.append(int(next_dim))
        ga = xp.reshape(g_raw, tuple(target_shape))
        ga = xp.reshape(ga, orig_shape)
        return [unbroadcast(ga, tuple(a.shape), xp)]


class UnsqueezeOp(OpNode):
    """
    Unsqueeze (expand_dims): insert singleton dimensions at given axes.
    axes: int or tuple/list of axes (supports negative indices).
    """

    @classmethod
    def create(cls, a, axes):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        a_raw = to_raw(a.data)
        orig_shape = tuple(a_raw.shape)
        orig_ndim = a_raw.ndim

        # normalize input axes to a list
        if isinstance(axes, int):
            axes_list = [axes]
        else:
            axes_list = list(axes)

        # Normalize negative axes relative to insertion in original array.
        norm_axes = []
        for ax in axes_list:
            if ax < 0:
                ax_norm = ax + (orig_ndim + 1)
            else:
                ax_norm = ax
            if not (0 <= ax_norm <= orig_ndim):
                raise IndexError(
                    f"unsqueeze axis {ax} out of range for array of ndim {orig_ndim}"
                )
            norm_axes.append(int(ax_norm))

        sorted_axes = sorted(norm_axes)
        inserted_axes = []
        out_val = a_raw
        for i, ax in enumerate(sorted_axes):
            insert_at = ax + i  # account for earlier insertions shifting indices
            out_val = xp.expand_dims(out_val, axis=insert_at)
            inserted_axes.append(insert_at)

        out = Tensor(to_ap(out_val), requires_grad=a.requires_grad, device=a.device)
        return cls(
            [a],
            out,
            {
                "xp": xp,
                "orig_shape": orig_shape,
                "inserted_axes": tuple(inserted_axes),
                "axes": tuple(sorted_axes),
            },
        )

    def backward(self, grad_out):
        """
        Tensor-mode: if grad_out is Tensor, squeeze axes (in reverse order) using Tensor ops,
                    then _tensor_unbroadcast to orig_shape.
        Legacy: use xp.squeeze and unbroadcast.
        Returns: [grad_input]
        """
        # lazy import Tensor
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        xp = self.metadata["xp"]
        orig_shape = tuple(self.metadata["orig_shape"])
        inserted_axes = tuple(self.metadata["inserted_axes"])
        a = self.inputs[0]

        # Tensor-mode branch
        if Tensor is not None and isinstance(grad_out, Tensor):
            g_t = grad_out
            # If no inserted axes, just reshape back
            if len(inserted_axes) == 0:
                gin = g_t.reshape(orig_shape)
                return [_tensor_unbroadcast(gin, tuple(a.shape))]

            # squeeze inserted axes in reverse order
            for ax in sorted(inserted_axes, reverse=True):
                # try Tensor.squeeze(axis)
                g_t = g_t.squeeze(axis=ax)
                # ensure shape equals orig_shape
            g_t = g_t.reshape(orig_shape)
            return [_tensor_unbroadcast(g_t, tuple(a.shape))]

        # Legacy raw branch
        g_raw = xp.asarray(grad_out)
        if len(inserted_axes) == 0:
            ga = g_raw.reshape(orig_shape)
            return [unbroadcast(ga, tuple(a.shape), xp)]

        ga = g_raw
        for ax in sorted(inserted_axes, reverse=True):
            ga = xp.squeeze(ga, axis=ax)
        ga = ga.reshape(orig_shape)
        return [unbroadcast(ga, tuple(a.shape), xp)]


# ---------- in-place nodes ----------
class InplaceBase(OpNode):
    def _check_version(self, a, expected):
        if getattr(a, "_version", None) != expected + 1 and getattr(
            a, "is_leaf", False
        ):
            raise RuntimeError("in-place modification detected")


class IAddOp(InplaceBase):
    @classmethod
    def create(cls, a, b):
        xp = detect_xp_from_obj(a)
        return cls(
            [a, b],
            a,
            {
                "expected_version": getattr(a, "_version", 0),
                "a_shape": tuple(getattr(a, "shape", ())),
                "b_shape": tuple(getattr(b, "shape", ())),
                "xp": xp,
            },
        )

    def backward(self, grad_out):
        xp = self.metadata["xp"]
        a = self.inputs[0]
        b = self.inputs[1]
        self._check_version(a, self.metadata["expected_version"])

        # Tensor-mode
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        if Tensor is not None and isinstance(grad_out, Tensor):
            g = grad_out
            ga = _tensor_unbroadcast(g, tuple(getattr(a, "shape", ())))
            gb = (
                _tensor_unbroadcast(g, tuple(getattr(b, "shape", ())))
                if getattr(b, "requires_grad", False)
                else None
            )
            return [ga, gb]

        # legacy raw
        g_raw = xp.asarray(grad_out)
        return [
            unbroadcast(g_raw, tuple(getattr(a, "shape", ())), xp),
            unbroadcast(g_raw, tuple(getattr(b, "shape", ())), xp),
        ]


class ISubOp(InplaceBase):
    @classmethod
    def create(cls, a, b):
        xp = detect_xp_from_obj(a)
        return cls([a, b], a, {"expected_version": getattr(a, "_version", 0), "xp": xp})

    def backward(self, grad_out):
        xp = self.metadata["xp"]
        a = self.inputs[0]
        b = self.inputs[1]
        self._check_version(a, self.metadata["expected_version"])

        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        if Tensor is not None and isinstance(grad_out, Tensor):
            g = grad_out
            ga = _tensor_unbroadcast(g, tuple(getattr(a, "shape", ())))
            gb = (
                _tensor_unbroadcast(-g, tuple(getattr(b, "shape", ())))
                if getattr(b, "requires_grad", False)
                else None
            )
            return [ga, gb]

        g_raw = xp.asarray(grad_out)
        return [
            unbroadcast(g_raw, tuple(getattr(a, "shape", ())), xp),
            unbroadcast(-g_raw, tuple(getattr(b, "shape", ())), xp),
        ]


class IMulOp(InplaceBase):
    @classmethod
    def create(cls, a, b):
        xp = detect_xp_from_obj(a)
        orig = to_raw(a.data).copy()
        return cls(
            [a, b],
            a,
            {"expected_version": getattr(a, "_version", 0), "orig": orig, "xp": xp},
        )

    def backward(self, grad_out):
        xp = self.metadata["xp"]
        a = self.inputs[0]
        b = self.inputs[1]
        self._check_version(a, self.metadata["expected_version"])

        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        # Tensor-mode branch
        if Tensor is not None and isinstance(grad_out, Tensor):
            g = grad_out
            # ensure b_t is Tensor-like for arithmetic
            b_t = (
                b
                if isinstance(b, Tensor)
                else _ensure_tensor_like(getattr(b, "data", b), like_tensor=g)
            )
            ga = _tensor_unbroadcast(g * b_t, tuple(getattr(a, "shape", ())))
            if getattr(b, "requires_grad", False):
                # orig stored as raw array -> wrap as Tensor constant
                orig_t = _ensure_tensor_like(self.metadata["orig"], like_tensor=g)
                gb = _tensor_unbroadcast(g * orig_t, tuple(getattr(b, "shape", ())))
            else:
                gb = None
            return [ga, gb]

        # legacy raw branch
        g_raw = xp.asarray(grad_out)
        ga = unbroadcast(g_raw * to_raw(b.data), tuple(getattr(a, "shape", ())), xp)
        gb = (
            unbroadcast(
                g_raw * self.metadata["orig"], tuple(getattr(b, "shape", ())), xp
            )
            if getattr(b, "requires_grad", False)
            else None
        )
        return [ga, gb]


class IDivOp(InplaceBase):
    @classmethod
    def create(cls, a, b):
        xp = detect_xp_from_obj(a)
        orig = to_raw(a.data).copy()
        return cls(
            [a, b],
            a,
            {"expected_version": getattr(a, "_version", 0), "orig": orig, "xp": xp},
        )

    def backward(self, grad_out):
        xp = self.metadata["xp"]
        a = self.inputs[0]
        b = self.inputs[1]
        self._check_version(a, self.metadata["expected_version"])

        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        if Tensor is not None and isinstance(grad_out, Tensor):
            g = grad_out
            b_t = (
                b
                if isinstance(b, Tensor)
                else _ensure_tensor_like(getattr(b, "data", b), like_tensor=g)
            )
            ga = _tensor_unbroadcast(g / b_t, tuple(getattr(a, "shape", ())))
            if getattr(b, "requires_grad", False):
                orig_t = _ensure_tensor_like(self.metadata["orig"], like_tensor=g)
                gb = _tensor_unbroadcast(
                    -g * orig_t / (b_t**2), tuple(getattr(b, "shape", ()))
                )
            else:
                gb = None
            return [ga, gb]

        g_raw = xp.asarray(grad_out)
        ga = unbroadcast(g_raw / to_raw(b.data), tuple(getattr(a, "shape", ())), xp)
        gb = (
            unbroadcast(
                -g_raw * self.metadata["orig"] / (to_raw(b.data) ** 2),
                tuple(getattr(b, "shape", ())),
                xp,
            )
            if getattr(b, "requires_grad", False)
            else None
        )
        return [ga, gb]


class IModOp(InplaceBase):
    @classmethod
    def create(cls, a, b):
        xp = detect_xp_from_obj(a)
        orig = to_raw(a.data).copy()
        return cls(
            [a, b],
            a,
            {"expected_version": getattr(a, "_version", 0), "orig": orig, "xp": xp},
        )

    def backward(self, grad_out):
        xp = self.metadata["xp"]
        a = self.inputs[0]
        b = self.inputs[1]
        self._check_version(a, self.metadata["expected_version"])

        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        # raw branch
        if Tensor is not None and isinstance(grad_out, Tensor):
            # tensor-mode branch
            g = grad_out
            ga = _tensor_unbroadcast(g, tuple(getattr(a, "shape", ())))

            if getattr(b, "requires_grad", False):
                # Need floor(orig / b). orig is raw stored; wrap orig as Tensor constant.
                orig_t = _ensure_tensor_like(self.metadata["orig"], like_tensor=g)
                b_t = (
                    b
                    if isinstance(b, Tensor)
                    else _ensure_tensor_like(getattr(b, "data", b), like_tensor=g)
                )

                # Prefer tensor floor if available, else compute via backend floor on raw and wrap
                flo = (orig_t / b_t).floor()
                gb = _tensor_unbroadcast(-g * flo, tuple(getattr(b, "shape", ())))
            else:
                gb = None

            return [ga, gb]

        g_raw = xp.asarray(grad_out)
        ga = unbroadcast(g_raw, tuple(getattr(a, "shape", ())), xp)
        if getattr(b, "requires_grad", False):
            # grad wrt b: -g * floor(orig / b)
            gb_raw = -g_raw * xp.floor(self.metadata["orig"] / to_raw(b.data))
            gb = unbroadcast(gb_raw, tuple(getattr(b, "shape", ())), xp)
        else:
            gb = None
        return [ga, gb]


# ---------- comparison & logical convenience factories ----------
# ---------- comparisons (no grad) ----------
class CmpOp(OpNode):
    @classmethod
    def create(cls, a, b, op):
        from ..tensor import Tensor

        xp = detect_xp_from_obj(a)
        ar = to_raw(a.data)
        br = to_raw(b.data)
        if op == "eq":
            out_val = ar == br
        elif op == "ne":
            out_val = ar != br
        elif op == "lt":
            out_val = ar < br
        elif op == "le":
            out_val = ar <= br
        elif op == "gt":
            out_val = ar > br
        elif op == "ge":
            out_val = ar >= br
        else:
            raise ValueError("cmp op")
        out = Tensor(to_ap(out_val), requires_grad=False, device=a.device)
        return cls([a, b], out, {"op": op, "xp": xp})

    def backward(self, grad_out):
        try:
            from ..tensor import zeros
        except Exception:
            Tensor = None
        if Tensor is not None and isinstance(grad_out, Tensor):
            a_t = (
                self.inputs[0]
                if isinstance(self.inputs[0], Tensor)
                else _ensure_tensor_like(
                    getattr(self.inputs[0], "data", self.inputs[0]),
                    like_tensor=grad_out,
                )
            )
            b_t = (
                self.inputs[1]
                if isinstance(self.inputs[1], Tensor)
                else _ensure_tensor_like(
                    getattr(self.inputs[1], "data", self.inputs[1]),
                    like_tensor=grad_out,
                )
            )
            return [
                zeros(a_t.shape, device=grad_out.device),
                zeros(b_t.shape, device=grad_out.device),
            ]
        xp = self.metadata["xp"]
        return [
            xp.zeros_like(to_raw(self.inputs[0].data)),
            xp.zeros_like(to_raw(self.inputs[1].data)),
        ]


def eq_node(a, b):
    return CmpOp.create(a, b, "eq")


def ne_node(a, b):
    return CmpOp.create(a, b, "ne")


def lt_node(a, b):
    return CmpOp.create(a, b, "lt")


def le_node(a, b):
    return CmpOp.create(a, b, "le")


def gt_node(a, b):
    return CmpOp.create(a, b, "gt")


def ge_node(a, b):
    return CmpOp.create(a, b, "ge")


# --- CopyFromOp for in-place mapping ---
class CopyFromOp(OpNode):
    """
    CopyFromOp: copy (in-place effect) src -> dst and returns dst as output.
    - create(src, dst): copies src.data into dst.data immediately and returns a node with
      inputs=[src], output=dst.
    - backward(grad_out): gradient for src (dst is in-place target), with
      unbroadcast with shape of src.
    """

    @classmethod
    def create(cls, src, dst):
        """
        Perform the in-place copy at creation time (like PyTorch's x.copy_(y)).
        We set dst.data to a copy of src.data, and record shapes/xp in metadata.
        """
        xp = detect_xp_from_obj(src)
        # perform the actual copy (use raw copy to avoid aliasing)
        dst.data = to_ap(to_raw(src.data).copy())
        # ensure dst.requires_grad reflects whether src/dst require grad (conservative)
        dst.requires_grad = getattr(src, "requires_grad", False) or getattr(
            dst, "requires_grad", False
        )

        meta = {
            "xp": xp,
            "src_shape": tuple(getattr(src, "shape", ())),
            "dst_shape": tuple(getattr(dst, "shape", ())),
        }
        node = cls([src], dst, metadata=meta)
        # attach op to dst if backend expects it
        try:
            dst.op = node
        except Exception:
            pass
        return node

    def backward(self, grad_out):
        """
        Backward: gradient flows from dst back to src (identity with unbroadcasting).
        Returns a list with single element: gradient for src (or raw array / Tensor).
        """
        xp = self.metadata["xp"]
        src_shape = tuple(self.metadata["src_shape"])

        # Tensor-mode branch (create_graph=True)
        try:
            from ..tensor import Tensor
        except Exception:
            Tensor = None

        if Tensor is not None and isinstance(grad_out, Tensor):
            # grad_out is a Tensor -> keep it as Tensor ops and unbroadcast to src_shape
            g_t = grad_out
            g_src = _tensor_unbroadcast(g_t, src_shape)
            return [g_src]

        # legacy raw branch
        g_raw = xp.asarray(grad_out)
        g_src_raw = unbroadcast(g_raw, src_shape, xp)
        return [g_src_raw]


# end ops_nodes.py
