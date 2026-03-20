# cattorch/tensor.py
from __future__ import annotations

import weakref
import warnings

# ---------------------------
# Utility / no_grad context
# ---------------------------
from contextlib import ContextDecorator
from typing import Any, Optional, Tuple

import numpy as np

from .autograd import engine as autograd_engine
from .autograd import ops_nodes as _nodes  # factories & OpNode classes
from .backend import _array as ap
from .backend import dtypes as ctdtypes


# ---------------------------
# Utility / no_grad context
# ---------------------------
class _NoGrad(ContextDecorator):
    """
    Context manager (and decorator) that temporarily disables building the autograd graph.
    Usage:
        with thetorch.no_grad():
            ...
        @thetorch.no_grad()
        def fn(...): ...
    """

    def __enter__(self):
        # Save previous state and disable building graph for all Tensor subclasses
        self._old = Tensor._build_graph
        Tensor._build_graph = False
        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore previous state
        Tensor._build_graph = self._old
        # Do not suppress exceptions
        return False


# exported function returning a context/decorator like torch.no_grad
def no_grad():
    return _NoGrad()


def _check_inplace_allowed(tensor):
    if (
        getattr(tensor, "is_leaf", False)
        and tensor.requires_grad
        and tensor.__class__.build_graph_enabled()
    ):
        raise RuntimeError(
            "a leaf Variable that requires grad is being used in an in-place operation"
        )


# ---------------------------
# Tensor
# ---------------------------
class Tensor:
    _build_graph: bool = True

    @classmethod
    def build_graph_enabled(cls) -> bool:
        return cls._build_graph

    def __setattr__(self, name, value):
        # Prevent attaching grad_fn while graph building is disabled (no_grad context).
        if name in ("grad_fn", "_grad_fn") and (
            not self.__class__.build_graph_enabled()
        ):
            try:
                object.__setattr__(self, name, None)
            except Exception:
                pass
            return
        object.__setattr__(self, name, value)

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
        name: Optional[str] = None,
    ):
        # unify to ap.Array
        if isinstance(data, ap.Array):
            self._data = data
        else:
            try:
                self._data = ap.Array(data, device=device, dtype=dtype)
            except Exception:
                # fallback creating ap.Array from raw data
                self._data = ap.Array(data)

        self.requires_grad = bool(requires_grad)
        self.grad = None

        self.grad_fn = None
        self.grad_fn_name = None

        self._parents = tuple()
        self._strong_parents = None

        self._version = 0
        self._is_leaf = True
        self.op = getattr(self, "op", None)

        self.device = device or getattr(self._data, "device", None)
        self.dtype = dtype or getattr(self._data, "dtype", None)
        self.name = name

    # ---------------------------
    # basic array properties
    # ---------------------------
    @property
    def data(self) -> ap.Array:
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, ap.Array):
            value = ap.Array(value)
        self._data = value
        self._version = getattr(self, "_version", 0) + 1

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        from math import prod

        return int(prod(self.shape))

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    # ---------------------------
    # Array-like accessors
    # ---------------------------
    @property
    def xp(self):
        return getattr(self._data, "xp", __import__("numpy"))

    def asnumpy(self):
        try:
            return self._data.asnumpy()
        except Exception:
            return np.asarray(self._data)

    # ---------------------------
    # internal helpers
    # ---------------------------
    def _ensure_tensor(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor(other, requires_grad=False)

    def _add_parents(self, *parents):
        if not self.__class__.build_graph_enabled():
            return

        refs = []
        for p in parents:
            if p is None:
                refs.append(None)
            else:
                try:
                    refs.append(weakref.ref(p))
                except Exception:
                    refs.append(None)
        self._parents = tuple(refs)

    # ---------------------------
    # basic dunders and ops
    # ---------------------------
    def __repr__(self):
        o_str = f"Tensor({repr(self._data)}"
        if self.requires_grad:
            o_str += f", requires_grad={self.requires_grad}"
        if self.device != "cpu":
            o_str += f", device={self.device}"
        if self.grad_fn is not None:
            o_str += f", grad_fn=<{self.grad_fn_name}>"
        o_str += ")"
        return o_str

    # arithmetic dunders -> delegate to nodes
    def __add__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.AddOp.create(self, other_t)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, other_t)
        out._is_leaf = False
        return out

    __radd__ = __add__

    def __sub__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.SubOp.create(self, other_t)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, other_t)
        out._is_leaf = False
        return out

    def __rsub__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.SubOp.create(other_t, self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(other_t, self)
        out._is_leaf = False
        return out

    def __mul__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.MulOp.create(self, other_t)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, other_t)
        out._is_leaf = False
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.DivOp.create(self, other_t)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, other_t)
        out._is_leaf = False
        return out

    def __rtruediv__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.DivOp.create(other_t, self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(other_t, self)
        out._is_leaf = False
        return out

    def __floordiv__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.FloorDivOp.create(self, other_t)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, other_t)
        out._is_leaf = False
        return out

    def __rfloordiv__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.FloorDivOp.create(other_t, self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(other_t, self)
        out._is_leaf = False
        return out

    def __mod__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.ModOp.create(
            self,
            other_t,
        )
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, other_t)
        out._is_leaf = False
        return out

    def __matmul__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.MatMulOp.create(self, other_t)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, other_t)
        out._is_leaf = False
        return out

    def matmul(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.MatMulOp.create(self, other_t)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, other_t)
        out._is_leaf = False
        return out

    def __rmatmul__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.MatMulOp.create(other_t, self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(other_t, self)
        out._is_leaf = False
        return out

    def __pow__(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.PowOp.create(self, other_t)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, other_t)
        out._is_leaf = False
        return out

    def power(self, other):
        return self**other

    def __neg__(self):
        neg_one = Tensor(
            ap.Array(-1, dtype=self.dtype, device=self.device), requires_grad=False
        )
        node = _nodes.MulOp.create(self, neg_one)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, neg_one)
        out._is_leaf = False
        return out

    # Right-sided power (scalar ** tensor)
    def __rpow__(self, other):
        """Right-sided power: scalar ** tensor"""
        other_t = self._ensure_tensor(other)
        node = _nodes.PowOp.create(other_t, self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(other_t, self)
        out._is_leaf = False
        return out

    # ---------------------------
    # comparisons (no grad)
    # ---------------------------
    def __eq__(self, other):
        other_t = self._ensure_tensor(other)
        return _nodes.eq_node(self, other_t).output

    def __ne__(self, other):
        other_t = self._ensure_tensor(other)
        return _nodes.ne_node(self, other_t).output

    def __lt__(self, other):
        other_t = self._ensure_tensor(other)
        return _nodes.lt_node(self, other_t).output

    def __le__(self, other):
        other_t = self._ensure_tensor(other)
        return _nodes.le_node(self, other_t).output

    def __gt__(self, other):
        other_t = self._ensure_tensor(other)
        return _nodes.gt_node(self, other_t).output

    def __ge__(self, other):
        other_t = self._ensure_tensor(other)
        return _nodes.ge_node(self, other_t).output

    # ---------------------------
    # inplace ops: try to preserve in-place semantics using nodes
    # ---------------------------
    def _inplace_helper(self, other, opname):
        # opname: "iadd", "isub", "imul", "itruediv"
        if opname == "iadd":
            node_cls = _nodes.IAddOp
        elif opname == "isub":
            node_cls = _nodes.ISubOp
        elif opname == "imul":
            node_cls = _nodes.IMulOp
        elif opname == "itruediv":
            node_cls = _nodes.IDivOp
        else:
            raise ValueError("unsupported in-place op")

        other_t = self._ensure_tensor(other)
        node = node_cls.create(self, other_t)
        out = node.output
        # out is typically 'self' (in-place), so set grad_fn on out and parents
        try:
            out.grad_fn = node
            out.grad_fn_name = node.__class__.__name__
            out._add_parents(self, other_t)
            out._is_leaf = False
        except Exception:
            pass
        return out

    def __iadd__(self, other):
        _check_inplace_allowed(self)
        other_t = self._ensure_tensor(other)
        # create functional result (a + b) so we keep original history, then copy into self in-place
        orig_grad_fn = getattr(self, "grad_fn", None)
        orig_pars = getattr(self, "_parents", tuple())
        # create a shallow alias that preserves the original grad_fn and parents
        prev = Tensor(
            _nodes.to_ap(_nodes.to_raw(self.data).copy()),
            requires_grad=self.requires_grad,
            device=self.device,
        )
        # attach original parents to the alias so backward can reach original leaves
        prev._parents = orig_pars
        prev.grad_fn = orig_grad_fn
        func_node = _nodes.AddOp.create(prev, other_t)
        func_out = func_node.output
        # ensure func_out has grad_fn set (functional node)
        func_out.grad_fn = func_node
        func_out.grad_fn_name = func_node.__class__.__name__

        # coerce parents of func_out to include the alias and rhs (so backward flows)
        func_out._add_parents(prev, other_t)
        func_out._is_leaf = False
        # copy functional result into self (in-place effect)
        copy_node = _nodes.CopyFromOp.create(func_out, self)
        out = copy_node.output
        try:
            out.grad_fn = copy_node
            out.grad_fn_name = copy_node.__class__.__name__
            out._add_parents(func_out)
            out._is_leaf = False
        except Exception:
            pass
        return out

    def __isub__(self, other):
        _check_inplace_allowed(self)
        other_t = self._ensure_tensor(other)
        orig_grad_fn = getattr(self, "grad_fn", None)
        orig_pars = getattr(self, "_parents", tuple())
        prev = Tensor(
            _nodes.to_ap(_nodes.to_raw(self.data).copy()),
            requires_grad=self.requires_grad,
            device=self.device,
        )
        prev._parents = orig_pars
        prev.grad_fn = orig_grad_fn
        func_node = _nodes.SubOp.create(prev, other_t)
        func_out = func_node.output
        func_out.grad_fn = func_node
        func_out.grad_fn_name = func_node.__class__.__name__

        func_out._add_parents(prev, other_t)
        func_out._is_leaf = False
        copy_node = _nodes.CopyFromOp.create(func_out, self)
        out = copy_node.output
        try:
            out.grad_fn = copy_node
            out.grad_fn_name = copy_node.__class__.__name__
            out._add_parents(func_out)
            out._is_leaf = False
        except Exception:
            pass
        return out

    def __imul__(self, other):
        _check_inplace_allowed(self)
        other_t = self._ensure_tensor(other)
        orig_grad_fn = getattr(self, "grad_fn", None)
        orig_pars = getattr(self, "_parents", tuple())
        prev = Tensor(
            _nodes.to_ap(_nodes.to_raw(self.data).copy()),
            requires_grad=self.requires_grad,
            device=self.device,
        )
        prev._parents = orig_pars
        prev.grad_fn = orig_grad_fn
        func_node = _nodes.MulOp.create(prev, other_t)
        func_out = func_node.output
        func_out.grad_fn = func_node
        func_out.grad_fn_name = func_node.__class__.__name__

        func_out._add_parents(prev, other_t)
        func_out._is_leaf = False
        copy_node = _nodes.CopyFromOp.create(func_out, self)
        out = copy_node.output
        try:
            out.grad_fn = copy_node
            out.grad_fn_name = copy_node.__class__.__name__
            out._add_parents(func_out)
            out._is_leaf = False
        except Exception:
            pass
        return out

    def __itruediv__(self, other):
        _check_inplace_allowed(self)
        other_t = self._ensure_tensor(other)
        orig_grad_fn = getattr(self, "grad_fn", None)
        orig_pars = getattr(self, "_parents", tuple())
        prev = Tensor(
            _nodes.to_ap(_nodes.to_raw(self.data).copy()),
            requires_grad=self.requires_grad,
            device=self.device,
        )
        prev._parents = orig_pars
        prev.grad_fn = orig_grad_fn
        func_node = _nodes.DivOp.create(prev, other_t)
        func_out = func_node.output
        func_out.grad_fn = func_node
        func_out.grad_fn_name = func_node.__class__.__name__

        func_out._add_parents(prev, other_t)
        func_out._is_leaf = False
        copy_node = _nodes.CopyFromOp.create(func_out, self)
        out = copy_node.output
        try:
            out.grad_fn = copy_node
            out.grad_fn_name = copy_node.__class__.__name__
            out._add_parents(func_out)
            out._is_leaf = False
        except Exception:
            pass
        return out

    # ---------------------------
    # indexing (basic)
    # ---------------------------
    def __getitem__(self, idx):
        node = _nodes.IndexOp.create(self, idx)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def masked_fill(self, mask, value):
        """
        Instance method: self.masked_fill(mask, value)
        mask can be Tensor/ndarray/bool array; value can be scalar or Tensor.
        """
        # ensure other arg as Tensor only if it's already a Tensor (to track grad)
        value_obj = value if isinstance(value, Tensor) else value
        node = _nodes.MaskedFillOp.create(self, mask, value_obj)
        out = node.output
        out.grad_fn = node
        # attach parents for autograd bookkeeping similarly to other methods in your tensor.py
        out._add_parents(
            self,
            *([mask] if isinstance(mask, Tensor) else []),
            *([value] if isinstance(value, Tensor) else []),
        )
        out._is_leaf = False
        return out

    def chunk(self, chunks: int, dim: int = 0):
        """Split tensor into `chunks` pieces along `dim`, similar to torch.chunk.
        Returns a tuple of Tensors (may contain empty tensors if chunks > size).
        Implemented using slicing so gradients route through IndexOp.
        """
        if chunks <= 0:
            raise ValueError("chunks must be >= 1")
        size = self.shape[dim]
        base, rem = divmod(size, chunks)
        pieces = []
        start = 0
        for i in range(chunks):
            cur = base + (1 if i < rem else 0)
            idx = [slice(None)] * self.ndim
            idx[dim] = slice(start, start + cur)
            pieces.append(self[tuple(idx)])
            start += cur
        return tuple(pieces)

    # ---------------------------
    # unary op wrappers (methods)
    # ---------------------------
    def exp(self):
        node = _nodes.ExpOp.create(self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def log(self):
        node = _nodes.LogOp.create(self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def sqrt(self):
        node = _nodes.SqrtOp.create(self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def sin(self):
        node = _nodes.TrigOp.create(self, "sin")
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def cos(self):
        node = _nodes.TrigOp.create(self, "cos")
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def tan(self):
        node = _nodes.TrigOp.create(self, "tan")
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def abs(self):
        node = _nodes.AbsOp.create(self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def tanh(self):
        node = _nodes.TanhOp.create(self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def sign(self):
        node = _nodes.SignOp.create(self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def sigmoid(self):
        node = _nodes.SigmoidOp.create(self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def gelu(self):
        node = _nodes.GeluOp.create(self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def leaky_relu(self):
        node = _nodes.LeakyReluOp.create(self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def relu(self):
        node = _nodes.ReluOp.create(self)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def softmax(self, dim: int = None):
        if dim is None:
            raise TypeError(
                "softmax() received an invalid combination of arguments - got (), but expected:* (int dim)"
            )
        node = _nodes.SoftmaxOp.create(self, dim=dim)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def log_softmax(self, dim: int = None):
        if dim is None:
            raise TypeError(
                "log_softmax() received an invalid combination of arguments - got (), but expected:* (int dim)"
            )
        node = _nodes.LogSoftmaxOp.create(self, dim=dim)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    # ---------------------------
    # shape / view helpers
    # ---------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = list(shape[0])
        else:
            shape = list(shape)
        node = _nodes.ReshapeOp.create(self, tuple(shape))
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def swapaxes(self, axis0, axis1):
        if axis1 is None and isinstance(axis0, (tuple, list)) and len(axis0) > 1:
            axis1 = axis0[0]
            axis0 = axis0[1]
        node = _nodes.SwapaxesOp.create(self, axis0, axis1)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out
# --- in cattorch/tensor.py ---

    def transpose(self, *axes):
        """
        General transpose:
        - no args: reverse axes
        - two ints: swap those two axes
        - a single tuple/list: use that as perm
        This implementation is consistent with permute(): it calls the
        TransposeOp factory and *returns node.output* (attaching grad_fn and parents).
        """
        from .autograd import ops_nodes as _nodes

        # normalize perm
        rank = len(getattr(self, "shape", ()))
        if len(axes) == 0:
            perm = tuple(reversed(range(rank)))
        elif len(axes) == 1:
            arg = axes[0]
            if isinstance(arg, (tuple, list)):
                perm = tuple(arg)
            elif isinstance(arg, int):
                # single int -> swap with last dim? (not typical) -> treat as fallback
                raise ValueError("transpose: single integer not supported; provide two ints or a tuple")
            else:
                raise ValueError("transpose: unsupported argument")
        elif len(axes) == 2:
            a0, a1 = axes
            # support negative indices
            if a0 < 0:
                a0 = a0 + rank
            if a1 < 0:
                a1 = a1 + rank
            perm_list = list(range(rank))
            perm_list[a0], perm_list[a1] = perm_list[a1], perm_list[a0]
            perm = tuple(perm_list)
        else:
            raise ValueError("transpose: expected 0, 2, or a single tuple/list argument")

        # call factory: it returns an OpNode (consistent with other op factories)
        node = _nodes.TransposeOp.create(self, perm)
        out = node.output if not hasattr(node, "data") else node
        # attach op link so autograd engine knows this op
        out.grad_fn = node
        # register parent(s)
        out._add_parents(self)
        return out


    def permute(self, *dims):
        """
        Named permute for arbitrary permutation of axes.
        Make permute consistent with transpose by using the same factory.
        """
        from .autograd import ops_nodes as _nodes

        rank = len(getattr(self, "shape", ()))
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            perm = tuple(dims[0])
        else:
            perm = tuple(dims)

        if any(d < 0 for d in perm):
            perm = tuple(d + rank if d < 0 else d for d in perm)

        if len(perm) != rank:
            raise ValueError("permute: permutation must have same number of dimensions as tensor")

        node = _nodes.TransposeOp.create(self, perm)
        # node is expected to be an OpNode -> use its output
        out = node.output if not hasattr(node, "data") else node
        out.grad_fn = node
        out._add_parents(self)
        return out

    @property
    def T(self):
        if self.ndim < 2:
            return self
        return self.transpose(0, 1)

    def reshape_as(self, *shape):
        return self.reshape(*shape)

    # ---------------------------
    # reductions / comparisons / indexing helpers (exposed as methods)
    # ---------------------------
    def sum(self, dim: int = None, keepdims: bool = False):
        node = _nodes.ReduceSumOp.create(self, dim=dim, keepdims=keepdims)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def mean(self, dim: int = None, keepdims: bool = False):
        node = _nodes.ReduceMeanOp.create(self, dim=dim, keepdims=keepdims)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def var(self, dim: int = None, keepdims: bool = False):
        node = _nodes.ReduceVarOp.create(self, dim=dim, keepdims=keepdims)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def prod(self, dim: int = None, keepdims: bool = False):
        node = (
            _nodes.ReduceProdOp.create(self, dim, keepdims)
            if hasattr(_nodes, "ReduceProdOp")
            else None
        )
        if node is not None:
            out = node.output
            out.grad_fn = node
            out.grad_fn_name = node.__class__.__name__
            out._add_parents(self)
            out._is_leaf = False
            return out
        # fallback to numpy reduction if MaxOp missing
        raw = _nodes.to_raw(self.data)
        val = raw.max()
        return Tensor(_nodes.to_ap(val), requires_grad=False)

    def max(self, dim: int = None, keepdims: bool = False):
        node = (
            _nodes.ReduceMaxOp.create(self, dim, keepdims)
            if hasattr(_nodes, "ReduceMaxOp")
            else None
        )
        if node is not None:
            out = node.output
            out.grad_fn = node
            out.grad_fn_name = node.__class__.__name__
            out._add_parents(self)
            out._is_leaf = False
            return out
        # fallback to numpy reduction if MaxOp missing
        raw = _nodes.to_raw(self.data)
        val = raw.max()
        return Tensor(_nodes.to_ap(val), requires_grad=False)

    def min(self, dim: int = None, keepdims: bool = False):
        node = (
            _nodes.ReduceMinOpe.create(self, dim, keepdims)
            if hasattr(_nodes, "ReduceMinOpe")
            else None
        )
        if node is not None:
            out = node.output
            out.grad_fn = node
            out.grad_fn_name = node.__class__.__name__
            out._add_parents(self)
            out._is_leaf = False
            return out
        # fallback to numpy reduction if MaxOp missing
        raw = _nodes.to_raw(self.data)
        val = raw.max()
        return Tensor(_nodes.to_ap(val), requires_grad=False)

    def argmax(self, dim: int = None):
        node = (
            _nodes.ArgMaxOp.create(self, dim) if hasattr(_nodes, "ArgMaxOp") else None
        )
        if node is not None:
            out = node.output
            out.grad_fn = node
            out.grad_fn_name = node.__class__.__name__
            out._add_parents(self)
            out._is_leaf = False
            return out
        raw = _nodes.to_raw(self.data)
        val = raw.argmax()
        return Tensor(_nodes.to_ap(val), requires_grad=False)

    def maximum(self, other):
        other_t = self._ensure_tensor(other)
        node = _nodes.MaximumOp.create(self, other_t)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self, other_t)
        out._is_leaf = False
        return out

    def minimum(self, other) -> Tensor:
        other_t = self._ensure_tensor(other)
        node = _nodes.MinimumOp.create(self, other_t)
        out = node.output
        try:
            out.grad_fn = node
            out._add_parents(self, other_t)
            out._is_leaf = False
        except Exception:
            pass
        return out

    def floor(self):
        node = _nodes.FloorOp.create(self)
        out = node.output
        try:
            out.grad_fn = node
            out._add_parents(self)
            out._is_leaf = False
        except Exception:
            pass
        return out

    def cumsum(self, dim: int = None):
        node = _nodes.CumsumOp.create(self, dim=dim)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def repeat_interleave(self, repeats: int, dim: int = 0):
        node = _nodes.RepeatInterleaveOp.create(self, repeats=repeats, dim=dim)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def squeeze(self, dim: int = None):
        node = _nodes.SqueezeOp.create(self, dim)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    def unsqueeze(self, dim=None):
        if dim is None:
            raise TypeError(
                'unsqueeze() missing 1 required positional arguments: "dim"'
            )
        node = _nodes.UnsqueezeOp.create(self, dim)
        out = node.output
        out.grad_fn = node
        out.grad_fn_name = node.__class__.__name__
        out._add_parents(self)
        out._is_leaf = False
        return out

    # ---------------------------
    # other helpers / factories
    # ---------------------------
    def detach(self):
        # shallow copy but not requiring grad
        t = Tensor(self.data, requires_grad=False, device=self.device, dtype=self.dtype)
        return t

    def clone(self):
        return Tensor(
            _nodes.to_ap(_nodes.to_raw(self.data).copy()),
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def to(self, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return Tensor(
            self.data.to(device),
            requires_grad=self.requires_grad,
            device=device,
            dtype=dtype,
        )

    def isfinite(self):
        return Tensor(self.xp.isfinite(self.data), dtype=ctdtypes.bool)

    def where(self, condition: "Tensor", other) -> "Tensor":
        # lazy import to avoid circulars (Tensor is defined in this module)
        if not isinstance(condition, Tensor):
            # allow passing raw arrays as condition
            condition = Tensor(condition, requires_grad=False)

        condition = 1 - condition
        condition = condition.to(device=self.device, dtype=ctdtypes.bool)

        # call the instance method masked_fill (it will create the MaskedFillOp, set grad_fn, parents)
        return self.masked_fill(condition, other)

    def astype(self, dtype, copy: bool = False):
        """
        Return a new Tensor with data cast to `dtype`.

        - dtype: a dtype understood by the backend (numpy/cupy dtype or string or ctdtypes.*)
        - copy: if False and dtype equals current dtype, may return self (no copy).
        Behavior matches .to(dtype=...) semantics in this codebase:
        - does NOT create a grad_fn (it's not an autograd op), simply returns a new Tensor.
        - preserves `requires_grad` flag (but does not preserve grad_fn / history).
        """
        # normalize requested dtype
        if dtype is None:
            return self

        # if dtype matches and copy not requested -> return self
        current_dtype = getattr(self._data, "dtype", None)
        # Try to compare in a tolerant way
        try:
            same_dtype = current_dtype == dtype
        except Exception:
            same_dtype = False

        if same_dtype and not copy:
            return self

        # Try to use ap.Array.astype if provided by backend
        try:
            if hasattr(self._data, "astype"):
                try:
                    new_arr = self._data.astype(dtype)
                except TypeError:
                    # Some ap.Array.astype may want a numpy dtype object; try convert via xp
                    xp = self.xp
                    raw = xp.asarray(self._data)
                    raw_cast = raw.astype(dtype)
                    new_arr = ap.Array(raw_cast, device=self.device, dtype=dtype)
            else:
                # fallback: convert to raw numpy/cupy array then astype and wrap
                xp = self.xp
                raw = xp.asarray(self._data)
                raw_cast = raw.astype(dtype)
                new_arr = ap.Array(raw_cast, device=self.device, dtype=dtype)
        except Exception:
            # very robust fallback: use numpy as last resort
            try:
                raw = self.xp.asarray(self._data)
                raw_cast = raw.astype(dtype)
                new_arr = ap.Array(raw_cast, device=self.device, dtype=dtype)
            except Exception:
                # if all fails, raise
                raise

        # return new Tensor (preserve requires_grad flag, but do not attach grad_fn)
        return Tensor(
            new_arr, requires_grad=self.requires_grad, device=self.device, dtype=dtype
        )

    # ---------------------------
    # backward entrypoint
    # ---------------------------
    def backward(
        self,
        grad: Optional["Tensor"] = None,
        retain_graph: bool = True,
        create_graph: bool = False,
    ):
        if getattr(self, "requires_grad", True):
            if create_graph:
                warnings.warn(
                    "Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak."
                    "We recommend using autograd.grad when creating the graph to avoid this."
                    "If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak."
                )
                autograd_engine.backward(
                    self, grad=grad, retain_graph=retain_graph, create_graph=create_graph
                )
            else:
                with no_grad():
                    autograd_engine.backward(
                        self, grad=grad, retain_graph=retain_graph, create_graph=create_graph
                    )

    # ---------------------------
    # small factories as module-level helpers (exposed in __init__)
    # ---------------------------


def arange(*shape, device=None, dtype=np.int32):
    from .backend._array import Array

    raw = np.arange(*shape, dtype=dtype)
    return Tensor(Array(raw, device=device), requires_grad=False, device=device)


def zeros(shape, device=None, dtype=np.float32):
    from .backend._array import Array

    raw = np.zeros(shape, dtype=dtype)
    return Tensor(Array(raw, device=device), requires_grad=False, device=device)


def ones(shape, device=None, dtype=np.float32):
    from .backend._array import Array

    raw = np.ones(shape, dtype=dtype)
    return Tensor(Array(raw, device=device), requires_grad=False, device=device)


def full(shape, fill_value, device=None, dtype=np.float32):
    from .backend._array import Array

    raw = np.full(shape, fill_value, dtype=dtype)
    return Tensor(Array(raw, device=device), requires_grad=False, device=device)


def empty(shape, device=None, dtype=np.float32):
    from .backend._array import Array

    raw = np.empty(shape, dtype=dtype)
    return Tensor(Array(raw, device=device), requires_grad=False, device=device)


def eye(n, device=None, dtype=np.float32):
    from .backend._array import Array

    raw = np.eye(n, dtype=dtype)
    return Tensor(Array(raw, device=device), requires_grad=False, device=device)


def linspace(start, stop, num=50, device=None, dtype=np.float32):
    from .backend._array import Array

    raw = np.linspace(start, stop, num=num, dtype=dtype)
    return Tensor(Array(raw, device=device), requires_grad=False, device=device)


def ones_like(x):
    return ones(x.shape, device=getattr(x, "device", None))


def zeros_like(x):
    return zeros(x.shape, device=getattr(x, "device", None))


def empty_like(x):
    return empty(x.shape, device=getattr(x, "device", None))


def full_like(x, fill_value):
    return full(x.shape, fill_value, device=getattr(x, "device", None))


def rand(*shape, device=None):
    from .backend._array import Array

    raw = np.random.rand(*shape).astype(np.float32)
    return Tensor(Array(raw, device=device), requires_grad=False, device=device)


def randn(*shape, device=None):
    from .backend._array import Array

    raw = np.random.randn(*shape).astype(np.float32)
    return Tensor(Array(raw, device=device), requires_grad=False, device=device)


def randint(low, high, shape, device=None):
    from .backend._array import Array

    raw = np.random.randint(low, high, size=shape).astype(np.int32)
    return Tensor(Array(raw, device=device), requires_grad=False, device=device)


def tril(x, k=0):
    raw = _nodes.to_raw(x.data if isinstance(x, Tensor) else x)
    out = np.tril(raw, k=k)
    return Tensor(
        _nodes.to_ap(out), requires_grad=False, device=getattr(x, "device", None)
    )


def rand_like(x):
    return rand(x.shape)


def randn_like(x):
    return randn(x.shape)
