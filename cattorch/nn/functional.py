# cattorch/nn/functional.py
from typing import List, Tuple

from ..tensor import Tensor
from ..autograd import ops_nodes as _nodes


# Functional Access to Non-Dunder Methods
def transpose(input: Tensor, dim1: int, dim2: int):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.transpose(dim1, dim2)


def permute(input: Tensor, *dims: Tuple[int]):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.permute(dims)


def reshape(input: Tensor, *shape: Tuple[int]):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.reshape(shape)


def exp(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.exp()


def log(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.log()


def sum(input: Tensor, dim: int = None, keepdims: bool = False):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.sum(dim, keepdims)


def cumsum(input: Tensor, dim: int = None):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.cumsum(dim)


def mean(input: Tensor, dim: int = None, keepdims: bool = False):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.mean(dim, keepdims)


def var(input: Tensor, dim: int = None, keepdims: bool = False):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.var(dim, keepdims)


def max(input: Tensor, dim: int = None, keepdims: bool = False):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.max(dim, keepdims)


def argmax(input: Tensor, dim: int = None):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.argmax(dim)


def maximum(a, b):
    """
    Functional elementwise maximum. Accepts Tensors or scalars/ndarrays.
    Returns a Tensor (tracks grad if either input requires_grad).
    """
    # ensure Tensor wrappers
    if not isinstance(a, Tensor):
        a = Tensor(a, requires_grad=False)
    if not isinstance(b, Tensor):
        b = Tensor(b, requires_grad=False)
    return a.maximum(b)


def minimum(a, b):
    """
    Functional elementwise minimum. Accepts Tensors or scalars/ndarrays.
    Returns a Tensor (tracks grad if either input requires_grad).
    """
    if not isinstance(a, Tensor):
        a = Tensor(a, requires_grad=False)
    if not isinstance(b, Tensor):
        b = Tensor(b, requires_grad=False)
    return a.minimum(b)


def masked_fill(input: Tensor, mask: Tensor, value: int):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    if not isinstance(mask, Tensor):
        mask = Tensor(mask, requires_grad=False)
    return input.masked_fill(mask, value)


def matmul(a: Tensor, b: Tensor):
    if not isinstance(a, Tensor):
        a = Tensor(a, requires_grad=False)
    if not isinstance(b, Tensor):
        b = Tensor(b, requires_grad=False)
    return a @ b


def abs(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.abs()


def clamp(input: Tensor, min: float, max: float):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.clamp(min, max)


def sqrt(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.sqrt()


def sin(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.sin()


def cos(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.cos()


def tan(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.tan()


def silu(input: Tensor):
    return input * (1 / (1 + exp(-input)))


def sigmoid(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.sigmoid()


def relu(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.relu()


def leaky_relu(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.leaky_relu()


def gelu(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.gelu()


def tanh(input: Tensor):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.tanh()


def softmax(input: Tensor, dim: int = -1):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.softmax(dim=dim)


def log_softmax(input: Tensor, dim: int = -1):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.log_softmax(dim=dim)


def chunk(input: Tensor, chunks: int, dim: int = 0):
    if not isinstance(input, Tensor):
        input = Tensor(input, requires_grad=False)
    return input.chunk(chunks=chunks, dim=dim)


# MultiTensor Ops
def concatenate(tensors: List[Tensor], dim: int = 0):
    _tensors = []
    for x in tensors:
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=False)
        _tensors.append(x)

    node = _nodes.ConcatOp.create(_tensors, dim)
    out = node.output
    out.grad_fn = node
    out.grad_fn_name = node.__class__.__name__
    out._add_parents(*_tensors)
    out._is_leaf = False
    return out


def concat(tensors: List[Tensor], dim: int = 0):
    _tensors = []
    for x in tensors:
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=False)
        _tensors.append(x)
    node = _nodes.ConcatOp.create(_tensors, dim)
    out = node.output
    out.grad_fn = node
    out.grad_fn_name = node.__class__.__name__
    out._add_parents(*_tensors)
    out._is_leaf = False
    return out


def stack(tensors: List[Tensor], dim: int = 0):
    _tensors = []
    for x in tensors:
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=False)
        _tensors.append(x)
    node = _nodes.StackOp.create(_tensors, dim)
    out = node.output
    out.grad_fn = node
    out.grad_fn_name = node.__class__.__name__
    out._add_parents(*_tensors)
    out._is_leaf = False
    return out
