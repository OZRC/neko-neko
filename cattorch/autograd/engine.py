# cattorch/autograd/engine.py
import time
from typing import Any

from ..backend import _array as _ap
from ..backend.utils import _ensure_array, to_raw

Array = _ap.Array


def backward(
    output,
    grad: Any = None,
    retain_graph: bool = None,
    create_graph: bool = False,
    time_backward: bool = False,
):
    """
    Deterministic backward engine for OpNode-based autograd.

    - If create_graph is False (legacy): node.backward is expected to accept raw arrays (ap.Array)
      and return raw arrays; gradients are accumulated as Tensor(Array(...), requires_grad=False).
    - If create_graph is True (tensor-mode): node.backward is expected to accept Tensor grad_out
      and return Tensor gradients; engine will accumulate Tensor gradients (so the gradient
      computation itself forms a graph and higher-order derivatives are possible).
    - The function uses lazy import of Tensor to avoid circular imports.

    Note: For create_graph=True, we set retain_graph=True internally because building a graph of
    the derivatives requires retaining graph structure.
    """
    # lazy import to avoid circular
    from ..tensor import Tensor  # late import

    # If user requested create_graph, ensure we keep the forward graph (safe default)
    if retain_graph is None or create_graph:
        retain_graph = create_graph

    t0 = time.time()

    create_graph = bool(create_graph)

    # 1) initialize output.grad
    if grad is None:
        # create a Tensor seed (ones_like) — requires_grad flag left to ones_like implementation
        # (ones_like typically returns a Tensor with requires_grad=False)
        from ..tensor import ones_like  # lazy

        output.grad = ones_like(output)
    else:
        # grad provided
        if isinstance(grad, Tensor):
            output.grad = grad
        else:
            # wrap raw grad into Tensor (requires_grad=False)
            output.grad = Tensor(_ensure_array(grad, output.data), requires_grad=False)

    # 2) build topo order (DFS)
    topo, visited = [], set()

    def dfs(t):
        tid = id(t)
        if tid in visited:
            return
        visited.add(tid)
        node = getattr(t, "grad_fn", None)
        if node is not None:
            for ref in getattr(t, "_parents", tuple()):
                if ref is None:
                    continue
                p = ref()
                if p is not None:
                    dfs(p)
        topo.append(t)

    dfs(output)

    # 3) backward pass
    for t in reversed(topo):
        node = getattr(t, "grad_fn", None)
        if node is None:
            continue

        # grab the grad_out (as Tensor object)
        grad_out_tensor = getattr(t, "grad", None)
        if grad_out_tensor is None:
            continue

        # Legacy call paths: node might be a simple callable without .backward
        if callable(node) and not hasattr(node, "backward"):
            # try node with raw arrays first (legacy)
            # prepare raw if we have Tensor
            grad_out_raw = (
                to_raw(grad_out_tensor.data)
                if isinstance(grad_out_tensor, Tensor)
                else grad_out_tensor
            )
            node(grad_out_raw)
            continue

        if not hasattr(node, "backward"):
            continue

        # Prefer calling node.backward with a Tensor argument.
        # If node.backward only supports raw arrays, fallback to calling with raw and wrap results to Tensor.
        grads = None
        grads = node.backward(grad_out_tensor)

        if grads is None:
            continue

        parents = [
            ref() if ref is not None else None
            for ref in getattr(t, "_parents", tuple())
        ]

        # accumulate into parents
        for p, g in zip(parents, grads):
            g.requires_grad = g.requires_grad if create_graph else False
            if p is None or g is None:
                continue
            if not getattr(p, "requires_grad", False):
                continue

            # Ensure g is a Tensor
            if not isinstance(g, Tensor):
                # g might be raw ap.Array or numpy; convert to Tensor with proper device via _ensure_array
                g = Tensor(_ensure_array(g, p.data), requires_grad=False)
            # accumulate as Tensor addition to preserve graph
            if p.grad is None:
                p.grad = g
            else:
                p.grad = p.grad + g
        
    # 4) cleanup graph
    if not retain_graph:
        for t in topo:
            t.grad_fn = None
            t._parents = tuple()

    if time_backward:
        print(f"[backward] time = {time.time() - t0:.4f}s")


def grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
):
    """
    Compute and return the gradients of `outputs` w.r.t `inputs`.

    - outputs: Tensor or list/tuple of Tensors.
    - inputs: Tensor or list/tuple of Tensors.
    - grad_outputs: None or Tensor or list/tuple of Tensors (same length as outputs). If None, uses ones_like.
    - retain_graph: whether to retain forward graph (passed to backward); note create_graph=True forces retain_graph=True.
    - create_graph: whether to build graph of the gradient (higher-order derivatives).
    - allow_unused: if True return None for inputs that did not receive grad; else raise.
    """
    # lazy import to avoid circular
    from ..tensor import ones_like

    # normalize outputs and inputs to lists
    if isinstance(outputs, (list, tuple)):
        outs = list(outputs)
    else:
        outs = [outputs]

    if isinstance(inputs, (list, tuple)):
        ins = list(inputs)
    else:
        ins = [inputs]

    if len(outs) == 0:
        return [None for _ in ins]

    # normalize grad_outputs
    if grad_outputs is None:
        gos = [ones_like(o) for o in outs]
    else:
        if isinstance(grad_outputs, (list, tuple)):
            gos = list(grad_outputs)
        else:
            gos = [grad_outputs]

    if len(gos) != len(outs):
        raise ValueError("grad_outputs must have the same length as outputs")

    # Save original grads for inputs so we don't clobber user state
    orig_grads = [getattr(t, "grad", None) for t in ins]

    try:
        # For multiple outputs: call backward for each; use retain_graph True for all but the last
        for i, (out, go) in enumerate(zip(outs, gos)):
            last = i == len(outs) - 1
            # if create_graph True we want to build graph -> pass create_graph and keep retain_graph True
            rg = retain_graph or (not last)
            # call Tensor.backward so lazy imports consistent (backward signature supports create_graph)
            # Note: the Tensor.backward wrapper should forward create_graph to this engine.backward
            if (
                getattr(out, "requires_grad", False)
                or getattr(out, "grad_fn", None) is not None
            ):
                # If go is not Tensor wrap it (engine.backward will convert)
                out.backward(go, retain_graph=rg, create_graph=create_graph)
            else:
                # output not requiring grad nor has grad_fn -> nothing to do
                continue

        # collect results
        grads_out = []
        for t, orig in zip(ins, orig_grads):
            g = getattr(t, "grad", None)
            if g is None:
                if allow_unused:
                    grads_out.append(None)
                else:
                    raise RuntimeError(
                        "One of the differentiated Tensors does not require grad or did not receive grad"
                    )
            else:
                grads_out.append(g)
    finally:
        # restore original grads
        for t, orig in zip(ins, orig_grads):
            t.grad = orig

    return grads_out
