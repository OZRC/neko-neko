# cattorch/einsum.py
from __future__ import annotations

import string
import weakref
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .autograd.ops_nodes import (
    BroadcastOp,
    DiagonalOp,
    ExpandDimsOp,
    IdentityOp,
    MatMulOp,
    MultiMulOp,
    OpNode,
    ReduceSumOp,
    TransposeOp,
)
from .backend import _array as ap
from .backend.utils import get_xp_for, to_ap, to_raw
from .tensor import Tensor

# optional cupy
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

# =========================
# Einsum symbolic builder & helpers
# =========================

ALL_INDICES = list(string.ascii_letters)
ENABLE_LOG = False


def log(*args, **kwargs):
    if ENABLE_LOG:
        print(*args, **kwargs)


def sample_ellipsis(ndim: int, used: Set[str]) -> List[str]:
    """Pick a contiguous block of unused letters of length `ndim`."""
    for i in range(len(ALL_INDICES) - ndim + 1):
        candidate = ALL_INDICES[i : i + ndim]
        if all(c not in used for c in candidate):
            return candidate
    raise ValueError("Not enough available indices to sample ellipsis.")


def expand_equation(equation: str, operands: List) -> Tuple[str, Optional[List[str]]]:
    """
    Expand '...' in einsum equation by sampling unused index names.
    Returns (new_equation, ellipsis_inds_or_None).
    """
    if "->" not in equation:
        raise ValueError(
            "einsum equation must include '->' to indicate output indices."
        )
    lhs, rhs = equation.split("->")
    inputs = lhs.split(",")
    if "..." not in equation:
        return equation, None

    needed = []
    has_ellipsis_flags = []
    for sub, op in zip(inputs, operands):
        if "..." in sub:
            explicit = len([c for c in sub if c.isalpha()])
            try:
                ndim = int(op.ndim)
            except Exception:
                ndim = len(getattr(op, "shape", tuple(op.shape)))
            k = ndim - explicit
            if k < 0:
                raise RuntimeError(
                    f"einsum(): number of explicit subscripts ({explicit}) greater than operand dimensions ({ndim})"
                )
            needed.append(k)
            has_ellipsis_flags.append(True)
        else:
            needed.append(0)
            has_ellipsis_flags.append(False)

    used = set("".join(inputs).replace(".", "") + rhs.replace(".", ""))

    if len(operands) == 1 and has_ellipsis_flags[0]:
        k = needed[0]
        ellipsis_inds = [] if k == 0 else sample_ellipsis(k, used)
        new_inputs = [inputs[0].replace("...", "".join(ellipsis_inds))]
        if "..." in rhs:
            rhs = rhs.replace("...", "".join(ellipsis_inds))
        new_eq = ",".join(new_inputs) + "->" + rhs
        return new_eq, ellipsis_inds if ellipsis_inds else None

    pool_len = max(needed) if any(has_ellipsis_flags) else 0
    ellipsis_inds = None
    if pool_len > 0:
        ellipsis_inds = sample_ellipsis(pool_len, used)

    _inputs = []
    for sub, k in zip(inputs, needed):
        if "..." in sub:
            if k == 0:
                expanded = sub.replace("...", "")
            else:
                # take the last k elements from ellipsis_inds to match shape alignment
                slice_inds = ellipsis_inds[pool_len - k : pool_len]
                expanded = sub.replace("...", "".join(slice_inds))
            _inputs.append(expanded)
        else:
            _inputs.append(sub)

    if "..." in rhs:
        if pool_len == 0:
            rhs = rhs.replace("...", "")
        else:
            rhs = rhs.replace("...", "".join(ellipsis_inds))

    new_eq = ",".join(_inputs) + "->" + rhs
    return new_eq, ellipsis_inds


@dataclass(frozen=True)
class Index:
    name: str
    size: int

    def __eq__(self, other):
        if not isinstance(other, Index):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class SymTensor:
    name: str
    indices: List[Index]
    shape: tuple


@dataclass
class DiagonalConstraint:
    index: Index
    tensor: SymTensor


@dataclass
class EqualityConstraint:
    index: Index
    tensors: List[SymTensor]


@dataclass
class SymbolicGraph:
    tensors: List[SymTensor]
    indices: Dict[str, Index]
    constraints: List
    output_indices: List[Index]
    reduction_indices: Set[Index]
    ellipsis: Optional[List[str]] = None
    contraction_order: Optional[List[Tuple[int, int]]] = None

    def __str__(self):
        s = "SymbolicGraph:\n"
        s += " Tensors:\n"
        for t in self.tensors:
            s += f"  {t.name}: shape={t.shape}, indices={[i.name for i in t.indices]}\n"
        s += " Constraints:\n"
        for c in self.constraints:
            if isinstance(c, DiagonalConstraint):
                s += f"  Diagonal: index={c.index.name} in tensor={c.tensor.name}\n"
            elif isinstance(c, EqualityConstraint):
                ts = ", ".join(t.name for t in c.tensors)
                s += f"  Equality: index={c.index.name} in tensors=[{ts}]\n"
        s += " Output Indices: " + ", ".join(i.name for i in self.output_indices) + "\n"
        s += (
            " Reduction Indices: "
            + ", ".join(i.name for i in self.reduction_indices)
            + "\n"
        )
        if self.ellipsis:
            s += " Ellipsis indices (ordered): " + ",".join(self.ellipsis) + "\n"
        if self.contraction_order:
            s += (
                " Contraction order (pairs of tensor indices): "
                + str(self.contraction_order)
                + "\n"
            )
        return s


class EinsumSymbolicBuilder:
    def __init__(self, equation: str, operands: List):
        self.ellipsis_order: Optional[List[str]] = None
        if "..." in equation:
            equation, ellipsis_inds = expand_equation(equation, operands)
            if ENABLE_LOG:
                log(f"  [Builder] Expanded equation: {equation}")
            if ellipsis_inds:
                self.ellipsis_order = list(ellipsis_inds)
        self.eq = equation.replace(" ", "").strip()
        self.operands = operands

    def build(self) -> SymbolicGraph:
        if "->" not in self.eq:
            raise ValueError("einsum equation must include '->'")
        lhs, rhs = self.eq.split("->")
        subs = lhs.split(",")

        index_map: Dict[str, Index] = {}
        tensors: List[SymTensor] = []

        # Build index map (name -> Index(name, size))
        for i, (sub, op) in enumerate(zip(subs, self.operands)):
            try:
                ndim = int(op.ndim)
                shape = tuple(op.shape)
            except Exception:
                # raw ndarray or similar
                shape = tuple(op.shape)
                ndim = len(shape)

            if len(sub) != ndim:
                raise RuntimeError(
                    f"einsum(): number of subscripts ({len(sub)}) does not match operand dimensions ({ndim}) for operand {i}"
                )

            for ax, c in enumerate(sub):
                if c not in index_map:
                    index_map[c] = Index(c, int(shape[ax]))
                else:
                    # If index comes from ellipsis expansion we allow handling elsewhere
                    if self.ellipsis_order and c in self.ellipsis_order:
                        pass
                    else:
                        assert index_map[c].size == int(shape[ax])

        # Build SymTensor list
        for i, (sub, op) in enumerate(zip(subs, self.operands)):
            try:
                shape = tuple(op.shape)
            except Exception:
                shape = tuple(op.shape)
            idxs = [index_map[c] for c in sub]
            tensors.append(SymTensor(name=f"T{i}", indices=idxs, shape=shape))

        constraints = []

        # Diagonal constraints: same index repeated within same tensor
        for t in tensors:
            cnt = Counter(i.name for i in t.indices)
            for k, v in cnt.items():
                if v > 1:
                    constraints.append(DiagonalConstraint(index_map[k], t))

        # Equality constraints: same index appears across tensors (except ellipsis)
        usage = defaultdict(list)
        for t in tensors:
            for idx in set(t.indices):
                usage[idx].append(t)

        for idx, ts in usage.items():
            if len(ts) > 1 and (
                not self.ellipsis_order or idx.name not in self.ellipsis_order
            ):
                constraints.append(EqualityConstraint(idx, ts))

        output_indices = [index_map[c] for c in rhs]
        reduction_indices = set(index_map.values()) - set(output_indices)

        graph = SymbolicGraph(
            tensors=tensors,
            indices=index_map,
            constraints=constraints,
            output_indices=output_indices,
            reduction_indices=reduction_indices,
            ellipsis=self.ellipsis_order,
        )

        graph.contraction_order = compute_greedy_contraction_order(graph)
        return graph


def compute_greedy_contraction_order(graph: SymbolicGraph) -> List[Tuple[int, int]]:
    """A simple greedy heuristic to pick contraction pairs that minimize immediate intermediate size."""
    tensors = list(graph.tensors)
    orig_indices = list(range(len(tensors)))
    order: List[Tuple[int, int]] = []
    cur_indices = [list(t.indices) for t in tensors]

    while len(cur_indices) > 1:
        best = None
        best_cost = None
        for i in range(len(cur_indices)):
            for j in range(i + 1, len(cur_indices)):
                a = cur_indices[i]
                b = cur_indices[j]
                union_dict = {}
                for idx in a + b:
                    union_dict[idx.name] = idx
                cost = 1
                for idx in union_dict.values():
                    cost *= idx.size
                if best is None or cost < best_cost:
                    best = (i, j)
                    best_cost = cost
        if best is None:
            break
        i, j = best
        order.append((orig_indices[i], orig_indices[j]))
        # merge j into i
        new_indices = []
        name_set = set()
        for idx in cur_indices[i] + cur_indices[j]:
            if idx.name not in name_set:
                new_indices.append(idx)
                name_set.add(idx.name)
        cur_indices[i] = new_indices
        # remove j
        cur_indices.pop(j)
        orig_indices.pop(j)
    return order


# =========================
# Lowering: SymbolicGraph -> primitive op sequence (forward + op list)
# =========================


class SymbolicLowerer:
    def __init__(
        self,
        graph: SymbolicGraph,
        arrays: List,
        orig_tensors: Optional[Dict[str, Tensor]] = None,
    ):
        self.graph = graph
        self.arrays = arrays
        self.orig_tensors = orig_tensors or {}

    def _check_ellipsis_broadcastable(
        self, tensor_map: Dict[str, Tensor], tensor_indices_map: Dict[str, List[Index]]
    ):
        """Verify ellipsis (batch) dims are broadcastable across tensors and return target batch shape."""
        if not self.graph.ellipsis:
            return ()
        ell_order = list(self.graph.ellipsis)
        batch_shapes: Dict[str, Tuple[int, ...]] = {}
        for t in self.graph.tensors:
            tname = t.name
            idxs = tensor_indices_map[tname]
            actual_shape = tuple(to_raw(tensor_map[tname].data).shape)
            bs = []
            for pos, idx in enumerate(idxs):
                if idx.name in ell_order:
                    if pos >= len(actual_shape):
                        raise ValueError(f"Axis position mismatch for tensor {tname}")
                    bs.append(actual_shape[pos])
            batch_shapes[tname] = tuple(bs)

        def broadcast_pair(s1, s2):
            if not s1:
                return s2
            if not s2:
                return s1
            l1, l2 = len(s1), len(s2)
            L = max(l1, l2)
            s1r = (1,) * (L - l1) + s1
            s2r = (1,) * (L - l2) + s2
            out = []
            for a, b in zip(s1r, s2r):
                if a == b:
                    out.append(a)
                elif a == 1:
                    out.append(b)
                elif b == 1:
                    out.append(a)
                else:
                    raise RuntimeError(
                        f"Einsum ellipsis batch dims not broadcastable: {s1} vs {s2}"
                    )
            return tuple(out)

        target = ()
        for s in batch_shapes.values():
            target = broadcast_pair(target, s)
        return target

    def lower(self):
        ops: List[OpNode] = []
        tensor_map: Dict[str, Tensor] = {}
        tensor_indices_map: Dict[str, List[Index]] = {}

        # create original leaf Tensors (reuse user-provided Tensor if exists)
        for t, arr in zip(self.graph.tensors, self.arrays):
            orig = self.orig_tensors.get(t.name)
            if orig is not None:
                orig.name = t.name
                tensor_map[t.name] = orig
                tensor_indices_map[t.name] = list(t.indices)
            else:
                leaf = Tensor(arr, requires_grad=False)
                leaf.name = t.name
                tensor_map[t.name] = leaf
                tensor_indices_map[t.name] = list(t.indices)

        # Diagonal constraints: reduce repeated axes inside each tensor by selecting diagonal
        for c in self.graph.constraints:
            if not isinstance(c, DiagonalConstraint):
                continue
            tname = c.tensor.name
            while True:
                T = tensor_map[tname]
                indices = tensor_indices_map[tname]
                axes = [i for i, idx in enumerate(indices) if idx == c.index]
                if len(axes) < 2:
                    break
                ax0, ax1 = axes[0], axes[-1]
                node = DiagonalOp.create(T, ax0=ax0, ax1=ax1, index=c.index)
                ops.append(node)
                tensor_map[tname] = node.output
                # drop the second occurrence of the diagonal index
                new_indices = [idx for i_idx, idx in enumerate(indices) if i_idx != ax1]
                tensor_indices_map[tname] = new_indices

        # Ellipsis broadcast check
        try:
            target_batch = self._check_ellipsis_broadcastable(
                tensor_map, tensor_indices_map
            )
            log("  [Lower] Ellipsis broadcast target:", target_batch)
        except RuntimeError as e:
            raise RuntimeError(f"Einsum lowering error: {e}")

        # Try matmul fast-path detection
        matmul_info = self._detect_matmul_symbolic()
        if matmul_info:
            # Two-tensor batched matmul lowering
            A_sym, B_sym = self.graph.tensors[0], self.graph.tensors[1]
            A_tensor = tensor_map[A_sym.name]
            B_tensor = tensor_map[B_sym.name]
            A_indices = tensor_indices_map[A_sym.name]
            B_indices = tensor_indices_map[B_sym.name]

            def names(indices_list):
                return [idx.name for idx in indices_list]

            batch = matmul_info["batch"]
            k = matmul_info["contracted"]
            a_m = matmul_info["A_m"]
            b_n = matmul_info["B_n"]

            desired_A = batch + [a_m, k]
            desired_B = batch + [k, b_n]

            def prepare_for_matmul(_tname, t_tensor, t_indices, desired):
                curr_names = names(t_indices)
                # permute existing axes to match desired where possible
                perm = [curr_names.index(n) for n in desired if n in curr_names]
                perm_rest = [i for i in range(len(curr_names)) if i not in perm]
                perm_full = perm + perm_rest
                if perm_full != list(range(len(curr_names))):
                    tnode = TransposeOp.create(
                        t_tensor,
                        perm_full,
                        indices_in=curr_names,
                        indices_out=[curr_names[p] for p in perm_full],
                    )
                    ops.append(tnode)
                    log("  [Lower] TransposeOp (matmul prep):", tnode.describe())
                    t_tensor = tnode.output
                    t_indices = [t_indices[i] for i in perm_full]
                    curr_names = names(t_indices)
                # insert missing axes as needed
                for idx_pos, name in enumerate(desired):
                    if name not in curr_names:
                        enode = ExpandDimsOp.create(
                            t_tensor,
                            dim=idx_pos,
                            indices_in=curr_names,
                            indices_out=curr_names[:idx_pos]
                            + [name]
                            + curr_names[idx_pos:],
                        )
                        ops.append(enode)
                        log("  [Lower] ExpandDimsOp (matmul prep):", enode.describe())
                        t_tensor = enode.output
                        inserted_size = (
                            1
                            if (self.graph.ellipsis and name in self.graph.ellipsis)
                            else self.graph.indices[name].size
                        )
                        t_indices = (
                            t_indices[:idx_pos]
                            + [Index(name, inserted_size)]
                            + t_indices[idx_pos:]
                        )
                        curr_names = names(t_indices)
                return t_tensor, t_indices

            prepA_tensor, prepA_indices = prepare_for_matmul(
                A_sym.name, A_tensor, A_indices, desired_A
            )
            prepB_tensor, prepB_indices = prepare_for_matmul(
                B_sym.name, B_tensor, B_indices, desired_B
            )

            batch_shape_A = (
                tuple(to_raw(prepA_tensor.data).shape[: len(batch)])
                if len(batch) > 0
                else ()
            )
            batch_shape_B = (
                tuple(to_raw(prepB_tensor.data).shape[: len(batch)])
                if len(batch) > 0
                else ()
            )

            def broadcast_shape(s1, s2):
                if not s1:
                    return s2
                if not s2:
                    return s1
                if len(s1) != len(s2):
                    if len(s1) < len(s2):
                        s1 = (1,) * (len(s2) - len(s1)) + s1
                    else:
                        s2 = (1,) * (len(s1) - len(s2)) + s2
                out = []
                for a, b in zip(s1, s2):
                    if a == 1:
                        out.append(b)
                    elif b == 1:
                        out.append(a)
                    elif a == b:
                        out.append(a)
                    else:
                        raise ValueError(f"Cannot broadcast batch dims {s1} and {s2}")
                return tuple(out)

            try:
                target_batch_shape = broadcast_shape(batch_shape_A, batch_shape_B)
            except Exception as e:
                raise ValueError(f"Einsum matmul batch broadcast error: {e}")

            A_M = self.graph.indices[a_m].size
            K_size = self.graph.indices[k].size
            B_N = self.graph.indices[b_n].size
            target_A_shape = target_batch_shape + (A_M, K_size)
            target_B_shape = target_batch_shape + (K_size, B_N)

            if tuple(to_raw(prepA_tensor.data).shape) != tuple(target_A_shape):
                bA = BroadcastOp.create(
                    prepA_tensor,
                    target_A_shape,
                    indices_in=[idx.name for idx in prepA_indices],
                    indices_out=[*batch, a_m, k],
                )
                ops.append(bA)
                log("  [Lower] BroadcastOp (matmul A):", bA.describe())
                prepA_tensor = bA.output
            if tuple(to_raw(prepB_tensor.data).shape) != tuple(target_B_shape):
                bB = BroadcastOp.create(
                    prepB_tensor,
                    target_B_shape,
                    indices_in=[idx.name for idx in prepB_indices],
                    indices_out=[*batch, k, b_n],
                )
                ops.append(bB)
                log("  [Lower] BroadcastOp (matmul B):", bB.describe())
                prepB_tensor = bB.output

            mnode = MatMulOp.create(
                prepA_tensor,
                prepB_tensor,
                left_indices=desired_A,
                right_indices=desired_B,
                out_indices=[idx.name for idx in self.graph.output_indices],
            )
            ops.append(mnode)
            log("  [Lower] MatMulOp:", mnode.describe())
            out_tensor = mnode.output

            current_out_names = batch + [a_m, b_n]
            desired_out_names = [idx.name for idx in self.graph.output_indices]
            if current_out_names != desired_out_names:
                perm = [current_out_names.index(n) for n in desired_out_names]
                tnode = TransposeOp.create(
                    out_tensor,
                    perm,
                    indices_in=current_out_names,
                    indices_out=desired_out_names,
                )
                ops.append(tnode)
                log("  [Lower] TransposeOp (final):", tnode.describe())
                out_tensor = tnode.output

            out_tensor.name = "out"
            if len(ops) == 0:
                inode = IdentityOp.create(out_tensor, out_name="out")
                ops.append(inode)
                out_tensor = inode.output

            # attach weak refs for callers who want to reuse these ops
            try:
                out_tensor._einsum_node_output_refs = [
                    weakref.ref(node.output) for node in ops
                ]
            except Exception:
                pass
            return ops, out_tensor

        # Fallback path: align all tensors to a global order, broadcast, multiply, and reduce
        output_idxs = list(self.graph.output_indices)
        remaining: List[Index] = []
        if self.graph.ellipsis:
            for name in self.graph.ellipsis:
                if (
                    name in self.graph.indices
                    and self.graph.indices[name] not in output_idxs
                ):
                    remaining.append(self.graph.indices[name])
        for k, idx in self.graph.indices.items():
            if idx not in output_idxs and (
                not self.graph.ellipsis or idx.name not in self.graph.ellipsis
            ):
                remaining.append(idx)
        global_order = output_idxs + remaining

        aligned_tensors: List[Tensor] = []
        aligned_indices_map: Dict[str, List[Index]] = {}

        for t in self.graph.tensors:
            name = t.name
            orig = tensor_map[name]
            indices = list(tensor_indices_map[name])
            curr_tensor = orig
            for j, gidx in enumerate(global_order):
                if gidx in indices:
                    cur_pos = indices.index(gidx)
                    if cur_pos != j:
                        n = len(indices)
                        perm = list(range(n))
                        axis = perm.pop(cur_pos)
                        perm.insert(j, axis)
                        tnode = TransposeOp.create(
                            curr_tensor,
                            perm,
                            indices_in=[i.name for i in indices],
                            indices_out=[i.name for i in indices],
                        )
                        ops.append(tnode)
                        log("  [Lower] TransposeOp:", tnode.describe())
                        curr_tensor = tnode.output
                        idx_obj = indices.pop(cur_pos)
                        indices.insert(j, idx_obj)
                else:
                    enode = ExpandDimsOp.create(
                        curr_tensor,
                        dim=j,
                        indices_in=[i.name for i in indices],
                        indices_out=[(i.name if i else None) for i in indices],
                    )
                    ops.append(enode)
                    log("  [Lower] ExpandDimsOp:", enode.describe())
                    curr_tensor = enode.output
                    if self.graph.ellipsis and gidx.name in self.graph.ellipsis:
                        indices.insert(j, Index(gidx.name, 1))
                    else:
                        indices.insert(j, gidx)
            # determine target shape considering ellipsis broadcasting
            target_shape_list = []
            for idx in indices:
                if self.graph.ellipsis and idx.name in (self.graph.ellipsis or []):
                    ell_pos = self.graph.ellipsis.index(idx.name)
                    if len(target_batch) == 0:
                        size = idx.size
                    else:
                        if ell_pos < len(target_batch):
                            size = target_batch[ell_pos]
                        else:
                            size = idx.size
                else:
                    size = idx.size
                target_shape_list.append(size)
            target_shape = tuple(target_shape_list)

            if tuple(to_raw(curr_tensor.data).shape) != tuple(target_shape):

                def can_broadcast(from_shape, to_shape):
                    if len(from_shape) > len(to_shape):
                        return False
                    from_r = (1,) * (len(to_shape) - len(from_shape)) + from_shape
                    for a, b in zip(from_r, to_shape):
                        if a == b or a == 1:
                            continue
                        else:
                            return False
                    return True

                if not can_broadcast(to_raw(curr_tensor.data).shape, target_shape):
                    raise ValueError(
                        f"Einsum lowering error: cannot broadcast tensor {name} shape {to_raw(curr_tensor.data).shape} to target {target_shape}"
                    )
                bnode = BroadcastOp.create(
                    curr_tensor,
                    target_shape,
                    indices_in=[i.name for i in tensor_indices_map[name]],
                    indices_out=[i.name for i in indices],
                )
                ops.append(bnode)
                log("  [Lower] BroadcastOp:", bnode.describe())
                curr_tensor = bnode.output
            aligned_tensors.append(curr_tensor)
            aligned_indices_map[name] = indices

        if len(aligned_tensors) == 0:
            out_tensor = Tensor(to_ap(np.array(0.0)), requires_grad=False)
            out_tensor.name = "out"
            inode = IdentityOp.create(out_tensor, out_name="out")
            ops.append(inode)
            out_tensor = inode.output
            try:
                out_tensor._einsum_node_output_refs = [
                    weakref.ref(node.output) for node in ops
                ]
            except Exception:
                pass
            return ops, out_tensor

        if len(aligned_tensors) == 1:
            current = aligned_tensors[0]
        else:
            inputs_list = aligned_tensors
            indices_meta = [
                [i.name for i in aligned_indices_map[self.graph.tensors[k].name]]
                for k in range(len(self.graph.tensors))
            ]
            mmnode = MultiMulOp.create(inputs_list, indices=indices_meta)
            ops.append(mmnode)
            log("  [Lower] MultiMulOp:", mmnode.describe())
            current = mmnode.output

        reduction_positions: List[int] = []
        reduction_names = set(idx.name for idx in self.graph.reduction_indices)
        for pos, idx in enumerate(global_order):
            if idx.name in reduction_names:
                reduction_positions.append(pos)

        if reduction_positions:
            rnode = ReduceSumOp.create(
                current,
                dim=tuple(reduction_positions),
                keepdims=False,
                indices_in=[global_order[p].name for p in range(len(global_order))],
                indices_out=[o.name for o in self.graph.output_indices],
            )
            ops.append(rnode)
            log("  [Lower] ReduceSumOp:", rnode.describe())
            out_tensor = rnode.output
        else:
            out_tensor = current

        # Guarantee final output is produced by an op node
        if not any(node.output is out_tensor for node in ops):
            inode = IdentityOp.create(out_tensor, out_name="out")
            ops.append(inode)
            out_tensor = inode.output

        out_tensor.name = "out"
        try:
            out_tensor._einsum_node_output_refs = [
                weakref.ref(node.output) for node in ops
            ]
        except Exception:
            pass

        log("\n[Lowering summary]")
        for i, node in enumerate(ops):
            log(f" op[{i}] {node.describe()}")
        log("-" * 40)
        return ops, out_tensor

    def _detect_matmul_symbolic(self):
        """
        Detect if the symbolic graph represents a batched matmul of two tensors.
        If so, return {'contracted', 'batch', 'A_m', 'B_n'}; otherwise return None.
        """
        if len(self.graph.tensors) != 2:
            return None
        A = self.graph.tensors[0]
        B = self.graph.tensors[1]
        out_names = [idx.name for idx in self.graph.output_indices]
        A_names = [idx.name for idx in A.indices]
        B_names = [idx.name for idx in B.indices]

        common = set(A_names) & set(B_names)
        contracted = [n for n in common if n not in out_names]
        if len(contracted) != 1:
            return None
        batch = [n for n in common if n in out_names]
        A_out_nonbatch = [n for n in A_names if n in out_names and n not in batch]
        B_out_nonbatch = [n for n in B_names if n in out_names and n not in batch]
        if len(A_out_nonbatch) != 1 or len(B_out_nonbatch) != 1:
            return None
        return {
            "contracted": contracted[0],
            "batch": batch,
            "A_m": A_out_nonbatch[0],
            "B_n": B_out_nonbatch[0],
        }


# =========================
# Backward Executor (Tensor-only traversal)
# =========================

def accumulate_grad(t: Tensor, grad: Tensor):
    """
    Accumulate a gradient Tensor into t.grad (Tensor).

    This function assumes `grad` is a Tensor. If a non-Tensor is provided,
    the caller should convert it prior to calling. Here we accept either and
    convert non-Tensor to Tensor with requires_grad=False for safety.
    """
    from .tensor import Tensor as _Tensor  # local import to avoid cycles

    if not isinstance(grad, _Tensor):
        # best-effort convert raw gradient to Tensor (no graph preserved)
        grad = _Tensor(to_ap(grad), requires_grad=False)

    if not getattr(t, "requires_grad", False):
        return

    if getattr(t, "grad", None) is None:
        t.grad = grad
    else:
        # both should be Tensor (convert if needed)
        existing = t.grad
        if not isinstance(existing, _Tensor):
            existing = _Tensor(existing, requires_grad=False)
        t.grad = existing + grad


class BackwardExecutor:
    """
    Tensor-only backward executor: expects grad_out to be a Tensor.
    Calls OpNode.backward with Tensor and expects list[Tensor] returned.
    """

    def __init__(self, ops: List[OpNode], output: Tensor):
        self.ops = ops
        self.output = output

    def _ensure_tensor(self, x) -> Tensor:
        """Ensure x is a Tensor; if not convert to Tensor with requires_grad=False."""
        if isinstance(x, Tensor):
            return x
        return Tensor(to_ap(x), requires_grad=False)

    def backward(self, grad_out):
        """
        Run backward assuming Tensor-mode. If grad_out is not a Tensor,
        convert it to a Tensor(with requires_grad=False) but *do not* preserve graph.
        """
        # Ensure grad_out is a Tensor
        if not isinstance(grad_out, Tensor):
            grad_out = self._ensure_tensor(grad_out)

        # initialize output.grad
        if getattr(self.output, "grad", None) is None:
            accumulate_grad(self.output, grad_out)

        # traverse ops in reverse topological order; each node.backward must accept Tensor
        for node in reversed(self.ops):
            out_t = node.output
            if getattr(out_t, "grad", None) is None:
                continue
            out_grad = out_t.grad
            # ensure out_grad is Tensor
            if not isinstance(out_grad, Tensor):
                out_grad = self._ensure_tensor(out_grad)
            # call node.backward with Tensor and expect Tensor grads back
            in_grads = node.backward(out_grad)
            # be robust: if node.backward returned raw arrays, convert them
            processed_in_grads: List[Tensor] = []
            for g in in_grads:
                if isinstance(g, Tensor):
                    processed_in_grads.append(g)
                else:
                    processed_in_grads.append(self._ensure_tensor(g))
            for inp, gin in zip(node.inputs, processed_in_grads):
                accumulate_grad(inp, gin)

        # collect grads dict for convenience (values will be Tensor)
        grads: Dict[str, Tensor] = {}
        for node in self.ops:
            for inp in node.inputs:
                if getattr(inp, "grad", None) is not None:
                    grads[inp.name] = inp.grad
        if getattr(self.output, "grad", None) is not None:
            grads[self.output.name] = self.output.grad
        return grads


# =========================
# High-level einsum (user-facing)
# =========================


def einsum(equation: str, *operands):
    """
    einsum(equation, *operands) -> Tensor

    Tries a fast-path using xp.einsum (numpy/cupy). If not possible (or if
    backward is required), falls back to symbolic lowering into OpNodes.

    Returned Tensor will have requires_grad=True if any input Tensor had
    requires_grad=True.
    """
    arrays = []
    orig_map: Dict[str, Tensor] = {}
    any_requires = False

    for i, op in enumerate(operands):
        if isinstance(op, Tensor):
            arrays.append(to_raw(op.data))
            orig_map[f"T{i}"] = op
            any_requires = any_requires or getattr(op, "requires_grad", False)
        else:
            if isinstance(op, ap.Array):
                arrays.append(to_raw(op))
            else:
                arrays.append(
                    op
                    if (
                        isinstance(op, np.ndarray)
                        or (cp is not None and isinstance(op, cp.ndarray))
                    )
                    else np.asarray(op)
                )

    # Fast forward path: try xp.einsum for the forward result
    try:
        xp = get_xp_for(*arrays)
        raw_arrays = [
            a
            if (
                isinstance(a, np.ndarray)
                or (cp is not None and isinstance(a, cp.ndarray))
            )
            else np.asarray(a)
            for a in arrays
        ]
        try:
            res_raw = xp.einsum(equation, *raw_arrays)
        except Exception:
            # fallback: ensure arrays are raw backend arrays
            raw_arrays2 = [to_raw(a) for a in raw_arrays]
            res_raw = xp.einsum(equation, *raw_arrays2)
        out_tensor = Tensor(to_ap(res_raw), requires_grad=any_requires)
        out_tensor.name = "out"
    except Exception:
        # Fast path failed: fallback to symbolic builder/lowerer approach
        operands_for_builder = []
        for a in arrays:
            if isinstance(a, ap.Array):
                operands_for_builder.append(to_raw(a))
            else:
                operands_for_builder.append(a)
        builder = EinsumSymbolicBuilder(equation, list(operands_for_builder))
        graph = builder.build()
        lowerer = SymbolicLowerer(graph, list(arrays), orig_tensors=orig_map)
        ops, out_tensor = lowerer.lower()

    # If any input requires grad, attach backward wrapper based on lowered ops.
    if any_requires:
        try:
            runtime_ops: List[OpNode] = []
            # Prefer runtime ops extracted from the lowered output if available
            if hasattr(out_tensor, "_einsum_node_output_refs"):
                node_output_refs = out_tensor._einsum_node_output_refs
                for out_ref in node_output_refs:
                    out_obj = out_ref()
                    if out_obj is None:
                        continue
                    node = getattr(out_obj, "op", None)
                    if node is None:
                        continue
                    runtime_ops.append(node)
            else:
                # Build runtime ops now using original operand arrays
                arrays_for_builder = []
                for i, op in enumerate(operands):
                    if isinstance(op, Tensor):
                        arrays_for_builder.append(to_raw(op.data))
                    else:
                        arrays_for_builder.append(
                            op
                            if (
                                isinstance(op, np.ndarray)
                                or (cp is not None and isinstance(op, cp.ndarray))
                            )
                            else np.asarray(op)
                        )
                builder = EinsumSymbolicBuilder(equation, list(arrays_for_builder))
                graph = builder.build()
                lowerer = SymbolicLowerer(
                    graph, arrays_for_builder, orig_tensors=orig_map
                )
                runtime_ops, out_tensor_local = lowerer.lower()
                out_tensor = out_tensor_local
            node_output_refs = [weakref.ref(node.output) for node in runtime_ops]
        except Exception:
            # Could not build backward graph — attach a friendly placeholder
            def _backward_unavailable(in_grad):
                raise RuntimeError(
                    "Backward for einsum not available for these operands/equation in this environment."
                )

            try:
                out_tensor.requires_grad = True
            except Exception:
                pass
            try:
                out_tensor.grad_fn = _backward_unavailable
            except Exception:
                try:
                    out_tensor._grad_fn = _backward_unavailable
                except Exception:
                    pass
            out_tensor.grad_fn_name = "<EinsumBackwardUnavailable>"
            parents = tuple(orig_map.values())
            if parents:
                try:
                    out_tensor._add_parents(*parents)
                except Exception:
                    try:
                        out_tensor._parents = parents
                    except Exception:
                        pass
            return out_tensor

        def _backward_wrapper(in_grad):
            """
            Wrap incoming grad into Tensor (if needed) and run Tensor-only BackwardExecutor.
            - If incoming in_grad is not a Tensor, it will be converted to Tensor with requires_grad=False.
            """
            runtime_ops_live: List[OpNode] = []
            for out_ref in node_output_refs:
                out_obj = out_ref()
                if out_obj is None:
                    continue
                node = getattr(out_obj, "op", None)
                if node is None:
                    continue
                runtime_ops_live.append(node)
            be = BackwardExecutor(runtime_ops_live, out_tensor)
            be.backward(in_grad)
            del runtime_ops_live

        # attach backward metadata to out_tensor robustly
        try:
            out_tensor.requires_grad = True
        except Exception:
            pass
        try:
            out_tensor.grad_fn = _backward_wrapper
        except Exception:
            try:
                out_tensor._grad_fn = _backward_wrapper
            except Exception:
                out_tensor.__dict__["grad_fn"] = _backward_wrapper
        try:
            out_tensor.grad_fn_name = "<EinsumBackward>"
        except Exception:
            try:
                out_tensor.__dict__["grad_fn_name"] = "<EinsumBackward>"
            except Exception:
                pass

        parents = tuple(orig_map.values())
        if parents:
            try:
                out_tensor._add_parents(*parents)
            except Exception:
                try:
                    out_tensor._parents = parents
                except Exception:
                    pass

    # store optional debug handles
    try:
        out_tensor._symbolic_graph = locals().get("graph", None)
    except Exception:
        pass

    return out_tensor


__all__ = ["einsum"]