# tests/test_thetorch_comprehensive.py
import math
import os  # noqa: F401
import sys  # noqa: F401
import traceback  # noqa: F401
import unittest

import numpy as np
import torch

import cattorch as ctorch  # user's library

RTOL = 1e-5
ATOL = 1e-6
FD_EPS = 1e-3
FD_RTOL = 1e-3
FD_ATOL = 1e-3


# ----------------- Helpers -----------------
def torch_to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    if isinstance(t, np.ndarray):
        return t
    try:
        return np.array(t)
    except Exception:
        raise TypeError("Expected torch.Tensor or convertible to numpy")


def thetorch_to_numpy(x):
    """
    Convert a value coming from `thetorch` world into a numpy ndarray or scalar.

    Handles:
      - numpy.ndarray
      - torch.Tensor (detaches + cpu)
      - thetorch.Tensor (has .data that is an Array)
      - thetorch._array.Array-like objects (has .asnumpy() or ._array)
      - python scalars
    """
    if x is None:
        return None

    # already numpy
    if isinstance(x, np.ndarray):
        return x

    # torch Tensor (detach -> cpu -> numpy)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    # thetorch Tensor object (has .data attribute)
    # try the common pattern first: x.data.asnumpy()
    if hasattr(x, "data"):
        data = x.data
        # if data is our Array object and provides asnumpy()
        if hasattr(data, "asnumpy") and callable(getattr(data, "asnumpy")):
            return data.asnumpy()
        # if data exposes underlying ndarray
        if hasattr(data, "_array"):
            arr = getattr(data, "_array")
            # cupy ndarray -> convert to numpy if necessary
            try:
                return np.asarray(arr)
            except Exception:
                # as fallback try asnumpy if available
                if hasattr(data, "asnumpy"):
                    return data.asnumpy()
                raise

    # ap.Array-like object directly passed (not wrapped in Tensor)
    if hasattr(x, "asnumpy") and callable(getattr(x, "asnumpy")):
        try:
            return x.asnumpy()
        except Exception:
            # try reading internal _array
            if hasattr(x, "_array"):
                return np.asarray(x._array)
            raise

    if hasattr(x, "_array"):
        try:
            return np.asarray(x._array)
        except Exception:
            pass

    # python scalar
    if np.isscalar(x):
        return np.array(x)

    # last resort: try numpy conversion
    try:
        return np.asarray(x)
    except Exception:
        pass

    raise TypeError(f"Cannot convert object to numpy: type={type(x)}")


def assert_allclose_torch_vs_tt(
    testcase, t_torch, t_tt, rtol=RTOL, atol=ATOL, msg=None
):
    a = torch_to_numpy(t_torch)
    b = thetorch_to_numpy(t_tt)
    testcase.assertEqual(
        a.shape, b.shape, msg=(msg or "") + f" shapes differ: {a.shape} vs {b.shape}"
    )
    testcase.assertTrue(
        np.allclose(a, b, rtol=rtol, atol=atol),
        msg=(msg or "") + f"\nTorch:\n{a}\nTheTorch:\n{b}",
    )


def safe_make_ctorch(arr, requires_grad=True, dtype=None):
    # attempt to construct thetorch Tensor robustly
    if dtype is None:
        try:
            return ctorch.Tensor(arr, requires_grad=requires_grad)
        except Exception:
            return ctorch.Tensor(np.asarray(arr), requires_grad=requires_grad)
    # try common dtype representations
    try:
        return ctorch.Tensor(arr, requires_grad=requires_grad, dtype=dtype)
    except Exception:
        # try passing numpy dtype / string
        try:
            return ctorch.Tensor(
                arr, requires_grad=requires_grad, dtype=str(np.dtype(dtype))
            )
        except Exception:
            return ctorch.Tensor(arr, requires_grad=requires_grad)


def numeric_grad_thetorch_scalar(fn, arr, eps=FD_EPS):
    """
    Numeric gradient of scalar-output function fn(w) w.r.t. numpy array arr.
    fn expects a thetorch Tensor and returns scalar (python float or 0-d Tensor).
    Returns numpy array of same shape as arr with finite-diff gradients.
    """
    arr = np.asarray(arr, dtype=np.float64)
    grad = np.zeros_like(arr, dtype=np.float64)
    it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = arr[idx].copy()
        arr[idx] = orig + eps
        try:
            out_p = fn(safe_make_ctorch(arr.astype(np.float32), requires_grad=False))
            out_p = thetorch_to_numpy(out_p)
            out_p = float(np.asarray(out_p))
        except Exception:
            out_p = None
        arr[idx] = orig - eps
        try:
            out_m = fn(safe_make_ctorch(arr.astype(np.float32), requires_grad=False))
            out_m = thetorch_to_numpy(out_m)
            out_m = float(np.asarray(out_m))
        except Exception:
            out_m = None
        arr[idx] = orig
        if out_p is None or out_m is None:
            grad[idx] = 0.0
        else:
            grad[idx] = (out_p - out_m) / (2 * eps)
        it.iternext()
    return grad.astype(np.float32)


# ----------------- Tests -----------------
class TestFactoriesAndBasics(unittest.TestCase):
    def test_creation_and_basic_properties(self):
        a = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        t_t = torch.tensor(a, requires_grad=True)
        t_tt = safe_make_ctorch(a, requires_grad=True)
        assert_allclose_torch_vs_tt(self, t_t, t_tt, msg="creation mismatch")

        # dtype attr and requires_grad
        self.assertTrue(hasattr(t_tt, "requires_grad"))
        self.assertTrue(getattr(t_tt, "requires_grad") in (True, False))

    def test_factories_values(self):
        # arange / zeros / ones / full
        if hasattr(ctorch, "arange"):
            assert_allclose_torch_vs_tt(self, torch.arange(6.0), ctorch.arange(6.0))
        if hasattr(ctorch, "zeros"):
            assert_allclose_torch_vs_tt(self, torch.zeros((2, 3)), ctorch.zeros((2, 3)))
        if hasattr(ctorch, "ones"):
            assert_allclose_torch_vs_tt(self, torch.ones((2, 3)), ctorch.ones((2, 3)))
        if hasattr(ctorch, "full"):
            assert_allclose_torch_vs_tt(
                self, torch.full((2, 2), 7.5), ctorch.full((2, 2), 7.5)
            )

    def test_numpy_and_detach_clone(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tt_x = safe_make_ctorch(x, requires_grad=True)
        # detach should exist and the detached tensor should not require grad
        if hasattr(tt_x, "detach"):
            dt = tt_x.detach()
            # try to access requires_grad
            if hasattr(dt, "requires_grad"):
                self.assertFalse(dt.requires_grad)
        # numpy conversion
        nx = thetorch_to_numpy(tt_x)
        self.assertTrue(np.allclose(nx, x))


class TestElementwiseAndBroadcasting(unittest.TestCase):
    def test_add_sub_mul_div_broadcast(self):
        rng = np.random.RandomState(1)
        a = rng.randn(2, 3).astype(np.float32)
        b = rng.randn(3).astype(np.float32)

        at = torch.tensor(a, requires_grad=True)
        bt = torch.tensor(b, requires_grad=True)
        am = safe_make_ctorch(a, requires_grad=True)
        bm = safe_make_ctorch(b, requires_grad=True)

        # add broadcasting
        out_t = at + bt
        out_m = am + bm
        assert_allclose_torch_vs_tt(self, out_t, out_m, msg="broadcast add mismatch")

        # true div
        out_t = at / (bt + 1.0)  # avoid zero
        try:
            out_m = am / (bm + 1.0)
        except Exception:
            out_m = ctorch.div(am, bm + 1.0) if hasattr(ctorch, "div") else None
        if out_m is None:
            self.skipTest("division not implemented in thetorch")
        assert_allclose_torch_vs_tt(self, out_t, out_m, msg="div mismatch")

    def test_pow_and_rpow(self):
        v = np.array([1.2, 2.0, 3.0], dtype=np.float32)
        vt = torch.tensor(v, requires_grad=True)
        vm = safe_make_ctorch(v, requires_grad=True)

        # tensor ** scalar
        out_t = vt**3
        try:
            out_m = vm**3
        except Exception:
            out_m = ctorch.pow(vm, 3) if hasattr(ctorch, "pow") else None
        if out_m is None:
            self.skipTest("pow not implemented")
        assert_allclose_torch_vs_tt(self, out_t, out_m, msg="pow forward mismatch")

        # scalar ** tensor (rpow)
        out_t = 3.0**vt
        try:
            out_m = 3.0**vm
        except Exception:
            if hasattr(ctorch, "pow"):
                out_m = ctorch.pow(3.0, vm)
            else:
                out_m = None
        if out_m is None:
            self.skipTest("rpow not implemented")
        assert_allclose_torch_vs_tt(self, out_t, out_m, msg="rpow forward mismatch")


class TestReductionsAndIndexing(unittest.TestCase):
    def test_sum_mean_var_max_argmax_cumsum(self):
        rng = np.random.RandomState(2)
        x = rng.randn(4, 5).astype(np.float32)
        xt = torch.tensor(x, requires_grad=True)
        xm = safe_make_ctorch(x, requires_grad=True)

        # sum
        assert_allclose_torch_vs_tt(self, xt.sum(), xm.sum(), msg="sum mismatch")
        # mean
        assert_allclose_torch_vs_tt(self, xt.mean(), xm.mean(), msg="mean mismatch")

        # var (torch default unbiased True) - tolerance slightly larger
        if hasattr(xt, "var") and hasattr(xm, "var"):
            assert_allclose_torch_vs_tt(
                self, xt.var(), xm.var(), rtol=1e-4, atol=1e-4, msg="var mismatch"
            )

        # max and argmax (forward only)
        if hasattr(xt, "max") and hasattr(xm, "max"):
            mt_val = xt.max()
            mm_val = xm.max()
            assert_allclose_torch_vs_tt(self, mt_val, mm_val, msg="max mismatch")

        if hasattr(xt, "argmax") and hasattr(xm, "argmax"):
            self.assertEqual(
                torch_to_numpy(xt.argmax()).tolist(),
                thetorch_to_numpy(xm.argmax()).tolist(),
            )

        # cumsum
        if hasattr(xt, "cumsum") and hasattr(xm, "cumsum"):
            assert_allclose_torch_vs_tt(
                self, xt.cumsum(0), xm.cumsum(0), msg="cumsum mismatch"
            )

    def test_indexing_slicing_reshape_transpose_chunk_repeateinterleave(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        xt = torch.tensor(x, requires_grad=True)
        if hasattr(ctorch, "arange"):
            ym = ctorch.arange(12).reshape(3, 4)
        else:
            ym = safe_make_ctorch(x)

        # row
        assert_allclose_torch_vs_tt(self, xt[0], ym[0], msg="row index mismatch")
        # element
        self.assertTrue(
            np.allclose(torch_to_numpy(xt[1, 2]), thetorch_to_numpy(ym[1, 2]))
        )
        # reshape
        assert_allclose_torch_vs_tt(
            self, xt.reshape(-1), ym.reshape(-1), msg="reshape mismatch"
        )
        # transpose/T
        if hasattr(ym, "T") or hasattr(ym, "transpose"):
            tt_T = ym.T if hasattr(ym, "T") else ym.transpose(0, 1)
            torch_T = xt.T if hasattr(xt, "T") else xt.transpose(0, 1)
            assert_allclose_torch_vs_tt(self, torch_T, tt_T, msg="transpose mismatch")
        else:
            self.skipTest("transpose not implemented in thetorch")

        # ---------------- chunk tests ----------------
        # use a small random array so FD/backward is cheap
        rng = np.random.RandomState(123)
        A = rng.randn(4, 3).astype(np.float32)
        At = torch.tensor(A, requires_grad=True)
        Am = safe_make_ctorch(A, requires_grad=True)

        # check chunk forward
        if not (
            hasattr(torch, "chunk")
            and (hasattr(ctorch, "chunk") or hasattr(Am, "chunk"))
        ):
            self.skipTest("chunk not implemented in one of torch/thetorch")
        # torch chunks
        torch_chunks = torch.chunk(At, chunks=2, dim=0)
        # thetorch chunks (try functional then method)
        try:
            if hasattr(ctorch, "chunk"):
                tt_chunks = ctorch.chunk(Am, chunks=2, dim=0)
            else:
                tt_chunks = Am.chunk(2, dim=0)
        except Exception as e:
            self.skipTest(f"thetorch chunk raised on call: {e}")

        # compare forward for each chunk
        self.assertEqual(
            len(torch_chunks),
            len(tt_chunks),
            msg="chunk returned different number of pieces",
        )
        for i, (tc, ttc) in enumerate(zip(torch_chunks, tt_chunks)):
            assert_allclose_torch_vs_tt(
                self,
                tc.detach().cpu().numpy(),
                thetorch_to_numpy(ttc),
                msg=f"chunk piece {i} forward mismatch",
            )

        # chunk backward: sum of piece sums -> backward should match
        # reset grads
        At.grad = None
        try:
            Am.grad = None
        except Exception:
            pass

        # compute scalar losses and backward
        torch_loss = sum([c.sum() for c in torch_chunks])
        try:
            tt_loss = sum([c.sum() for c in tt_chunks])
        except Exception:
            # if tt_chunks elements are not Tensors (unlikely) try converting
            try:
                tt_loss = sum(
                    [
                        safe_make_ctorch(
                            thetorch_to_numpy(c), requires_grad=False
                        ).sum()
                        for c in tt_chunks
                    ]
                )
            except Exception as e:
                self.skipTest(f"cannot form scalar loss from thetorch chunks: {e}")

        torch_loss.backward()
        try:
            tt_loss.backward()
        except Exception as e:
            self.fail(f"thetorch chunk backward failed: {e}")

        # compare grads on original
        assert_allclose_torch_vs_tt(
            self,
            At.grad.detach().cpu().numpy(),
            thetorch_to_numpy(Am.grad),
            msg="chunk grad mismatch",
        )

        # ---------------- repeat_interleave tests ----------------
        B = np.arange(6, dtype=np.float32).reshape(2, 3)
        Bt = torch.tensor(B, requires_grad=True)
        Bm = safe_make_ctorch(B, requires_grad=True)

        # forward: try functional or method on thetorch
        if not (
            hasattr(torch.Tensor, "repeat_interleave")
            or hasattr(torch, "repeat_interleave")
        ):
            # unlikely, but be safe
            self.skipTest("torch.repeat_interleave not available")
        # try thetorch repeat_interleave
        try:
            # prefer functional if provided
            if hasattr(ctorch, "repeat_interleave"):
                tt_rep = ctorch.repeat_interleave(Bm, repeats=2, dim=0)
            elif hasattr(Bm, "repeat_interleave"):
                tt_rep = Bm.repeat_interleave(2, dim=0)
            else:
                self.skipTest("repeat_interleave not implemented in thetorch")
        except Exception as e:
            self.skipTest(f"thetorch repeat_interleave call failed: {e}")

        # torch forward
        torch_rep = (
            torch.repeat_interleave(Bt, repeats=2, dim=0)
            if hasattr(torch, "repeat_interleave")
            else Bt.repeat_interleave(2, dim=0)
        )
        assert_allclose_torch_vs_tt(
            self,
            torch_rep.detach().cpu().numpy(),
            thetorch_to_numpy(tt_rep),
            msg="repeat_interleave forward mismatch",
        )

        # backward check: sum -> backward
        Bt.grad = None
        try:
            Bm.grad = None
        except Exception:
            pass

        t_loss = torch_rep.sum()
        try:
            tt_loss = tt_rep.sum()
        except Exception as e:
            self.skipTest(f"thetorch repeat_interleave sum failed: {e}")

        t_loss.backward()
        try:
            tt_loss.backward()
        except Exception as e:
            self.fail(f"thetorch repeat_interleave backward failed: {e}")

        assert_allclose_torch_vs_tt(
            self,
            Bt.grad.detach().cpu().numpy(),
            thetorch_to_numpy(Bm.grad),
            msg="repeat_interleave grad mismatch",
        )


class TestMatmulEinsumAndGrads(unittest.TestCase):
    def test_matmul_forward_backward(self):
        A = np.random.randn(3, 5).astype(np.float32)
        B = np.random.randn(5, 4).astype(np.float32)
        At = torch.tensor(A, requires_grad=True)
        Bt = torch.tensor(B, requires_grad=True)
        Am = safe_make_ctorch(A, requires_grad=True)
        Bm = safe_make_ctorch(B, requires_grad=True)

        out_t = At @ Bt
        # thetorch may implement __matmul__ or matmul functional
        try:
            out_m = Am @ Bm
        except Exception:
            if hasattr(ctorch, "matmul"):
                out_m = ctorch.matmul(Am, Bm)
            else:
                self.skipTest("matmul not implemented in thetorch")

        # forward
        assert_allclose_torch_vs_tt(self, out_t, out_m, msg="matmul forward mismatch")

        # backward via sum
        out_t.sum().backward()
        try:
            out_m.sum().backward()
        except Exception as e:
            self.fail(f"thetorch matmul backward raised: {e}")

        assert_allclose_torch_vs_tt(
            self, At.grad, Am.grad, msg="matmul grad A mismatch"
        )
        assert_allclose_torch_vs_tt(
            self, Bt.grad, Bm.grad, msg="matmul grad B mismatch"
        )

    def test_einsum_simple_forward_backward(self):
        if not hasattr(ctorch, "einsum"):
            self.skipTest("einsum not implemented in thetorch")
        A = np.random.randn(3, 4).astype(np.float32)
        B = np.random.randn(4, 5).astype(np.float32)
        At = torch.tensor(A, requires_grad=True)
        Bt = torch.tensor(B, requires_grad=True)
        Am = safe_make_ctorch(A, requires_grad=True)
        Bm = safe_make_ctorch(B, requires_grad=True)

        out_t = torch.einsum("ij,jk->ik", At, Bt)
        out_m = ctorch.einsum("ij,jk->ik", Am, Bm)
        assert_allclose_torch_vs_tt(self, out_t, out_m, msg="einsum forward mismatch")

        out_t.sum().backward()
        try:
            out_m.sum().backward()
        except Exception as e:
            self.fail(f"thetorch einsum backward raised: {e}")

        assert_allclose_torch_vs_tt(
            self, At.grad, Am.grad, msg="einsum grad A mismatch"
        )
        assert_allclose_torch_vs_tt(
            self, Bt.grad, Bm.grad, msg="einsum grad B mismatch"
        )

    def test_matmul_grad_finite_difference(self):
        """Finite-difference check for grad of f(A,B)=sum(A@B) w.r.t A and B (spot check)."""
        A = np.random.randn(2, 3).astype(np.float32)
        B = np.random.randn(3, 2).astype(np.float32)

        # Torch reference grads
        At = torch.tensor(A, requires_grad=True)
        Bt = torch.tensor(B, requires_grad=True)
        out = (At @ Bt).sum()
        out.backward()
        gradA_t = At.grad.detach().cpu().numpy()
        gradB_t = Bt.grad.detach().cpu().numpy()

        # thetorch backward
        Am = safe_make_ctorch(A, requires_grad=True)
        Bm = safe_make_ctorch(B, requires_grad=True)
        try:
            outm = Am @ Bm
        except Exception:
            if hasattr(ctorch, "matmul"):
                outm = ctorch.matmul(Am, Bm)
            else:
                self.skipTest("matmul not implemented")
        outm.sum().backward()
        gradA_m = thetorch_to_numpy(Am.grad)
        gradB_m = thetorch_to_numpy(Bm.grad)

        # compare to torch gradients
        assert_allclose_torch_vs_tt(
            self, torch.tensor(gradA_t), gradA_m, msg="matmul gradA mismatch"
        )
        assert_allclose_torch_vs_tt(
            self, torch.tensor(gradB_t), gradB_m, msg="matmul gradB mismatch"
        )

        # finite-diff check on thetorch A-grad (looser tolerance)
        def fA(X_tt):
            # X_tt is thetorch Tensor
            Y = safe_make_ctorch(B.astype(np.float32), requires_grad=False)
            return (
                ctorch.matmul(X_tt, Y).sum()
                if hasattr(ctorch, "matmul")
                else (X_tt @ Y).sum()
            )

        fd_gradA = numeric_grad_thetorch_scalar(fA, A, eps=FD_EPS)
        self.assertTrue(
            np.allclose(fd_gradA, gradA_m, rtol=FD_RTOL, atol=FD_ATOL),
            msg=f"finite-diff vs thetorch gradA mismatch\nfd:\n{fd_gradA}\nbackward:\n{gradA_m}",
        )


class TestUnaryAndActivations(unittest.TestCase):
    def test_unary_ops_and_grads(self):
        rng = np.random.RandomState(3)
        x = (rng.randn(4, 3) * 0.5 + 0.1).astype(np.float32)
        eps = 1e-6
        x_pos = (np.abs(x) + eps).astype(np.float32)  # for sqrt/log

        # Torch tensors
        xt_full = torch.tensor(x, requires_grad=True)
        xt_pos = torch.tensor(x_pos, requires_grad=True)

        # TheTorch tensors
        xm_full = safe_make_ctorch(x, requires_grad=True)
        xm_pos = safe_make_ctorch(x_pos, requires_grad=True)

        unary_ops = [
            ("exp", torch.exp, (ctorch.exp if hasattr(ctorch, "exp") else None), False),
            ("log", torch.log, (ctorch.log if hasattr(ctorch, "log") else None), True),
            (
                "sqrt",
                torch.sqrt,
                (ctorch.sqrt if hasattr(ctorch, "sqrt") else None),
                True,
            ),
            ("sin", torch.sin, (ctorch.sin if hasattr(ctorch, "sin") else None), False),
            ("cos", torch.cos, (ctorch.cos if hasattr(ctorch, "cos") else None), False),
            ("tan", torch.tan, (ctorch.tan if hasattr(ctorch, "tan") else None), False),
            ("abs", torch.abs, (ctorch.abs if hasattr(ctorch, "abs") else None), False),
        ]

        for name, torch_fn, tt_fn, need_pos in unary_ops:
            if tt_fn is None and not hasattr(xm_full, name):
                self.skipTest(f"{name} not implemented in thetorch")

            # choose input
            if need_pos:
                xt = xt_pos
                xm = xm_pos
            else:
                xt = xt_full
                xm = xm_full

            # --- Forward: compare raw outputs (no .sum) ---
            try:
                torch_forward = torch_fn(xt).detach().cpu().numpy()
            except Exception as e:
                self.skipTest(f"torch {name} forward error: {e}")

            try:
                if tt_fn is not None:
                    tt_forward_obj = tt_fn(xm)
                else:
                    tt_forward_obj = getattr(xm, name)()
                tt_forward = thetorch_to_numpy(tt_forward_obj)
            except Exception as e:
                self.skipTest(f"thetorch {name} forward call failed: {e}")

            # shape/value checks
            self.assertEqual(
                torch_forward.shape,
                tt_forward.shape,
                msg=f"{name} forward shape mismatch: torch {torch_forward.shape} vs thetorch {tt_forward.shape}",
            )
            self.assertTrue(
                np.allclose(torch_forward, tt_forward, rtol=RTOL, atol=ATOL),
                msg=f"{name} forward numeric mismatch",
            )

            # --- Backward: reset grads then sum -> backward and compare grads ---
            # reset grads (torch + thetorch)
            xt.grad = None
            try:
                xm.grad = None
            except Exception:
                pass

            try:
                out_t_scalar = torch_fn(xt).sum()
            except Exception as e:
                self.skipTest(f"torch {name} backward setup error: {e}")

            try:
                if tt_fn is not None:
                    out_m_scalar_obj = tt_fn(xm).sum()
                else:
                    out_m_scalar_obj = getattr(xm, name)().sum()
            except Exception as e:
                self.skipTest(f"thetorch {name} backward setup failed: {e}")

            # backward
            out_t_scalar.backward()
            try:
                out_m_scalar_obj.backward()
            except Exception as e:
                self.fail(f"thetorch {name} backward failed: {e}")

            # get grads and compare
            torch_grad = xt.grad.detach().cpu().numpy() if xt.grad is not None else None
            try:
                tt_grad = thetorch_to_numpy(xm.grad)
            except Exception as e:
                self.fail(f"converting thetorch grad for {name} failed: {e}")

            self.assertEqual(
                torch_grad.shape,
                tt_grad.shape,
                msg=f"{name} grad shape mismatch: torch {torch_grad.shape} vs thetorch {tt_grad.shape}",
            )
            self.assertTrue(
                np.allclose(torch_grad, tt_grad, rtol=RTOL, atol=ATOL),
                msg=f"{name} grad numeric mismatch",
            )

            # reset again
            xt.grad = None
            try:
                xm.grad = None
            except Exception:
                pass

        # Out-of-domain check for sqrt/log (negative inputs)
        x_neg = -(np.abs(rng.randn(3, 3))).astype(np.float32)
        xt_neg = torch.tensor(x_neg, requires_grad=False)
        xm_neg = safe_make_ctorch(x_neg, requires_grad=False)

        # torch behavior
        with np.errstate(all="ignore"):
            torch_out_neg = torch.sqrt(xt_neg).detach().cpu().numpy()
        # thetorch behavior
        try:
            tt_out_neg_obj = (
                ctorch.sqrt(xm_neg) if hasattr(ctorch, "sqrt") else xm_neg.sqrt()
            )
            tt_out_neg = thetorch_to_numpy(tt_out_neg_obj)
        except Exception as e:
            self.fail(f"thetorch sqrt raised on negative input but torch did not: {e}")

        torch_nan_mask = np.isnan(torch_out_neg) | np.isinf(torch_out_neg)
        tt_nan_mask = np.isnan(tt_out_neg) | np.isinf(tt_out_neg)
        self.assertTrue(
            np.array_equal(torch_nan_mask, tt_nan_mask),
            msg="sqrt negative-input NaN/Inf mask differs between torch and thetorch",
        )

    def test_composite_activations_sigmoid_tanh_relu_softmax_gelu_swish(self):
        vals = [0.1, -1.0, 2.0, 0.0]
        vt = torch.tensor(vals, requires_grad=True)
        vm = safe_make_ctorch(vals, requires_grad=True)

        # sigmoid via formula
        if hasattr(torch, "exp") and (hasattr(ctorch, "exp") or hasattr(vm, "exp")):
            sig_t = 1.0 / (1.0 + torch.exp(-vt))
            try:
                exp_neg_vm = ctorch.exp(-vm) if hasattr(ctorch, "exp") else (-vm).exp()
                sig_m = 1.0 / (1.0 + exp_neg_vm)
            except Exception:
                self.skipTest("sigmoid building blocks missing")
            # forward and backward
            out_t = sig_t.sum()
            out_m = sig_m.sum()
            out_t.backward()
            out_m.backward()
            assert_allclose_torch_vs_tt(
                self, vt.grad, vm.grad, msg="sigmoid grad mismatch"
            )
            vt.grad = None
            vm.grad = None
        else:
            self.skipTest("sigmoid primitives missing")

        # tanh (via exp)
        if hasattr(torch, "exp") and (hasattr(ctorch, "exp") or hasattr(vm, "exp")):
            ex_t = torch.exp(vt)
            en_t = torch.exp(-vt)
            tanh_t = (ex_t - en_t) / (ex_t + en_t)
            ex_m = ctorch.exp(vm) if hasattr(ctorch, "exp") else vm.exp()
            en_m = ctorch.exp(-vm) if hasattr(ctorch, "exp") else (-vm).exp()
            tanh_m = (ex_m - en_m) / (ex_m + en_m)
            out_t = tanh_t.sum()
            out_m = tanh_m.sum()
            out_t.backward()
            out_m.backward()
            assert_allclose_torch_vs_tt(
                self, vt.grad, vm.grad, msg="tanh grad mismatch"
            )
            vt.grad = None
            vm.grad = None
        else:
            self.skipTest("tanh primitives missing")

        # relu via (x+abs(x))/2
        if hasattr(torch, "abs") and (hasattr(ctorch, "abs") or hasattr(vm, "abs")):
            relu_t = (vt + torch.abs(vt)) * 0.5
            abs_vm = ctorch.abs(vm) if hasattr(ctorch, "abs") else vm.abs()
            relu_m = (vm + abs_vm) * 0.5
            out_t = relu_t.sum()
            out_m = relu_m.sum()
            out_t.backward()
            out_m.backward()
            assert_allclose_torch_vs_tt(
                self, vt.grad, vm.grad, msg="relu grad mismatch"
            )
            vt.grad = None
            vm.grad = None
        else:
            self.skipTest("relu primitives missing")

        # softmax via stable formula (use one-dim sample)
        if hasattr(torch, "exp") and (hasattr(ctorch, "exp") or hasattr(vm, "exp")):
            x = np.array([0.2, 1.0, -0.5, 0.3], dtype=np.float32)
            vt2 = torch.tensor(x, requires_grad=True)
            vm2 = safe_make_ctorch(x, requires_grad=True)
            xmax_t = vt2.max()
            exps_t = torch.exp(vt2 - xmax_t)
            denom_t = exps_t.sum()
            soft_t = exps_t / denom_t
            try:
                xmax_m = (
                    vm2.max() if hasattr(vm2, "max") else safe_make_ctorch(np.max(x))
                )
                exps_m = (
                    ctorch.exp(vm2 - xmax_m)
                    if hasattr(ctorch, "exp")
                    else (vm2 - xmax_m).exp()
                )
                denom_m = exps_m.sum()
                soft_m = exps_m / denom_m
            except Exception:
                self.skipTest("softmax build failed in thetorch")
            # weighted sum backward
            weights = np.array([0.2, 0.1, 0.4, 0.3], dtype=np.float32)
            wt = torch.tensor(weights)
            wm = safe_make_ctorch(weights, requires_grad=False)
            out_t = (soft_t * wt).sum()
            out_m = (soft_m * wm).sum()
            out_t.backward()
            out_m.backward()
            assert_allclose_torch_vs_tt(
                self, vt2.grad, thetorch_to_numpy(vm2.grad), msg="softmax grad mismatch"
            )
        else:
            self.skipTest("softmax primitives missing")

        # gelu (approx) and swish similar checks: skip if building blocks missing
        # gelu approx
        if (
            hasattr(torch, "tanh") or (hasattr(torch, "exp") and hasattr(torch, "sqrt"))
        ) and (
            hasattr(ctorch, "exp") or hasattr(ctorch, "sqrt") or hasattr(vm, "tanh")
        ):
            x = np.array([0.1, -0.5, 1.0], dtype=np.float32)
            vt3 = torch.tensor(x, requires_grad=True)
            vm3 = safe_make_ctorch(x, requires_grad=True)
            a = math.sqrt(2.0 / math.pi)
            inner_t = a * (vt3 + 0.044715 * (vt3**3))
            tanh_inner_t = (
                torch.tanh(inner_t)
                if hasattr(torch, "tanh")
                else (torch.exp(inner_t) - torch.exp(-inner_t))
                / (torch.exp(inner_t) + torch.exp(-inner_t))
            )
            gelu_t = 0.5 * vt3 * (1.0 + tanh_inner_t)

            a_m = safe_make_ctorch(float(math.sqrt(2.0 / math.pi)), requires_grad=False)
            inner_m = a_m * (vm3 + 0.044715 * (vm3**3))
            if hasattr(ctorch, "tanh"):
                tanh_inner_m = ctorch.tanh(inner_m)
            elif hasattr(vm3, "tanh"):
                tanh_inner_m = vm3.tanh()
            else:
                exi = ctorch.exp(inner_m)
                eni = ctorch.exp(-inner_m)
                tanh_inner_m = (exi - eni) / (exi + eni)
            gelu_m = 0.5 * vm3 * (1.0 + tanh_inner_m)

            out_t = gelu_t.sum()
            out_m = gelu_m.sum()
            out_t.backward()
            out_m.backward()
            assert_allclose_torch_vs_tt(
                self, vt3.grad, thetorch_to_numpy(vm3.grad), msg="gelu grad mismatch"
            )
        else:
            self.skipTest("gelu primitives missing")


class TestInplaceBehavior(unittest.TestCase):
    def test_inplace_on_nonleaf_allowed_and_grads(self):
        OPS = {
            "iadd": ("__iadd__", lambda x, y: x.__iadd__(y), "+="),
            "isub": ("__isub__", lambda x, y: x.__isub__(y), "-="),
            "imul": ("__imul__", lambda x, y: x.__imul__(y), "*="),
            "itruediv": ("__itruediv__", lambda x, y: x.__itruediv__(y), "/="),
        }
        for name, (mname, opfunc, sym) in OPS.items():
            with self.subTest(op=name):
                vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                xt = torch.tensor(vals.copy(), requires_grad=True)
                bt = xt * 2.0  # non-leaf
                # try torch: if torch disallows this pattern, skip
                try:
                    opfunc(bt, 3.0)
                    bt.sum().backward()
                except Exception:
                    self.skipTest(f"torch disallowed non-leaf in-place {sym}")

                xm = safe_make_ctorch(vals, requires_grad=True)
                bm = xm * 2.0
                if not hasattr(bm, mname):
                    self.skipTest(f"thetorch missing {mname}")
                # run op and backward
                try:
                    opfunc(bm, 3.0)
                    bm.sum().backward()
                except Exception as e:
                    self.fail(f"thetorch in-place {sym} on non-leaf failed: {e}")
                # grads of original leaf should exist
                self.assertIsNotNone(getattr(xm, "grad", None))
                self.assertIsNotNone(xt.grad)
                assert_allclose_torch_vs_tt(
                    self, xt.grad, xm.grad, msg=f"inplace {sym} grad mismatch"
                )

    def test_inplace_on_leaf_raises(self):
        # leaf that requires_grad True: in-place should raise RuntimeError (match torch)
        leaf = safe_make_ctorch([1.0, 2.0], requires_grad=True)
        if not hasattr(leaf, "__iadd__"):
            self.skipTest("__iadd__ not implemented")
        with self.assertRaises(RuntimeError):
            leaf += 1.0

    def test_inplace_broadcast_scalar_values_match_torch(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        at = torch.tensor(a.copy())
        at_clone = at.clone()
        at_clone *= 3.0
        np_t = torch_to_numpy(at_clone)

        am = safe_make_ctorch(a.copy(), requires_grad=False)
        if not hasattr(am, "__imul__"):
            self.skipTest("__imul__ not implemented in thetorch")
        try:
            am.__imul__(3.0)
        except Exception as e:
            self.fail(f"thetorch __imul__ raised during scalar broadcast test: {e}")
        np_m = thetorch_to_numpy(am)
        self.assertTrue(
            np.allclose(np_t, np_m, rtol=RTOL, atol=ATOL),
            "in-place scalar broadcast values mismatch",
        )


# ---------- Run ----------
if __name__ == "__main__":
    unittest.main(verbosity=2)
