# tests/test_einsum_unittest.py
import logging
import traceback  # noqa
import unittest

import numpy as np
import torch

import cattorch as ctorch

# ---------- Logging config (keep log for debug) ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_einsum")

# ---------- RNG deterministic (for reproduce the issuses) ----------
np.random.seed(0)
torch.manual_seed(0)

# ---------- Test cases ----------
BASIC_TESTS = [
    ("i->i", [(5,)]),
    ("ij->ij", [(3, 4)]),
    ("i,i->", [(7,), (7,)]),
    ("ik,kj->ij", [(3, 5), (5, 4)]),
    ("ijk->ij", [(2, 3, 4)]),
    ("ijk->", [(2, 3, 4)]),
    ("i,j->ij", [(4,), (6,)]),
    ("ijkjk->i", [(2, 3, 4, 3, 4)]),
    ("ii,jj->ij", [(3, 3), (4, 4)]),
    ("ab,cd->abcd", [(2, 3), (4, 5)]),
]

DIAGONAL_TESTS = [
    ("ii->i", [(5, 5)]),
    ("iiiii->i", [(4, 4, 4, 4, 4)]),
    ("ii->", [(6, 6)]),
    ("ijj->i", [(3, 4, 4)]),
    ("iijj->ij", [(3, 3, 4, 4)]),
    ("abcabc->abc", [(2, 3, 4, 2, 3, 4)]),
]

MULTI_TENSOR_TESTS = [
    ("abc,acd->abd", [(2, 3, 4), (2, 4, 5)]),
    ("bshd,bshd->bsh", [(2, 3, 4, 5), (2, 3, 4, 5)]),
    ("abca,acdb->bd", [(2, 3, 4, 2), (2, 4, 6, 3)]),
    ("ab,bc,cd->ad", [(2, 3), (3, 4), (4, 5)]),
    ("ai,bi,ci->abc", [(2, 4), (3, 4), (5, 4)]),
    ("abc,bcd,cde,def->af", [(2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7)]),
]

ELLIPSIS_TESTS = [
    ("i...->i", [(4, 5, 6, 7)]),
    ("i...->i...", [(3, 4, 5)]),
    ("...ij,...jk->...ik", [(4, 1, 2, 6), (4, 3, 6, 5)]),
    ("b...h,b...h->b...", [(3, 4, 5, 6), (3, 1, 5, 6)]),
    ("...ij,...ij->...", [(1, 3, 4), (5, 3, 4)]),
    ("a...i,b...i->ab...", [(2, 1, 4), (1, 3, 4)]),
    ("b...h->b...", [(2, 3, 4, 5)]),
]

EDGE_CASES = [
    ("i,i->", [(1,), (1,)]),
    ("a,a->", [(2,), (2,)]),
    ("a,b->ab", [(2,), (3,)]),
    ("...->", [(1, 1, 2, 3)]),
    ("ii->", [(1, 1)]),
    ("axb,bxc->ac", [(2, 3, 4), (4, 3, 5)]),
]

FAILS_CASES = [
    ("ij,i,i->", [(2, 3), (3,), (3,)]),
    ("...ij,...ij->...", [(2, 3, 4), (3, 3, 4)]),
    ("abaac->bc", [(2, 3, 4)]),
]

ALL_OK_TESTS = (
    BASIC_TESTS + DIAGONAL_TESTS + MULTI_TENSOR_TESTS + ELLIPSIS_TESTS + EDGE_CASES
)


# ---------- Helpers ----------
def to_numpy(x):
    """Convert an object (mtorch/torch/np) to numpy array for comparisons."""
    if x is None:
        return None
    if hasattr(x, "asnumpy"):
        return x.asnumpy()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, ctorch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    # last resort
    return np.array(x)


def _make_random_arrays(shapes):
    return [np.random.randn(*s).astype(np.float32) for s in shapes]


# ---------- UnitTest ----------
class TestEinsum(unittest.TestCase):
    def run_case_compare(self, eq, shapes):
        """
        Run the custom thetorch.einsum and torch.einsum for a single case.
        Returns tuple (custom_failed, pytorch_failed, out_np, out_t_np, my_tensors, torch_tensors)
        or raises AssertionError on mismatch.
        """
        logger.info("TEST CASE: %s  shapes=%s", eq, shapes)
        arrays = _make_random_arrays(shapes)

        custom_failed = False
        pytorch_failed = False
        out = None
        out_t = None
        my_tensors = None
        torch_tensors = None

        # run custom (thetorch)
        try:
            my_tensors = [ctorch.Tensor(a, requires_grad=True) for a in arrays]
            out = ctorch.einsum(eq, *my_tensors)
            out.sum().backward()
            logger.info("✅ Custom einsum ran for %s", eq)
        except Exception as e:
            custom_failed = True
            logger.exception("❌ Custom einsum raised exception for %s: %s", eq, e)

        # run pytorch
        try:
            torch_tensors = [torch.tensor(a, requires_grad=True) for a in arrays]
            out_t = torch.einsum(eq, *torch_tensors)
            out_t.sum().backward()
            logger.info("✅ PyTorch einsum ran for %s", eq)
        except Exception as e:
            pytorch_failed = True
            logger.exception("❌ PyTorch einsum raised exception for %s: %s", eq, e)

        # If both raised: caller can decide what to do (skip/fail)
        if custom_failed and pytorch_failed:
            logger.warning("Both implementations raised for %s", eq)
            return (True, True, None, None, my_tensors, torch_tensors)

        # If only one raised -> that's a mismatch
        if custom_failed and not pytorch_failed:
            raise AssertionError(
                "Custom implementation raised an exception but PyTorch did not."
            )
        if pytorch_failed and not custom_failed:
            raise AssertionError(
                "PyTorch raised an exception but custom implementation did not."
            )

        # both succeeded -> compare output and grads
        out_np = to_numpy(out.data if hasattr(out, "data") else out)
        out_t_np = to_numpy(out_t)
        logger.debug(
            "Output shapes: custom=%s torch=%s",
            getattr(out_np, "shape", None),
            getattr(out_t_np, "shape", None),
        )

        np.testing.assert_allclose(out_np, out_t_np, atol=1e-5)
        logger.info("Outputs match for %s", eq)

        # compare grads
        for i in range(len(arrays)):
            my_grad = getattr(my_tensors[i], "grad", None)
            torch_grad = getattr(torch_tensors[i], "grad")
            torch_grad_np = to_numpy(torch_grad)

            if my_grad is None:
                my_grad_np = np.zeros_like(torch_grad_np)
            else:
                my_grad_np = to_numpy(my_grad)

            diff = np.abs(my_grad_np - torch_grad_np).max()
            logger.info("  grad diff input %d: %g", i, diff)

            if diff >= 1e-5:
                raise AssertionError(f"grad mismatch for input {i}: max diff {diff}")

        return (False, False, out_np, out_t_np, my_tensors, torch_tensors)

    def test_all_ok_cases(self):
        """Execute all standard test cases; each is implemented as a subTest to facilitate debugging."""
        pass_cases = 0
        for eq, shapes in ALL_OK_TESTS:
            with self.subTest(equation=eq, shapes=shapes):
                custom_failed, pytorch_failed, *_ = self.run_case_compare(eq, shapes)
                # In the event that both fail, the test case will be skipped to prioritize focus on viable cases.
                if custom_failed and pytorch_failed:
                    self.skipTest(
                        f"Both implementations raised for {eq} shapes={shapes}"
                    )
                else:
                    pass_cases += 1
        logger.info(f"✅ Passed {pass_cases} / {len(ALL_OK_TESTS)}")

    # Known-fail cases are marked as expectedFailure
    @unittest.expectedFailure
    def test_known_fail_cases(self):
        both_fails = 0
        for eq, shapes in FAILS_CASES:
            # If both raise, force failure to trigger expectedFailure (xfail) behavior.
            custom_failed, pytorch_failed, *_ = self.run_case_compare(eq, shapes)
            if custom_failed and pytorch_failed:
                # Explicitly trigger a failure for expectedFailure detection
                both_fails += 1
        if both_fails < len(FAILS_CASES):
            logger.info(
                f"Expected both implementations raised (expected failure). But only {both_fails}/{len(FAILS_CASES)} passed."
            )
        else:
            raise AssertionError(
                f"✅ Both implementations raised (expected failure). {both_fails}/{len(FAILS_CASES)} passed."
            )


if __name__ == "__main__":
    # Direct execution prints logs and runs unittests
    unittest.main(verbosity=2)
