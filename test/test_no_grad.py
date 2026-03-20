import unittest
import numpy as np

import cattorch as ctorch


class TestNoGradMinimal(unittest.TestCase):
    def test_no_grad_simple_ops(self):
        """
        Test that tt.no_grad():
        - allows forward computation
        - does NOT create .grad
        """

        # create tensors
        a = ctorch.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = ctorch.Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
        c = ctorch.Tensor(np.random.randn(3, 5).astype(np.float32), requires_grad=True)

        # forward inside no_grad
        with ctorch.no_grad():
            y = a @ b    # matmul + add

        y = y + c
        y = y.sum()
        y.backward()

        # gradients must NOT exist
        self.assertTrue(getattr(a, "grad", None) is None, f"a.grad should be None, but got {getattr(a, "grad", None)}")
        self.assertTrue(getattr(b, "grad", None) is None, f"b.grad should be None, but got {getattr(b, "grad", None)}")
        self.assertFalse(getattr(c, "grad", None) is None, f"c.grad should not be None, but got {getattr(c, "grad", None)}")

        # y should also not have grad
        self.assertFalse(getattr(y, "grad", None) is None, f"y.grad should not be None, but got {getattr(y, "grad", None)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)