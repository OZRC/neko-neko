# cattorch/nn/layers.py
import numpy as np

from ..tensor import Tensor
from . import functional
from .module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # weight shape: (out_features, in_features) to match torch convention
        w = np.random.randn(out_features, in_features) * (1.0 / np.sqrt(in_features))
        b = np.zeros((out_features,)) if bias else None
        self.weight = Parameter(w)
        if bias:
            self.bias = Parameter(b)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features) -> output (..., out_features)
        # we rely on @ (matmul) broadcasting rules: x @ weight.T
        out = x.matmul(self.weight.transpose(0, 1))
        if self.bias is not None:
            # reshape bias to broadcast on leading dims
            newshape = (1,) * (out.ndim - 1) + (-1,)
            out = out + self.bias.reshape(*newshape)
        return out


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        w = np.random.randn(num_embeddings, embedding_dim) * 0.01
        self.weight = Parameter(w)

    def forward(self, indices) -> Tensor:
        # Accept Tensor or numpy/list indices.
        # If indices is a Tensor, extract its underlying array for indexing.
        if isinstance(indices, Tensor):
            # try backend-specific conversion
            data = indices.data
            if hasattr(data, "asnumpy"):
                idx_arr = data.asnumpy()
            else:
                idx_arr = np.array(data)
        else:
            idx_arr = np.array(indices)

        # Ensure integer dtype for indexing
        idx_arr = idx_arr.astype(np.int64)

        # Index into self.weight (Parameter) using a numpy integer array
        return self.weight[idx_arr]


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        # create mask with same shape
        mask = Tensor(np.random.rand(*x.shape) >= self.p, requires_grad=False)
        # scale to keep expectation same
        mask = mask / (1.0 - self.p)
        return x * mask


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        # gamma and beta
        gamma = np.ones(self.normalized_shape)
        beta = np.zeros(self.normalized_shape)
        self.gamma = Parameter(gamma)
        self.beta = Parameter(beta)

    def forward(self, x: Tensor) -> Tensor:
        # normalize over last len(normalized_shape) dims
        dim = (
            tuple(range(-len(self.normalized_shape), 0))
            if len(self.normalized_shape) > 0
            else ()
        )
        mean = x.mean(dim=dim, keepdims=True)
        var = ((x - mean) ** 2).mean(dim=dim, keepdims=True)
        inv = (var + self.eps) ** (-0.5)
        out = (x - mean) * inv
        # broadcast gamma/beta
        return out * self.gamma + self.beta


class Conv1D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Conv1D supporting groups.

        weight shape: (out_channels, in_channels//groups, kernel_size)
        bias shape: (out_channels,)
        """
        super().__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # initialize weights (out_channels, in_channels/groups, kernel_size)
        fan_in = in_channels // groups * kernel_size
        w = np.random.randn(out_channels, in_channels // groups, kernel_size) * (
            1.0 / np.sqrt(fan_in)
        )
        self.weight = Parameter(w)
        if bias:
            self.bias = Parameter(np.zeros((out_channels,)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch, in_channels, width)
        returns: (batch, out_channels, width_out)
        """
        B, C_in, W = x.shape
        C_out = self.out_channels
        K = self.kernel_size
        stride = self.stride
        pad = self.padding
        dil = self.dilation
        groups = self.groups

        # compute output width
        W_out = (W + 2 * pad - dil * (K - 1) - 1) // stride + 1
        if W_out <= 0:
            # return empty output shaped correctly
            return Tensor(np.zeros((B, C_out, 0)))

        out_batches = []
        for b in range(B):
            out_chs = []
            # iterate over output channels (support groups by mapping ic range)
            for oc in range(C_out):
                # determine which group this output channel belongs to
                g = oc * groups // C_out  # integer group index
                in_per_group = C_in // groups
                out_per_group = C_out // groups
                ic_start = g * in_per_group
                ic_end = (g + 1) * in_per_group
                # weight slice for this output channel (in_per_group, K)
                w_oc = self.weight[oc]  # shape (in_per_group, K)
                rows = []
                for i in range(W_out):
                    pos = i * stride - pad
                    acc = None
                    # accumulate over input channels and kernel
                    # iterate ic in group range
                    for ic in range(ic_start, ic_end):
                        for k in range(K):
                            idx = pos + k * dil
                            if 0 <= idx < W:
                                val = x[b, ic, idx] * w_oc[ic - ic_start, k]
                                if acc is None:
                                    acc = val
                                else:
                                    acc = acc + val
                            else:
                                # out of bounds -> skip (effectively add zero)
                                continue
                    if acc is None:
                        # all indices out of bounds -> zero
                        acc = Tensor(np.array(0.0))
                    rows.append(acc)
                ch = functional.stack(rows, dim=0)  # (W_out,)
                out_chs.append(ch)
            out_b = functional.stack(out_chs, dim=0)  # (C_out, W_out)
            out_batches.append(out_b)
        out = functional.stack(out_batches, dim=0)  # (B, C_out, W_out)

        if self.bias is not None:
            # bias: shape (out_channels,) -> broadcast to (B, C_out, W_out)
            out = out + self.bias.reshape((1, -1, 1))
        return out


class Conv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Conv2D supporting groups.

        kernel_size: int or (kh, kw)
        stride, padding, dilation: int or (h, w)
        weight shape: (out_channels, in_channels//groups, kh, kw)
        bias shape: (out_channels,)
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        # normalize stride/padding/dilation to tuples
        if isinstance(stride, int):
            stride_h, stride_w = stride, stride
        else:
            stride_h, stride_w = stride
        if isinstance(padding, int):
            pad_h, pad_w = padding, padding
        else:
            pad_h, pad_w = padding
        if isinstance(dilation, int):
            dil_h, dil_w = dilation, dilation
        else:
            dil_h, dil_w = dilation

        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = (stride_h, stride_w)
        self.padding = (pad_h, pad_w)
        self.dilation = (dil_h, dil_w)
        self.groups = groups

        fan_in = (in_channels // groups) * kh * kw
        w = np.random.randn(out_channels, in_channels // groups, kh, kw) * (
            1.0 / np.sqrt(fan_in)
        )
        self.weight = Parameter(w)
        if bias:
            self.bias = Parameter(np.zeros((out_channels,)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch, in_channels, height, width)
        returns: (batch, out_channels, height_out, width_out)
        """
        B, C_in, H, W = x.shape
        C_out = self.out_channels
        kh, kw = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation
        groups = self.groups

        H_out = (H + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
        W_out = (W + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1
        if H_out <= 0 or W_out <= 0:
            return Tensor(np.zeros((B, C_out, max(0, H_out), max(0, W_out))))

        out_batches = []
        for b in range(B):
            out_chs = []
            for oc in range(C_out):
                g = oc * groups // C_out
                in_per_group = C_in // groups
                out_per_group = C_out // groups
                ic_start = g * in_per_group
                ic_end = (g + 1) * in_per_group
                w_oc = self.weight[oc]  # (in_per_group, kh, kw)
                rows = []
                for i in range(H_out):
                    cols = []
                    for j in range(W_out):
                        pos_h = i * stride_h - pad_h
                        pos_w = j * stride_w - pad_w
                        acc = None
                        for ic in range(ic_start, ic_end):
                            for kh_i in range(kh):
                                for kw_j in range(kw):
                                    ih = pos_h + kh_i * dil_h
                                    iw = pos_w + kw_j * dil_w
                                    if 0 <= ih < H and 0 <= iw < W:
                                        val = (
                                            x[b, ic, ih, iw]
                                            * w_oc[ic - ic_start, kh_i, kw_j]
                                        )
                                        if acc is None:
                                            acc = val
                                        else:
                                            acc = acc + val
                                    else:
                                        # skip (zero)
                                        continue
                        if acc is None:
                            acc = Tensor(np.array(0.0))
                        cols.append(acc)
                    rows.append(functional.stack(cols, dim=0))  # (W_out,)
                out_chs.append(
                    functional.stack(rows, dim=0)
                )  # (kh_out, W_out) but actually (H_out, W_out)
            out_b = functional.stack(out_chs, dim=0)  # (C_out, H_out, W_out)
            out_batches.append(out_b)
        out = functional.stack(out_batches, dim=0)  # (B, C_out, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias.reshape((1, -1, 1, 1))
        return out


# -------------------------
# MaxPool1D / MaxPool2D
# - Supports kernel_size, stride, padding, dilation.
# - Uses explicit indexing and Tensor ops to remain autograd-friendly.
# -------------------------
class MaxPool1D(Module):
    def __init__(
        self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1
    ):
        """
        MaxPool1D over last spatial dim.
        x: (batch, channels, width)
        returns: (batch, channels, width_out)
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        B, C, W = x.shape
        k = self.kernel_size
        stride = self.stride
        pad = self.padding
        dil = self.dilation

        W_out = (W + 2 * pad - dil * (k - 1) - 1) // stride + 1
        if W_out <= 0:
            return Tensor(np.zeros((B, C, 0)))

        out_batches = []
        for b in range(B):
            out_chs = []
            for c in range(C):
                rows = []
                for i in range(W_out):
                    pos = i * stride - pad
                    acc = None
                    for kk in range(k):
                        idx = pos + kk * dil
                        if 0 <= idx < W:
                            val = x[b, c, idx]
                        else:
                            # padded value -> treat as 0.0 (consistent with zero padding)
                            val = Tensor(np.array(0.0))
                        if acc is None:
                            acc = val
                        else:
                            # elementwise maximum
                            acc = functional.maximum(acc, val)
                    if acc is None:
                        acc = Tensor(np.array(0.0))
                    rows.append(acc)
                out_chs.append(functional.stack(rows, dim=0))  # (W_out,)
            out_batches.append(functional.stack(out_chs, dim=0))  # (C, W_out)
        out = functional.stack(out_batches, dim=0)  # (B, C, W_out)
        return out


class MaxPool2D(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        """
        MaxPool2D over last two spatial dims.
        kernel_size: int or (kh, kw)
        stride, padding, dilation: int or tuple
        x: (batch, channels, H, W)
        returns: (batch, channels, H_out, W_out)
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        if stride is None:
            stride_h, stride_w = kh, kw
        else:
            if isinstance(stride, int):
                stride_h, stride_w = stride, stride
            else:
                stride_h, stride_w = stride

        if isinstance(padding, int):
            pad_h, pad_w = padding, padding
        else:
            pad_h, pad_w = padding

        if isinstance(dilation, int):
            dil_h, dil_w = dilation, dilation
        else:
            dil_h, dil_w = dilation

        self.kernel_size = (kh, kw)
        self.stride = (stride_h, stride_w)
        self.padding = (pad_h, pad_w)
        self.dilation = (dil_h, dil_w)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        kh, kw = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation

        H_out = (H + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
        W_out = (W + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1
        if H_out <= 0 or W_out <= 0:
            return Tensor(np.zeros((B, C, max(0, H_out), max(0, W_out))))

        out_batches = []
        for b in range(B):
            out_chs = []
            for c in range(C):
                rows = []
                for i in range(H_out):
                    cols = []
                    for j in range(W_out):
                        pos_h = i * stride_h - pad_h
                        pos_w = j * stride_w - pad_w
                        acc = None
                        for kh_i in range(kh):
                            for kw_j in range(kw):
                                ih = pos_h + kh_i * dil_h
                                iw = pos_w + kw_j * dil_w
                                if 0 <= ih < H and 0 <= iw < W:
                                    val = x[b, c, ih, iw]
                                else:
                                    # padded -> zero
                                    val = Tensor(np.array(0.0))
                                if acc is None:
                                    acc = val
                                else:
                                    acc = functional.maximum(acc, val)
                        if acc is None:
                            acc = Tensor(np.array(0.0))
                        cols.append(acc)
                    rows.append(functional.stack(cols, dim=0))  # (W_out,)
                out_chs.append(functional.stack(rows, dim=0))  # (H_out, W_out)
            out_batches.append(functional.stack(out_chs, dim=0))  # (C, H_out, W_out)
        out = functional.stack(out_batches, dim=0)  # (B, C, H_out, W_out)
        return out


# --- Losses ---
class MSELoss(Module):
    """Mean Squared Error loss. Expects input and target of same shape."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        sq = diff * diff
        if self.reduction == "mean":
            return functional.mean(sq)
        elif self.reduction == "sum":
            return functional.sum(sq)
        else:
            return sq


class BCELoss(Module):
    """Binary Cross Entropy. Expects input (probabilities in (0,1)) and target same shape."""

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # clamp inputs to avoid log(0)
        x = functional.clamp(input, min=self.eps, max=1.0 - self.eps)
        term1 = target * functional.log(x)
        term2 = (1 - target) * functional.log(1 - x)
        out = -(term1 + term2)
        if self.reduction == "mean":
            return functional.mean(out)
        elif self.reduction == "sum":
            return functional.sum(out)
        else:
            return out


class CrossEntropyLoss(Module):
    """Categorical Cross-Entropy (combines softmax + NLL).
    Expects input logits of shape (N, C) and targets as integer class indices shape (N,).
    Returns scalar loss (mean by default)."""

    def __init__(self, reduction: str = "mean", eps: float = 1e-9):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input: (N, C) logits; target: (N,) int indices (or broadcastable)
        # compute softmax probabilities
        exp = functional.exp(input)
        denom = functional.sum(exp, dim=1, keepdims=True)  # (N,1)
        probs = exp / (denom + 1e-12)
        # select the probability of the true class for each sample
        # target may be Tensor of ints or python list/np array
        if not isinstance(target, Tensor):
            inds = target
        else:
            try:
                inds = list(map(int, target.asnumpy()))
            except Exception:
                inds = target
        selected = probs[range(probs.shape[0]), inds]
        loss = -functional.log(selected + self.eps)
        if self.reduction == "mean":
            return functional.mean(loss)
        elif self.reduction == "sum":
            return functional.sum(loss)
        else:
            return loss


# --- Recurrent layers (simple, autograd-friendly, PyTorch-like API) ---
def _sigmoid(x: Tensor) -> Tensor:
    return 1.0 / (1.0 + functional.exp(-x))


class RNN(Module):
    """
    Simple RNN layer (vanilla). Expects input shaped (seq_len, batch, input_size).
    Returns (output, h_n) where output is (seq_len, batch, hidden_size) and h_n is (batch, hidden_size).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ):
        super().__init__()
        assert nonlinearity in ("tanh", "relu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        # weight shapes follow (hidden, in) convention for matmul with x @ weight.T
        w_ih = np.random.randn(hidden_size, input_size) * (1.0 / np.sqrt(input_size))
        w_hh = np.random.randn(hidden_size, hidden_size) * (1.0 / np.sqrt(hidden_size))
        self.weight_ih = Parameter(w_ih)
        self.weight_hh = Parameter(w_hh)
        if bias:
            self.bias_ih = Parameter(np.zeros((hidden_size,)))
            self.bias_hh = Parameter(np.zeros((hidden_size,)))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x: Tensor, hx: Tensor = None):
        # x: (seq_len, batch, input_size)
        seq_len, batch, _ = x.shape
        if hx is None:
            h_t = Tensor(np.zeros((batch, self.hidden_size)))
        else:
            h_t = hx
        outputs = []
        for t in range(seq_len):
            x_t = x[t]  # (batch, input)
            lin = x_t @ self.weight_ih.transpose(0, 1)
            if self.bias_ih is not None:
                lin = lin + self.bias_ih.reshape((1, -1))
            lin2 = h_t @ self.weight_hh.transpose(0, 1)
            if self.bias_hh is not None:
                lin2 = lin2 + self.bias_hh.reshape((1, -1))
            h_linear = lin + lin2
            if self.nonlinearity == "tanh":
                h_t = functional.tanh(h_linear)
            else:
                # simple ReLU implementation using clamp
                h_t = functional.clamp(h_linear, min=0.0)
            outputs.append(h_t)
        out = functional.stack(outputs, dim=0)
        return out, h_t


class GRU(Module):
    """
    Gated Recurrent Unit layer. Input: (seq_len, batch, input_size).
    Returns (output, h_n).
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # input weights (reset, update, new) stacked for convenience
        self.weight_ih = Parameter(
            np.random.randn(3 * hidden_size, input_size) * (1.0 / np.sqrt(input_size))
        )
        self.weight_hh = Parameter(
            np.random.randn(3 * hidden_size, hidden_size) * (1.0 / np.sqrt(hidden_size))
        )
        if bias:
            self.bias_ih = Parameter(np.zeros((3 * hidden_size,)))
            self.bias_hh = Parameter(np.zeros((3 * hidden_size,)))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x: Tensor, hx: Tensor = None):
        seq_len, batch, _ = x.shape
        if hx is None:
            h_t = Tensor(np.zeros((batch, self.hidden_size)))
        else:
            h_t = hx
        outputs = []
        for t in range(seq_len):
            x_t = x[t]  # (batch, input)
            # compute gates: use chunking by manual indexing/slicing
            # linear for input: (batch, 3*hidden)
            wx = x_t @ self.weight_ih.transpose(0, 1)
            if self.bias_ih is not None:
                wx = wx + self.bias_ih.reshape((1, -1))
            wh = h_t @ self.weight_hh.transpose(0, 1)
            if self.bias_hh is not None:
                wh = wh + self.bias_hh.reshape((1, -1))
            # split into r,z,n
            r = _sigmoid(wx[:, : self.hidden_size] + wh[:, : self.hidden_size])
            z = _sigmoid(
                wx[:, self.hidden_size : 2 * self.hidden_size]
                + wh[:, self.hidden_size : 2 * self.hidden_size]
            )
            n = functional.tanh(
                wx[:, 2 * self.hidden_size :] + r * wh[:, 2 * self.hidden_size :]
            )
            h_t = (1 - z) * n + z * h_t
            outputs.append(h_t)
        out = functional.stack(outputs, dim=0)
        return out, h_t


class LSTM(Module):
    """
    Long Short-Term Memory layer. Input: (seq_len, batch, input_size).
    Returns (output, (h_n, c_n)) where h_n and c_n are (batch, hidden_size).
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # weight matrices for input and hidden - stacked i,f,g,o
        self.weight_ih = Parameter(
            np.random.randn(4 * hidden_size, input_size) * (1.0 / np.sqrt(input_size))
        )
        self.weight_hh = Parameter(
            np.random.randn(4 * hidden_size, hidden_size) * (1.0 / np.sqrt(hidden_size))
        )
        if bias:
            self.bias_ih = Parameter(np.zeros((4 * hidden_size,)))
            self.bias_hh = Parameter(np.zeros((4 * hidden_size,)))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x: Tensor, hx: tuple = None):
        seq_len, batch, _ = x.shape
        if hx is None:
            h_t = Tensor(np.zeros((batch, self.hidden_size)))
            c_t = Tensor(np.zeros((batch, self.hidden_size)))
        else:
            h_t, c_t = hx
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            wx = x_t @ self.weight_ih.transpose(0, 1)
            if self.bias_ih is not None:
                wx = wx + self.bias_ih.reshape(1, -1)
            wh = h_t @ self.weight_hh.transpose(0, 1)
            if self.bias_hh is not None:
                wh = wh + self.bias_hh.reshape(1, -1)
            # split into gates
            i = _sigmoid(wx[:, : self.hidden_size] + wh[:, : self.hidden_size])
            f = _sigmoid(
                wx[:, self.hidden_size : 2 * self.hidden_size]
                + wh[:, self.hidden_size : 2 * self.hidden_size]
            )
            g = functional.tanh(
                wx[:, 2 * self.hidden_size : 3 * self.hidden_size]
                + wh[:, 2 * self.hidden_size : 3 * self.hidden_size]
            )
            o = _sigmoid(
                wx[:, 3 * self.hidden_size : 4 * self.hidden_size]
                + wh[:, 3 * self.hidden_size : 4 * self.hidden_size]
            )
            c_t = f * c_t + i * g
            h_t = o * functional.tanh(c_t)
            outputs.append(h_t)
        out = functional.stack(outputs, dim=0)
        return out, (h_t, c_t)
