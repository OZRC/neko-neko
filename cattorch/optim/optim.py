# cattorch/optim/optim.py
# Optimizers for cattorch: SGD, SGDMomentum, RMSProp, Adam, AdamW

from typing import Iterable, List, Dict
import numpy as np

from ..nn.module import Parameter
from ..backend import _array as ap
from ..backend.utils import to_raw


def _set_param_data(param: Parameter, np_arr: np.ndarray):
    """Set parameter.data to a new backend array constructed from np_arr, preserving device/dtype when possible."""
    device = getattr(param.data, "device", None)
    dtype = getattr(param.data, "dtype", None)
    try:
        if device is not None or dtype is not None:
            param.data = ap.Array(np_arr, device=device, dtype=dtype)
        else:
            param.data = ap.Array(np_arr)
    except Exception:
        # best-effort fallback
        param.data = ap.Array(np_arr)


class Optimizer:
    """Base optimizer. Subclasses should implement step().

    params can be an iterator of Parameter objects or any nested structure yielding Parameter.
    """

    def __init__(self, params: Iterable[Parameter], lr: float = 1e-3):
        self.defaults = {"lr": lr}
        # flatten
        self.param_list: List[Parameter] = []
        for p in params:
            if p is None:
                continue
            self.param_list.append(p)

    def zero_grad(self):
        for p in self.param_list:
            # safe way: set grad to None. If your framework expects Tensor.zero_(), adapt here.
            try:
                p.grad = None
            except Exception:
                # try removing attribute
                if hasattr(p, "grad"):
                    delattr(p, "grad")

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self, params: Iterable[Parameter], lr: float = 1e-3, weight_decay: float = 0.0
    ):
        super().__init__(params, lr=lr)
        self.weight_decay = weight_decay

    def step(self):
        lr = float(self.defaults["lr"])
        for p in self.param_list:
            if not hasattr(p, "grad") or p.grad is None:
                continue
            g = to_raw(p.grad)
            w = to_raw(p.data)
            if self.weight_decay:
                g = g + self.weight_decay * w
            new_w = w - lr * g
            _set_param_data(p, new_w)


class SGDMomentum(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(params, lr=lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        # state maps param -> velocity (numpy array)
        self.state: Dict[int, np.ndarray] = {}

    def step(self):
        lr = float(self.defaults["lr"])
        for p in self.param_list:
            if not hasattr(p, "grad") or p.grad is None:
                continue
            g = to_raw(p.grad)
            w = to_raw(p.data)
            if self.weight_decay:
                g = g + self.weight_decay * w
            pid = id(p)
            v = self.state.get(pid)
            if v is None:
                v = np.zeros_like(g)
            # v = momentum * v + (1 - dampening) * g
            v = self.momentum * v + (1.0 - self.dampening) * g
            self.state[pid] = v
            if self.nesterov:
                update = g + self.momentum * v
            else:
                update = v
            new_w = w - lr * update
            _set_param_data(p, new_w)


class RMSProp(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ):
        super().__init__(params, lr=lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.square_avg: Dict[int, np.ndarray] = {}
        self.avg: Dict[int, np.ndarray] = {} if centered else {}
        self.momentum_buffer: Dict[int, np.ndarray] = {}

    def step(self):
        lr = float(self.defaults["lr"])
        for p in self.param_list:
            if not hasattr(p, "grad") or p.grad is None:
                continue
            g = to_raw(p.grad)
            w = to_raw(p.data)
            if self.weight_decay:
                g = g + self.weight_decay * w
            pid = id(p)
            sq = self.square_avg.get(pid)
            if sq is None:
                sq = np.zeros_like(g)
            sq = self.alpha * sq + (1 - self.alpha) * (g * g)
            self.square_avg[pid] = sq
            if self.centered:
                avg = self.avg.get(pid)
                if avg is None:
                    avg = np.zeros_like(g)
                avg = self.alpha * avg + (1 - self.alpha) * g
                self.avg[pid] = avg
                avg_den = sq - avg * avg
                denom = np.sqrt(avg_den + self.eps)
            else:
                denom = np.sqrt(sq + self.eps)
            if self.momentum:
                buf = self.momentum_buffer.get(pid)
                if buf is None:
                    buf = np.zeros_like(g)
                buf = self.momentum * buf + g / denom
                self.momentum_buffer[pid] = buf
                step = lr * buf
            else:
                step = lr * g / denom
            new_w = w - step
            _set_param_data(p, new_w)


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr=lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr = float(self.defaults["lr"])
        bias_correction1 = 1 - self.beta1**self.step_count
        bias_correction2 = 1 - self.beta2**self.step_count
        for p in self.param_list:
            if not hasattr(p, "grad") or p.grad is None:
                continue
            g = to_raw(p.grad)
            w = to_raw(p.data)
            if self.weight_decay:
                g = g + self.weight_decay * w
            pid = id(p)
            m = self.m.get(pid)
            v = self.v.get(pid)
            if m is None:
                m = np.zeros_like(g)
            if v is None:
                v = np.zeros_like(g)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)
            self.m[pid] = m
            self.v[pid] = v
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            step = lr * m_hat / (np.sqrt(v_hat) + self.eps)
            new_w = w - step
            _set_param_data(p, new_w)


class AdamW(Adam):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        # AdamW uses decoupled weight decay; we keep weight_decay here and apply separately
        self.adamw_weight_decay = weight_decay

    def step(self):
        # same as Adam but apply decoupled weight decay (p <- p - lr * weight_decay * p) before/after step
        self.step_count += 1
        lr = float(self.defaults["lr"])
        bias_correction1 = 1 - self.beta1**self.step_count
        bias_correction2 = 1 - self.beta2**self.step_count
        for p in self.param_list:
            if not hasattr(p, "grad") or p.grad is None:
                continue
            g = to_raw(p.grad)
            w = to_raw(p.data)
            # decoupled weight decay
            if self.adamw_weight_decay:
                w = w - lr * self.adamw_weight_decay * w
            pid = id(p)
            m = self.m.get(pid)
            v = self.v.get(pid)
            if m is None:
                m = np.zeros_like(g)
            if v is None:
                v = np.zeros_like(g)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)
            self.m[pid] = m
            self.v[pid] = v
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            step = lr * m_hat / (np.sqrt(v_hat) + self.eps)
            new_w = w - step
            _set_param_data(p, new_w)


# End of optim.py
