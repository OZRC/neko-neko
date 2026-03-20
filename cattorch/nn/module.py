# cattorch/nn/module.py
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..tensor import Tensor


class Parameter(Tensor):
    """
    Marker subclass for parameters (like torch.nn.Parameter).

    Behaves like Tensor but marks a leaf parameter that by default requires grad.
    """

    def __init__(
        self,
        data,
        requires_grad: bool = True,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
        name: Optional[str] = None,
    ):
        # Delegate to Tensor - Tensor constructor will coerce to ap.Array
        super().__init__(
            data,
            requires_grad=bool(requires_grad),
            device=device,
            dtype=dtype,
            name=name,
        )
        # Ensure this is treated as a leaf parameter
        try:
            # Tensor may have internal flags; set _is_leaf to True
            object.__setattr__(self, "_is_leaf", True)
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"Parameter({super().__repr__()})"


# -----------------------------
# Module implementation
# -----------------------------
class Module:
    """
    Lightweight Module similar to torch.nn.Module.

    Features:
    - automatic registration of Parameters and sub-Modules via attribute assignment
    - traversal utilities: parameters(), named_parameters(), children(), modules()
    - state_dict / load_state_dict
    - train() / eval()
    - apply(fn) to recursively apply a function to submodules
    - zero_grad() helper
    """

    def __init__(self) -> None:
        # internal storage for parameters and submodules
        object.__setattr__(self, "_parameters", {})  # type: Dict[str, Parameter]
        object.__setattr__(self, "_modules", {})  # type: Dict[str, Module]
        object.__setattr__(self, "_training", True)

    # ---- attribute interception: register parameters and modules automatically ----
    def __setattr__(self, name: str, value: Any) -> None:
        # If setting a Parameter, register it
        if isinstance(value, Parameter):
            params: Dict[str, Parameter] = getattr(self, "_parameters", None)
            if params is None:
                object.__setattr__(self, name, value)
            else:
                params[name] = value
                object.__setattr__(self, name, value)
            return

        # If setting a Module, register it as a submodule
        if isinstance(value, Module):
            mods: Dict[str, Module] = getattr(self, "_modules", None)
            if mods is None:
                object.__setattr__(self, name, value)
            else:
                mods[name] = value
                object.__setattr__(self, name, value)
            return

        # regular attribute
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        # Remove from registries if present
        if "_parameters" in self.__dict__ and name in self.__dict__.get(
            "_parameters", {}
        ):
            self._parameters.pop(name, None)
        if "_modules" in self.__dict__ and name in self.__dict__.get("_modules", {}):
            self._modules.pop(name, None)
        object.__delattr__(self, name)

    # -------------------------
    # registration helpers
    # -------------------------
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if param is None:
            return
        if not isinstance(param, Parameter):
            raise TypeError("registered object must be a Parameter")
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        if module is None:
            return
        if not isinstance(module, Module):
            raise TypeError("registered object must be a Module")
        self._modules[name] = module
        object.__setattr__(self, name, module)

    # -------------------------
    # traversal utilities
    # -------------------------
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for pname, p in self._parameters.items():
            yield p
        if recurse:
            for m in self._modules.values():
                yield from m.parameters(recurse=True)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        # prefix used for recursive names
        for name, p in self._parameters.items():
            fullname = f"{prefix}{name}" if prefix == "" else f"{prefix}.{name}"
            yield fullname, p
        if recurse:
            for mname, m in self._modules.items():
                child_prefix = (
                    f"{prefix}{mname}" if prefix == "" else f"{prefix}.{mname}"
                )
                yield from m.named_parameters(child_prefix, recurse=True)

    def children(self) -> Iterator["Module"]:
        for m in self._modules.values():
            yield m

    def modules(self) -> Iterator["Module"]:
        # mimic torch: yield self first, then submodules
        yield self
        for m in self._modules.values():
            yield from m.modules()

    # -------------------------
    # state dict / load
    # -------------------------
    def state_dict(self) -> Dict[str, Any]:
        """
        Return a dict mapping "full_name" -> numpy.ndarray (parameter values).
        Submodules' parameters are namespaced with 'submodule.param'.
        """
        sd: Dict[str, Any] = {}
        for name, param in self.named_parameters(prefix="", recurse=True):
            # param.data is ap.Array -> convert to numpy
            try:
                arr = param.data.asnumpy()
            except Exception:
                # fallback: try to coerce via numpy.array
                arr = np.array(param.data)
            sd[name] = arr
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load a state dict produced by state_dict(). Only assigns to parameters that exist.
        Keys are expected to be namespaced like 'layer.weight' consistent with named_parameters.
        """
        # flatten our named parameters into dict for quick lookup
        local = {name: p for name, p in self.named_parameters(prefix="", recurse=True)}
        for key, val in state_dict.items():
            if key not in local:
                # skip unexpected keys (or optionally raise)
                continue
            p = local[key]
            # assign value - Tensor.data setter will coerce to ap.Array
            p.data = np.array(val)

    # -------------------------
    # mode: train / eval
    # -------------------------
    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "_training", bool(mode))
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    @property
    def training(self) -> bool:
        return bool(getattr(self, "_training", True))

    # -------------------------
    # apply / zero_grad / forward stub
    # -------------------------
    def apply(self, fn: Callable[["Module"], Any]) -> None:
        """
        Apply fn(module) to self and all submodules (depth-first).
        """
        fn(self)
        for m in self._modules.values():
            m.apply(fn)

    def zero_grad(self) -> None:
        """
        Sets gradient of all registered parameters to None (or zeros if you prefer).
        """
        for p in self.parameters(recurse=True):
            try:
                # Many frameworks set .grad = None to indicate cleared grad
                p.grad = None
            except Exception:
                pass

    # -------------------------
    # forward / call
    # -------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Module.forward must be implemented by subclasses")

    def __call__(self, *args, **kwargs):
        # Calls forward so subclasses can be used like functions.
        return self.forward(*args, **kwargs)

    # -------------------------
    # nice repr
    # -------------------------
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        lines: List[str] = []
        for name, module in self._modules.items():
            modstr = repr(module)
            modstr = _indent(modstr, 2)
            lines.append(f"({name}): {modstr}")
        for name, param in self._parameters.items():
            lines.append(f"({name}): Parameter(shape={tuple(param.shape)})")
        if len(lines) == 0:
            return f"{cls}()"
        inner = "\n".join(lines)
        return f"{cls}(\n{_indent(inner, 2)}\n)"


def _indent(text: str, num_spaces: int) -> str:
    pad = " " * num_spaces
    return "\n".join(pad + line for line in text.splitlines())
