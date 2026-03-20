# cattorch/__init__.py
import os  # noqa: F401

try:
    import cupy as cp  # type: ignore  # noqa: F401
except ImportError:
    pass  # cupy is not available

# ##### QUICK ENABLE FOR TENSOR CORE OPS ###
# device = cp.cuda.Device()
# # string containing the major index and the minor index.
# # For example, compute capability 3.5 is represented by the string ‘35’.
# cc_major, cc_minor = device.compute_capability
# if int(cc_major) >= 8:
#     os.environ["CUPY_TF32"] = "1"
##########################################

### Disable Warnings from Selected Packages ###
import warnings

from ._constants import *  # noqa: F403
from .backend.dtypes import *  # noqa: F403
from .einsum import einsum as einsum
from .nn.functional import *  # noqa: F403
from .tensor import (
    Tensor as Tensor,
)
from .tensor import (
    arange as arange,
)
from .tensor import (
    empty as empty,
)
from .tensor import (
    empty_like as empty_like,
)
from .tensor import (
    eye as eye,
)
from .tensor import (
    full as full,
)
from .tensor import (
    full_like as full_like,
)
from .tensor import (
    linspace as linspace,
)
from .tensor import (
    no_grad as no_grad,
)
from .tensor import (
    ones as ones,
)
from .tensor import (
    ones_like as ones_like,
)
from .tensor import (
    rand as rand,
)
from .tensor import (
    rand_like as rand_like,
)
from .tensor import (
    randn_like as randn_like,
)
from .tensor import (
    randint as randint,
)
from .tensor import (
    randn as randn,
)

from .tensor import (
    tril as tril,
)
from .tensor import (
    zeros as zeros,
)
from .tensor import (
    zeros_like as zeros_like,
)

from . import nn as nn

warnings.filterwarnings("ignore", module="pydantic")
###############################################
