import flax.linen as nn
from netket.utils.types import NNInitFunc
from netket.nn.masked_linear import default_kernel_init
from netket import jax as nkjax
import netket as nk
import jax
import jax.numpy as jnp

from typing import Any, Callable, Sequence, Type, Dict
from functools import partial
import netket.experimental as nkx
DType = Any

