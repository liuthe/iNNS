import flax.linen as nn
from netket.utils.types import NNInitFunc
from netket.nn.masked_linear import default_kernel_init
from netket import jax as nkjax
import jax
import jax.numpy as jnp

from typing import Any, Callable, Sequence, Type, Dict
from functools import partial
import netket.experimental as nkx
DType = Any

from dataclasses import field

class CombinedNeuralBackflow(nn.Module):
    hilbert: nkx.hilbert.SpinOrbitalFermions
    sys_backflow: nn.Module
    env_backflow: nn.Module
    #backflow_module: nn.Module  # The module class to use for the backflow 
    kernel_init: NNInitFunc = nn.initializers.normal()
    param_dtype: DType = jnp.float32
    
    @nn.compact
    def __call__(self, n):
        @partial(jnp.vectorize, signature='(n)->()')
        def log_sd(n):
            n_orbitals = self.hilbert.n_orbitals
            l, s, r = n[..., :2*n_orbitals], n[..., 2*n_orbitals:4*n_orbitals], n[..., 4*n_orbitals:]
            output_l = jax.lax.stop_gradient(self.env_backflow(l))
            output_s = self.sys_backflow(s)
            output_r = jax.lax.stop_gradient(self.env_backflow(r))
            return output_l + output_s + output_r
        return log_sd(n)