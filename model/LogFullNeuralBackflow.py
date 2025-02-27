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

class LogFullNeuralBackflow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    hidden_units: int
    kernel_init: NNInitFunc = default_kernel_init
    param_dtype: DType = float   
    
    @nn.compact
    def __call__(self, n):
        @partial(jnp.vectorize, signature='(n)->()')
        def log_sd(n):
            # Construct the Backflow. Takes as input strings of $N$ occupation numbers, outputs an $N x Nf$ matrix
            # that modifies the bare orbitals.
            F = nn.Dense(self.hidden_units, param_dtype=self.param_dtype)(n)
            F = jax.nn.tanh(F)
            # last layer, outputs N x Nf values
            F = nn.Dense(2 * self.hilbert.n_orbitals * self.hilbert.n_fermions, param_dtype=self.param_dtype)(F)
            
            #Find the positions of the occupied, backflow-modified orbitals
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            A = F.reshape((2 * self.hilbert.n_orbitals, self.hilbert.n_fermions))[R]
            return nkjax.logdet_cmplx(A)

        return log_sd(n)