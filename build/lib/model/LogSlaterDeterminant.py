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

class LogSlaterDeterminant(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    kernel_init: NNInitFunc = default_kernel_init
    param_dtype: DType = float

    def setup(self):
        self.Mup = self.param('Mup', self.kernel_init, 
                   (self.hilbert.n_orbitals, self.hilbert.n_fermions_per_spin[0]), 
                   self.param_dtype)   
        self.Mdown = self.param('Mdown', self.kernel_init, 
                   (self.hilbert.n_orbitals, self.hilbert.n_fermions_per_spin[1]), 
                   self.param_dtype) 

    @nn.compact
    def __call__(self, n):
        @partial(jnp.vectorize, signature='(n)->()')
        def log_sd(n):
            #Find the positions of the occupied orbitals 
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            Rup = R[:self.hilbert.n_fermions_per_spin[0]]
            Rdown = R[self.hilbert.n_fermions_per_spin[0]:self.hilbert.n_fermions_per_spin[0]+self.hilbert.n_fermions_per_spin[1]] - self.hilbert.n_orbitals
            
            # Extract the Nf x Nf submatrix of M corresponding to the occupied orbitals
            Aup = self.Mup[Rup]
            Adown = self.Mdown[Rdown]

            #return _logdet_cmplx(Aup) + _logdet_cmplx(Adown)
            return nkjax.logdet_cmplx(Aup) + nkjax.logdet_cmplx(Adown)

        return log_sd(n)
