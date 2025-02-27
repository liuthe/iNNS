import netket as nk
import netket.experimental as nkx

import jax
import jax.numpy as jnp

from model import LogFullNeuralBackflow
from hamiltonian.hubbard import Hubbard

t = 1.0
U = 8.0
N_f = 7

Lx = 4
Ly = 4

# Define the Hamiltonian
hi, H, graph = Hubbard(t, U, [Lx, Ly], [True, True], (N_f, N_f))
N = graph.n_nodes

model = LogFullNeuralBackflow.LogFullNeuralBackflow(hi, hidden_units=N)
sa = nkx.sampler.MetropolisParticleExchange(
    hi, graph=graph, n_chains=16, exchange_spins=False, sweep_size=64
)
op = nk.optimizer.Sgd(learning_rate=0.05)
vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)
preconditioner = nk.optimizer.SR(diag_shift=0.05)
gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)
bf_log='original_test_Nf7'
gs.run(n_iter=2000, out=bf_log)
