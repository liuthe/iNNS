import netket as nk
import netket.experimental as nkx

from model import LogSlaterDeterminant, LogFullNeuralBackflow, CombinedModel
from hamiltonian.hubbard import Hubbard, Hubbard_extend

from drivers.VMC_infinity import VMCInfinity
import copy

t = 1.0
U = 0.0
N_f = 24

Lx = 4 * 3
Ly = 4

# Define the Hamiltonian
hi, H, graph = Hubbard(t, U, [Lx, Ly], [False, True], (N_f, N_f))
# Define the sampler
#model = LogFullNeuralBackflow.LogFullNeuralBackflow(hi, param_dtype=complex, hidden_units=32)
model = LogSlaterDeterminant.LogSlaterDeterminant(hi, param_dtype=complex)
#sa = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=16, sweep_size=64)
sa = nk.sampler.MetropolisFermionHop(hi, graph=graph)
op = nk.optimizer.Sgd(learning_rate=0.05)
vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)
preconditioner = nk.optimizer.SR(diag_shift=0.05, holomorphic=True)

gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)
#bfsd_log=nk.logging.RuntimeLog()
bfsd_log = '4_4_3_U0_SD'
gs.run(n_iter=100, out=bfsd_log)