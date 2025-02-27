import netket as nk
import netket.experimental as nkx

from model import LogSlaterDeterminant, LogFullNeuralBackflow, CombinedModel
from hamiltonian.hubbard import Hubbard, Hubbard_extend

t = 1.0
U = 0.0
N_f = 8

Lx = 4
Ly = 4

# Define the Hamiltonian
hi, H, graph = Hubbard(t, U, [Lx, Ly], [True, True], (N_f, N_f))
exchange_graph = nk.graph.disjoint_union(graph, graph)
# Define the sampler
model = LogSlaterDeterminant.LogSlaterDeterminant(hi, param_dtype=complex)

#sa = nkx.sampler.MetropolisParticleExchange(hi, graph=graph, n_chains=16, exchange_spins=False, sweep_size=64)
#sa =  nk.sampler.MetropolisExchange(
#    hi, graph=exchange_graph, n_chains=16, sweep_size=64)
sa = nk.sampler.MetropolisFermionHop(
    hi, graph=graph
)
op = nk.optimizer.Sgd(learning_rate=0.01)
vstate = nk.vqs.MCState(sa, model, n_samples=512, n_discard_per_chain=16)
#print(vstate.parameters)
preconditioner = nk.optimizer.SR(diag_shift=0.05, holomorphic=True)

gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)
#bfsd_log=nk.logging.RuntimeLog()
bfsd_log = '4_4_U0_N8_pbc_test'
gs.run(n_iter=200, out=bfsd_log,step_size=1)
