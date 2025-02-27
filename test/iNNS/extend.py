import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
import netket.experimental as nkx

from model import LogSlaterDeterminant, LogFullNeuralBackflow, CombinedModel
from hamiltonian.hubbard import Hubbard, Hubbard_extend

from drivers.VMC_infinity import VMCInfinity
import copy

t = 1.0
U = 8.0
N_f = 8

Lx = 4
Ly = 4

# Define the Hamiltonian
hi, H, graph = Hubbard(t, U, [Lx, Ly], [False, True], (N_f, N_f))
hi_help, H_full, graph_full = Hubbard_extend(t, U, [Lx, Ly], [False, True], (N_f, N_f))
H = H.to_jax_operator()
H_full = H_full.to_jax_operator()
# Define the sampler
model_base = LogFullNeuralBackflow.LogFullNeuralBackflow(hi, 
                                                         #param_dtype=complex, 
                                                         hidden_units=16)
#model_base = LogFullNeuralBackflow(hi, param_dtype=complex, hidden_units=16)
sys_backflow = copy.deepcopy(model_base)
env_backflow = copy.deepcopy(model_base)
model = CombinedModel.CombinedNeuralBackflow(hi_help, 
                                             sys_backflow=sys_backflow, 
                                             env_backflow=env_backflow, 
                                             #param_dtype=complex
                                             )
#sa = nkx.sampler.MetropolisParticleExchange(hi_help, graph=graph, n_chains=16, exchange_spins=False, sweep_size=64)
sa =  nk.sampler.MetropolisExchange(hi_help, graph=graph_full, n_chains=16, sweep_size=64)
# sa = nk.sampler.MetropolisFermionHop(
#     hi_help, graph=graph
#     #, dtype=np.int8, n_chains=16, sweep_size=64
# )
vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)
preconditioner = nk.optimizer.SR(
    diag_shift=0.05, 
    #holomorphic=True
    )
op = nk.optimizer.Sgd(learning_rate=0.01)
gs_inf = VMCInfinity(H_full, op, 
                     variational_state=vstate, 
                     preconditioner=preconditioner)
#bfsd_log=nk.logging.RuntimeLog()
bfsd_log = '4_4_U8_N8_iNNS'
gs_inf.run(n_iter=2000, out=bfsd_log,step_size=2)