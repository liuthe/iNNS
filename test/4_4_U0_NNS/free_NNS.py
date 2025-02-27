import netket as nk
#import netket.experimental as nkx
import numpy as np

from model import LogSlaterDeterminant, LogFullNeuralBackflow, CombinedModel
from hamiltonian.hubbard import Hubbard, Hubbard_extend

t = 1.0
U = 0.0
N_f = 8

Lx = 4
Ly = 4

# Define the Hamiltonian
hi, H, graph = Hubbard(t, U, [Lx, Ly], [True, True], (N_f, N_f))

# Define the sampler
# obc hidden 16, pbc hidden 32
model = LogFullNeuralBackflow.LogFullNeuralBackflow(hi,
                                                    #param_dtype=complex, 
                                                    hidden_units=32
                                                    )

sa = nk.sampler.MetropolisFermionHop(
    hi, graph=graph
    #, dtype=np.int8, n_chains=16, sweep_size=64
)
op = nk.optimizer.Sgd(learning_rate=0.05)
vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)
preconditioner = nk.optimizer.SR(diag_shift=0.05#, 
                                 #holomorphic=True # this line should be removed when N is large 
                                 )

gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)
bfsd_log = '4_4_U0_N8_pbc_test'
gs.run(n_iter=300, out=bfsd_log,step_size=1)
