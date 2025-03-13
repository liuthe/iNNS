import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
#import netket.experimental as nkx
from ml_collections import ConfigDict
from frozendict import frozendict
import jax.numpy as jnp
import numpy as np

from model.LogTransformer import Transformer
from hamiltonian.hubbard import Hubbard, Hubbard_extend

import copy

t = 1.0
U = 8.0
N_f = 8

Lx = 4
Ly = 4

# Define the Hamiltonian
hi, H, graph = Hubbard(t, U, [Lx, Ly], [True, True], (N_f, N_f))
H = H.to_jax_operator()
# Define the model
model_config = dict(
        # Mask parameters
        causal_mask=False,
        # Embedding parameters
        do_emb=False, vocab_size=2*N_f, embedding_size=2,
        # Positional encoding parameters
        do_pos_emb=False, max_seq_len=2*Lx*Ly,
        # Linear mixing parameters
        do_mix=True, hidden_size=2,
        # Transformer block parameters
        num_layers=0,
        scan_layers=True,
        #remat=(""),
        remat=("MLP", "Attn"),
        ## MLP block parameters
        mlp_expansion=1,
        ## Attention block parameters
        head_dim=8,
        # Output layer parameters
        num_outputs=2*N_f,
        # Final determinant parameters
        do_det=True,
        # Data dtype
        dtype=jnp.bfloat16,
        out_dtype=jnp.float32,
        softmax_dtype=jnp.float32, 
)
model_config['num_heads'] = model_config['hidden_size'] // model_config['head_dim']

#frozen_data_config = frozendict(data_config)
frozen_model_config = frozendict(model_config)
model = Transformer(frozen_model_config)

#sa = nkx.sampler.MetropolisParticleExchange(hi, graph=graph, n_chains=16, exchange_spins=False, sweep_size=64)
sa = nk.sampler.MetropolisFermionHop(
    hi, graph=graph
    , dtype=np.int8, n_chains=16, sweep_size=64
)
op = nk.optimizer.Sgd(learning_rate=0.05)
vstate = nk.vqs.MCState(sa, model, 
                        n_samples=2**12, 
                        n_discard_per_chain=16)
preconditioner = nk.optimizer.SR(diag_shift=0.05)

gs = nk.VMC(H, op, 
            variational_state=vstate, 
            preconditioner=preconditioner
            )
bfsd_log = '4_4_U8_N8_pbc_test'
gs.run(n_iter=500, out=bfsd_log)