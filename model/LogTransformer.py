# Netket imports
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

# Gpu configurations
from utils.utils import install_package, set_XLA_flags_gpu
set_XLA_flags_gpu()

# Transformer imports
import functools
from typing import Any, Dict, Tuple

# Install ml_collections on colab
from ml_collections import ConfigDict

# Type aliases
PyTree = Any
Metrics = Dict[str, Tuple[jax.Array, ...]]

from utils.single_gpu import Batch, TrainState, accumulate_gradients, print_metrics

class MLPBlock(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        input_features = x.shape[-1]
        x = nn.LayerNorm(dtype=self.config.dtype, name="pre_norm")(x)
        x = nn.Dense(
            features=self.config.mlp_expansion * input_features,
            dtype=self.config.dtype,
            name="input_layer",
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            features=input_features,
            dtype=self.config.dtype,
            name="output_layer",
        )(x)
        x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(x)
        return x
    
def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array | None,
    softmax_dtype: jnp.dtype = jnp.float32,
):
    """Dot-product attention.

    Follows the setup of https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.dot_product_attention,
    but supports switch to float32 for numerical stability during softmax.

    Args:
        query: The query array, shape [..., num queries, num heads, hidden size].
        key: The key array, shape [..., num keys, num heads, hidden size].
        value: The value array, shape [..., num keys, num heads, hidden size].
        mask: The boolean mask array (0 for masked values, 1 for non-masked). If None, no masking is applied.
        softmax_dtype: The dtype to use for the softmax and dot-product operation.

    Returns:
        The attention output array, shape [..., num queries, num heads, hidden size].
    """
    num_features = query.shape[-1]
    dtype = query.dtype
    scale = num_features**-0.5
    query = query * scale
    # Switch dtype right before the dot-product for numerical stability.
    query = query.astype(softmax_dtype)
    key = key.astype(softmax_dtype)
    weights = jnp.einsum("...qhd,...khd->...hqk", query, key)
    if mask is not None:
        weights = jnp.where(mask, weights, jnp.finfo(softmax_dtype).min)
    weights = nn.softmax(weights, axis=-1)
    # After softmax, switch back to the original dtype
    weights = weights.astype(dtype)
    new_vals = jnp.einsum("...hqk,...khd->...qhd", weights, value)
    new_vals = new_vals.astype(dtype)
    return new_vals

class AttentionBlock(nn.Module):
    config: ConfigDict
    mask: jax.Array | None
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        input_features = x.shape[-1]
        x = nn.LayerNorm(dtype=self.config.dtype, name="pre_norm")(x)
        qkv = nn.DenseGeneral(
            features=(self.config.num_heads, self.config.head_dim * 3),
            dtype=self.config.dtype,
            name="qkv",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        x = dot_product_attention(q, k, v, mask=self.mask, softmax_dtype=self.config.softmax_dtype)
        x = nn.DenseGeneral(
            features=input_features,
            axis=(-2, -1),
            dtype=self.config.dtype,
            name="output_layer",
        )(x)
        x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(x)
        return x
    
class TransformerBlock(nn.Module):
    config: ConfigDict
    mask: jax.Array | None
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # MLP block
        mlp = MLPBlock
        if "MLP" in self.config.remat:
            mlp = nn.remat(mlp, prevent_cse=False)
        x = x + mlp(config=self.config, train=self.train, name="mlp")(x)
        # Attention block
        attn = AttentionBlock
        if "Attn" in self.config.remat:
            attn = nn.remat(attn, prevent_cse=False)
        x = x + attn(config=self.config, mask=self.mask, train=self.train, name="attn")(x)
        return x

class Transformer(nn.Module):
    config: ConfigDict

    @nn.compact
    def __call__(
        self, n: jax.Array, mask: jax.Array | None = None, train: bool = True
    ) -> jax.Array:
        if mask is None and self.config.causal_mask:
            mask = nn.make_causal_mask(n, dtype=jnp.bool_)
        # Input layer.
        x = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=self.config.dtype,
            name="embed",
        )(n)
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.config.max_seq_len, self.config.hidden_size),
        )
        pos_emb = pos_emb.astype(self.config.dtype)
        x = x + pos_emb[None, : x.shape[1]]
        # Transformer blocks.
        block_fn = functools.partial(TransformerBlock, config=self.config, mask=mask, train=train)
        if "Block" in self.config.remat:
            block_fn = nn.remat(block_fn, prevent_cse=False)
        if self.config.scan_layers:
            block = block_fn(name="block")
            x, _ = nn.scan(
                lambda module, carry, _: (module(carry), None),
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                length=self.config.num_layers,
            )(block, x, ())
        else:
            for l_idx in range(self.config.num_layers):
                x = block_fn(name=f"block_{l_idx}")(x)
        # Output layer.
        x = nn.LayerNorm(dtype=self.config.dtype, name="post_norm")(x)
        x = nn.Dense(
            features=self.config.num_outputs,
            dtype=self.config.dtype,
            name="output_layer",
        )(x)
        
        # Determinant layer
        x = x.astype(self.config.out_dtype)
        if self.config.determinant:
            filling = self.config.num_outputs
            @partial(jnp.vectorize, signature='(n),(n,m)->()')
            def logdet(n, out):
                R =  n.nonzero(size = filling)[0]
                A = out[R]
                return nkjax.logdet_cmplx(A)
            x = logdet(n, x)
        return x