from netket.driver import AbstractVariationalDriver

# Package for iter function
from collections.abc import Callable, Iterable
from netket.logging import AbstractLog, JsonLog
from netket.operator._abstract_observable import AbstractObservable

# Package for apply_gradient function
from functools import partial
import jax
from netket import config

CallbackT = Callable[[int, dict, "AbstractVariationalDriver"], bool]

class AbstractDriverInfinity(AbstractVariationalDriver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def iter(self, n_steps, step = 1):
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                self._dp = self._forward_and_backward()
                if i == 0:
                    yield self.step_count

                self._step_count += 1
                self.update_parameters(self._dp)
            self._optimizer_state, self.state.parameters = apply_extend(
                self._optimizer, self.state.parameters
            )

    # def run(
    #     self,
    #     n_iter: int,
    #     out: AbstractLog | Iterable[AbstractLog] | str | None = (),
    #     obs: dict[str, AbstractObservable] | None = None,
    #     step_size: int = 1,
    #     show_progress: bool = True,
    #     save_params_every: int = 50,  # for default logger
    #     write_every: int = 50,  # for default logger
    #     callback: CallbackT | Iterable[CallbackT] = lambda *x: True,
    #     timeit: bool = False,
    # ):
    #     print("Running with custom implementation!")
    #     super().run(n_iter, out, obs, step_size, show_progress, save_params_every, write_every, callback, timeit)
        
@partial(jax.jit, static_argnums=0)
def apply_extend(optimizer, params):
    # Update the environment backflow with the system backflow
    new_params = {
        **params,
        'env_backflow': jax.tree_util.tree_map(
            lambda env_subtree, sys_subtree: sys_subtree,
            params['env_backflow'],
            params['sys_backflow']
        )
    }
    # Update the optimizer state
    new_optimizer_state = optimizer.init(new_params)

    if config.netket_experimental_sharding:
        sharding = jax.sharding.PositionalSharding(jax.devices()).replicate()
        new_optimizer_state = jax.lax.with_sharding_constraint(
            new_optimizer_state, sharding
        )
        new_params = jax.lax.with_sharding_constraint(new_params, sharding)

    return new_optimizer_state, new_params