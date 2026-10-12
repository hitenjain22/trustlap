"""Neural networks (SPEC 7.1). Inference is ported to `web/src/policy/*.ts`.

* Actor: obs (33) [+ z (8) for teacher/student] → MLP 3×128 ELU → action mean (2), plus a
  state-independent log std (init −0.5).
* Critic: obs (33) + privileged (14) → MLP 3×128 ELU → V. Every critic sees the privileged
  vector during training ("asymmetric critic"); the deployed policy never needs it.
* Privileged encoder (teacher): privileged (14) → 64 ELU → 8, tanh.

Layer names are stable (`actor_0`, …) because `export.py` writes them in this order.
Init: orthogonal (gain √2 hidden, 0.01 actor output, 1.0 critic output), zero biases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from trustlap.config import CONFIG, NetworkConfig

ROOT2 = math.sqrt(2.0)


def dense(features: int, gain: float, name: str) -> nn.Dense:
    return nn.Dense(
        features,
        kernel_init=nn.initializers.orthogonal(gain),
        bias_init=nn.initializers.zeros,
        name=name,
    )


class MLP(nn.Module):
    """Hidden ELU layers, then a linear output. Layers are named `{prefix}_{i}`, `{prefix}_out`."""

    hidden: Sequence[int]
    out: int
    out_gain: float
    prefix: str

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for i, width in enumerate(self.hidden):
            x = nn.elu(dense(width, ROOT2, f"{self.prefix}_{i}")(x))
        return dense(self.out, self.out_gain, f"{self.prefix}_out")(x)


class ActorCritic(nn.Module):
    """Policy and value networks. `privileged_actor=True` makes it the teacher (actor sees z)."""

    privileged_actor: bool = False
    cfg: NetworkConfig = CONFIG.network
    action_size: int = 2

    def setup(self) -> None:
        c = self.cfg
        self.actor = MLP(c.hidden, self.action_size, 0.01, "actor")
        self.critic = MLP(c.hidden, 1, 1.0, "critic")
        if self.privileged_actor:
            self.encoder = MLP(c.encoder_hidden, c.latent_size, ROOT2, "encoder")
        self.log_std = self.param(
            "log_std", nn.initializers.constant(c.log_std_init), (self.action_size,)
        )

    def encode(self, privileged: jax.Array) -> jax.Array:
        """z = tanh(encoder(privileged)), shape [..., 8]."""
        return jnp.tanh(self.encoder(privileged))

    def act(self, obs: jax.Array, z: jax.Array | None = None) -> jax.Array:
        """Action mean from obs (and z when the actor takes one)."""
        x = obs if z is None else jnp.concatenate([obs, z], axis=-1)
        return self.actor(x)

    def with_latent(
        self, obs: jax.Array, z: jax.Array, privileged: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """(action mean, log std, value) acting on a given latent z instead of the encoder's
        (fine-tuning the actor on the student's estimate; the critic still sees privileged)."""
        return self.act(obs, z), self.log_std, self.value(obs, privileged)

    def value(self, obs: jax.Array, privileged: jax.Array) -> jax.Array:
        return self.critic(jnp.concatenate([obs, privileged], axis=-1))[..., 0]

    def __call__(
        self, obs: jax.Array, privileged: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """(action mean, log std, value) for normalized obs and privileged inputs."""
        z = self.encode(privileged) if self.privileged_actor else None
        return self.act(obs, z), self.log_std, self.value(obs, privileged)
