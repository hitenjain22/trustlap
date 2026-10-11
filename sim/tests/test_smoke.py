"""Smoke tests: the package imports and JAX can compile and run a function."""

import jax
import jax.numpy as jnp

import trustlap


def test_package_imports() -> None:
    assert trustlap.__version__ == "0.1.0"


def test_jax_jit_runs() -> None:
    @jax.jit
    def double(x: jax.Array) -> jax.Array:
        return 2.0 * x

    out = double(jnp.arange(4.0))
    assert jnp.allclose(out, jnp.array([0.0, 2.0, 4.0, 6.0]))
