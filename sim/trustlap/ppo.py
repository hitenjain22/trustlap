"""PureJaxRL-style PPO (SPEC 7.2): rollout, GAE and update epochs in one jitted function.

Everything here is per agent and `vmap`-able over a population axis (PBT, M4): parameters,
optimizer state, observation statistics and PRNG streams all live in `TrainState`, and the
tunable hyperparameters (learning rate, entropy coefficient) are passed as arrays so changing
them never triggers recompilation.

Must-get-right details (SPEC 7.2):
* GAE masks the bootstrap with `terminated`; on truncation (time limit) it bootstraps from
  V(s_final), the value of the pre-reset observation the env returns.
* Observations are normalized with running statistics (Welford, merged per rollout batch);
  the statistics are exported with the weights and frozen at deployment.
* Rewards are divided by the running std of the discounted return (training only).
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from trustlap.config import CONFIG, Config
from trustlap.env import Env, EnvState, Observation
from trustlap.networks import ActorCritic
from trustlap.track import TrackBank

OBS_CLIP = 10.0
LOG_2PI_E = math.log(2 * math.pi * math.e)


# ---------------------------------------------------------------------------------------------
# Running statistics
# ---------------------------------------------------------------------------------------------


class RunningStats(NamedTuple):
    """Running mean/variance (Chan et al. parallel merge of Welford statistics)."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array


def stats_init(size: int) -> RunningStats:
    return RunningStats(jnp.zeros(size), jnp.ones(size), jnp.asarray(1e-4))


def stats_update(stats: RunningStats, batch: jax.Array) -> RunningStats:
    """Merge a batch [N, size] (or [N]) into the running statistics."""
    b_mean = jnp.mean(batch, axis=0)
    b_var = jnp.var(batch, axis=0)
    b_count = batch.shape[0]
    delta = b_mean - stats.mean
    total = stats.count + b_count
    mean = stats.mean + delta * b_count / total
    m2 = stats.var * stats.count + b_var * b_count + delta**2 * stats.count * b_count / total
    return RunningStats(mean, m2 / total, total)


def normalize(stats: RunningStats, x: jax.Array) -> jax.Array:
    """(x − mean) / std, clipped to ±10."""
    return jnp.clip((x - stats.mean) / jnp.sqrt(stats.var + 1e-8), -OBS_CLIP, OBS_CLIP)


# ---------------------------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------------------------


class TrainState(NamedTuple):
    """Everything one PPO agent carries between updates."""

    params: Any
    opt_state: Any
    obs_stats: RunningStats  # policy observation (33)
    priv_stats: RunningStats  # privileged observation (14)
    ret_stats: RunningStats  # discounted return (scalar), for reward scaling
    ret: jax.Array  # [N] running discounted return per env
    env_state: EnvState
    obs: Observation
    key: jax.Array
    ep_return: jax.Array  # [N] raw return of the current episode
    update: jax.Array  # [] updates done


class Transition(NamedTuple):
    obs_raw: jax.Array
    priv_raw: jax.Array
    obs: jax.Array  # normalized (as the policy saw it)
    priv: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    v_next: jax.Array  # V(pre-reset next observation): the bootstrap value
    reward: jax.Array  # scaled
    terminated: jax.Array
    done: jax.Array  # terminated or truncated: cuts the GAE trace


def gaussian_logp(x: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Log density of a diagonal Gaussian, summed over action dims (unclipped sample)."""
    z = (x - mean) * jnp.exp(-log_std)
    return jnp.sum(-0.5 * z * z - log_std - 0.5 * math.log(2 * math.pi), axis=-1)


def gae(traj: Transition, gamma: float, lam: float) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation over a [T, N] trajectory. Returns (advantages, targets).

    δ_t = r_t + γ · V_next_t · (1 − terminated_t) − V_t
    A_t = δ_t + γλ · (1 − done_t) · A_{t+1}
    """

    def body(adv_next: jax.Array, t: Transition) -> tuple[jax.Array, jax.Array]:
        delta = t.reward + gamma * t.v_next * (1.0 - t.terminated) - t.value
        adv = delta + gamma * lam * (1.0 - t.done) * adv_next
        return adv, adv

    _, adv = jax.lax.scan(body, jnp.zeros_like(traj.value[0]), traj, reverse=True)
    return adv, adv + traj.value


def make_optimizer(cfg: Config = CONFIG) -> optax.GradientTransformation:
    """Grad-norm clipping, then Adam with an injectable learning rate."""
    return optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=cfg.ppo.lr, eps=1e-5),
    )


def set_learning_rate(opt_state: Any, lr: jax.Array) -> Any:
    clip_state, adam_state = opt_state
    hp = dict(adam_state.hyperparams)
    hp["learning_rate"] = jnp.asarray(lr, jnp.float32)
    return (clip_state, adam_state._replace(hyperparams=hp))


# ---------------------------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------------------------


class PPO:
    """PPO for one agent on a batched `Env`. Methods are pure and jit/vmap-friendly.

    `n_envs` cars per agent, rollouts of `rollout_len` steps, `minibatches` × `epochs` updates.
    """

    def __init__(
        self,
        env: Env,
        privileged_actor: bool = False,
        n_envs: int | None = None,
        rollout_len: int | None = None,
        cfg: Config = CONFIG,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.n_envs = n_envs or cfg.ppo.envs_per_agent
        self.rollout_len = rollout_len or cfg.ppo.rollout_len
        self.net = ActorCritic(privileged_actor=privileged_actor, cfg=cfg.network)
        self.optimizer = make_optimizer(cfg)
        batch = self.n_envs * self.rollout_len
        assert batch % cfg.ppo.minibatches == 0, "batch must split evenly into minibatches"

    def init(self, key: jax.Array, difficulty: jax.Array, stretch: jax.Array) -> TrainState:
        """Fresh parameters and environments. `difficulty`/`stretch`: [n_envs]."""
        k_net, k_env, k_run = jax.random.split(key, 3)
        o = self.cfg.obs
        params = self.net.init(k_net, jnp.zeros(o.policy_size), jnp.zeros(o.privileged_size))
        env_state, obs = self.env._reset(
            jax.random.split(k_env, self.n_envs), difficulty, stretch, self.env.bank
        )
        return TrainState(
            params=params,
            opt_state=self.optimizer.init(params),
            obs_stats=stats_init(o.policy_size),
            priv_stats=stats_init(o.privileged_size),
            ret_stats=stats_init(1)._replace(mean=jnp.zeros(()), var=jnp.ones(())),
            ret=jnp.zeros(self.n_envs),
            env_state=env_state,
            obs=obs,
            key=k_run,
            ep_return=jnp.zeros(self.n_envs),
            update=jnp.zeros((), jnp.int32),
        )

    # The jitted update ---------------------------------------------------------------------
    def update(
        self,
        ts: TrainState,
        difficulty: jax.Array,
        stretch: jax.Array,
        lr: jax.Array,
        ent_coef: jax.Array,
        bank: TrackBank,
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        """One PPO update: collect `rollout_len` steps from every env, then optimize."""
        p = self.cfg.ppo
        net = self.net

        def env_step(carry: TrainState, _: Any) -> tuple[TrainState, tuple[Transition, dict]]:
            ts = carry
            key, k_act, k_env = jax.random.split(ts.key, 3)
            obs_n = normalize(ts.obs_stats, ts.obs.policy)
            priv_n = normalize(ts.priv_stats, ts.obs.privileged)
            mean, log_std, value = net.apply(ts.params, obs_n, priv_n)
            action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            logp = gaussian_logp(action, mean, log_std)
            env_state, obs, reward, info, final_obs = self.env._step(
                ts.env_state,
                action,
                jax.random.split(k_env, self.n_envs),
                difficulty,
                stretch,
                bank,
            )
            v_next = net.apply(
                ts.params,
                normalize(ts.obs_stats, final_obs.policy),
                normalize(ts.priv_stats, final_obs.privileged),
                method=ActorCritic.value,
            )
            done = info.terminated | info.truncated
            # Reward scaling by the running std of the discounted return (PureJaxRL style).
            ret = ts.ret * p.gamma + reward
            ret_stats = stats_update(ts.ret_stats, ret)
            scaled = reward / jnp.sqrt(ret_stats.var + 1e-8)
            ep_return = ts.ep_return + reward
            stats = {
                "episodes": jnp.sum(done),
                "return_sum": jnp.sum(jnp.where(done, ep_return, 0.0)),
                "length_sum": jnp.sum(jnp.where(done, info.steps, 0)),
                "progress_sum": jnp.sum(jnp.where(done, info.progress_m, 0.0)),
                "laps": jnp.sum(info.lap_completed),
                "crashes": jnp.sum(info.crashed),
                "stalls": jnp.sum(info.stalled),
                "wrong_way": jnp.sum(info.wrong_way),
                "truncations": jnp.sum(info.truncated),
                "nan_resets": jnp.sum(info.nan_reset),
                "speed_sum": jnp.sum(info.speed_mps),
                # Curriculum success counts only cars at the curriculum difficulty.
                "cur_episodes": jnp.sum(
                    done & (difficulty > 0) | done & (jnp.max(difficulty) == 0)
                ),
                "cur_success": jnp.sum(
                    (info.lap_completed | info.truncated)
                    & ((difficulty > 0) | (jnp.max(difficulty) == 0))
                ),
            }
            transition = Transition(
                obs_raw=ts.obs.policy,
                priv_raw=ts.obs.privileged,
                obs=obs_n,
                priv=priv_n,
                action=action,
                logp=logp,
                value=value,
                v_next=v_next,
                reward=scaled,
                terminated=info.terminated.astype(jnp.float32),
                done=done.astype(jnp.float32),
            )
            ts = ts._replace(
                key=key,
                env_state=env_state,
                obs=obs,
                ret=jnp.where(done, 0.0, ret),
                ret_stats=ret_stats,
                ep_return=jnp.where(done, 0.0, ep_return),
            )
            return ts, (transition, stats)

        ts, (traj, step_stats) = jax.lax.scan(env_step, ts, None, length=self.rollout_len)
        advantages, targets = gae(traj, p.gamma, p.gae_lambda)

        # Optimize: epochs × shuffled minibatches.
        def loss_fn(
            params: Any, mb: Transition, adv: jax.Array, target: jax.Array
        ) -> tuple[jax.Array, dict]:
            mean, log_std, value = net.apply(params, mb.obs, mb.priv)
            logp = gaussian_logp(mb.action, mean, log_std)
            log_ratio = logp - mb.logp
            ratio = jnp.exp(log_ratio)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = -jnp.mean(
                jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - p.clip_eps, 1 + p.clip_eps) * adv)
            )
            v_loss = 0.5 * jnp.mean((value - target) ** 2)
            entropy = jnp.sum(log_std + 0.5 * LOG_2PI_E)
            total = pg_loss + p.vf_coef * v_loss - ent_coef * entropy
            return total, {
                "policy_loss": pg_loss,
                "value_loss": v_loss,
                "entropy": entropy,
                "approx_kl": jnp.mean((ratio - 1) - log_ratio),
                "clip_frac": jnp.mean(jnp.abs(ratio - 1) > p.clip_eps),
            }

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        opt_state = set_learning_rate(ts.opt_state, lr)
        batch = self.n_envs * self.rollout_len
        flat = jax.tree.map(lambda x: x.reshape((batch,) + x.shape[2:]), traj)
        flat_adv = advantages.reshape(batch)
        flat_target = targets.reshape(batch)

        def epoch(
            carry: tuple[Any, Any, jax.Array], _: Any
        ) -> tuple[tuple[Any, Any, jax.Array], dict]:
            params, opt_state, key = carry
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, batch).reshape(p.minibatches, -1)

            def minibatch(carry: tuple[Any, Any], idx: jax.Array) -> tuple[tuple[Any, Any], dict]:
                params, opt_state = carry
                mb = jax.tree.map(lambda x: x[idx], flat)
                (_, aux), grads = grad_fn(params, mb, flat_adv[idx], flat_target[idx])
                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                aux["grad_norm"] = optax.global_norm(grads)
                return (optax.apply_updates(params, updates), opt_state), aux

            (params, opt_state), aux = jax.lax.scan(minibatch, (params, opt_state), perm)
            return (params, opt_state, key), aux

        key, k_epochs = jax.random.split(ts.key)
        (params, opt_state, _), losses = jax.lax.scan(
            epoch, (ts.params, opt_state, k_epochs), None, length=p.update_epochs
        )

        # Statistics used next update (this rollout was normalized with the previous ones).
        obs_stats = stats_update(ts.obs_stats, flat.obs_raw)
        priv_stats = stats_update(ts.priv_stats, flat.priv_raw)

        sums = {k: jnp.sum(v) for k, v in step_stats.items()}
        var_t = jnp.var(flat_target)
        metrics = {
            **sums,
            **{k: jnp.mean(v) for k, v in losses.items()},
            "explained_variance": 1.0 - jnp.var(flat_target - flat.value) / (var_t + 1e-8),
            "log_std_mean": jnp.mean(params["params"]["log_std"]),
            "params_finite": jnp.all(
                jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(params)])
            ),
        }
        ts = ts._replace(
            params=params,
            opt_state=opt_state,
            obs_stats=obs_stats,
            priv_stats=priv_stats,
            key=key,
            update=ts.update + 1,
        )
        return ts, metrics


def policy_action(
    net: ActorCritic,
    params: Any,
    obs_stats: RunningStats,
    priv_stats: RunningStats,
    obs: Observation,
) -> jax.Array:
    """Deterministic (mean) action for deployment and evaluation (SPEC 7.1)."""
    obs_n = normalize(obs_stats, obs.policy)
    priv_n = normalize(priv_stats, obs.privileged)
    mean, _, _ = net.apply(params, obs_n, priv_n)
    return jnp.clip(mean, -1.0, 1.0)
