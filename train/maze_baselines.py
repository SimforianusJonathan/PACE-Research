"""MiniGrid curriculum branches adapted from the pinned JAX-UED examples."""

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jaxued.environments.maze import make_level_mutator_minimax
from jaxued.utils import max_mc


METHOD_SCORES = {"dr": None, "plr": "l1_value_loss", "plr_perp": "maxmc", "accel": "maxmc"}
NEW, REPLAY, MUTATION = range(3)


class BaselineTrainState(TrainState):
    sampler: dict
    replay_batches: jax.Array
    mutation_batches: jax.Array
    mutation_pending: jax.Array
    parent_levels: object


def level_score(method, dones, values, advantages, max_returns):
    if method == "plr":
        return jnp.abs(advantages).mean(axis=0)
    return max_mc(dones, values, max_returns)


def make_baseline_step(config, sampler, generate, collect, ppo, record_metrics):
    mutator = make_level_mutator_minimax(100) if config.method == "accel" else None
    num_envs = config.num_envs
    empty_indices = jnp.zeros(num_envs, jnp.int32)

    def step(state, _):
        student = state.student
        rng = state.rng
        if config.method == "dr":
            branch = jnp.int32(NEW)
        else:
            rng, decision_key = jax.random.split(rng)
            replay = sampler.sample_replay_decision(student.sampler, decision_key)
            branch = jnp.where(student.mutation_pending, MUTATION, replay.astype(jnp.int32))
        rng, level_key, reset_key = jax.random.split(rng, 3)

        def new_levels():
            return student.sampler, empty_indices, jax.vmap(generate)(jax.random.split(level_key, num_envs))

        def replay_levels():
            buffer, (indices, levels) = sampler.sample_replay_levels(student.sampler, level_key, num_envs)
            return buffer, indices, levels

        def mutate_levels():
            levels = jax.vmap(mutator, (0, 0, None))(
                jax.random.split(level_key, num_envs), student.parent_levels, config.mutation_edits,
            )
            return student.sampler, empty_indices, levels

        if config.method == "dr":
            buffer, indices, levels = new_levels()
        else:
            branches = (new_levels, replay_levels, mutate_levels) if mutator else (new_levels, replay_levels)
            buffer, indices, levels = jax.lax.switch(branch, branches)

        rng, carry, batch, stats, max_returns = collect(
            rng, reset_key, student, levels, num_envs, measure=record_metrics,
        )
        replay = branch == REPLAY
        if config.method == "dr":
            scores = jnp.zeros(num_envs, jnp.float32)
        else:
            _, _, dones, _, values, _, advantages = batch
            if config.score_type == "maxmc":
                historic = sampler.get_levels_extra(buffer, indices)["max_return"]
                max_returns = jnp.where(replay, jnp.maximum(historic, max_returns), max_returns)
                extra = {"max_return": max_returns}
            else:
                extra = None
            scores = level_score(config.method, dones, values, advantages, max_returns)
            buffer = jax.lax.cond(
                replay, lambda: sampler.update_batch(buffer, indices, scores, extra),
                lambda: sampler.insert_batch(buffer, levels, scores, extra)[0],
            )

        trained = replay | config.exploratory_grad_updates

        def skip_ppo(key, student, *_):
            # Match the PRNG consumption of the upstream no-gradient PPO epochs.
            key = jax.lax.fori_loop(0, config.ppo_epochs, lambda _, k: jax.random.split(k)[0], key)
            return key, student

        rng, student = jax.lax.cond(trained, ppo, skip_ppo, rng, student, carry, batch)
        if mutator:
            parents = jax.tree_util.tree_map(
                lambda old, new: jnp.where(replay, new, old), student.parent_levels, levels,
            )
        else:
            parents = None
        student = student.replace(
            sampler=buffer, replay_batches=student.replay_batches + replay,
            mutation_batches=student.mutation_batches + (branch == MUTATION),
            mutation_pending=replay & (mutator is not None), parent_levels=parents,
        )
        state = state.replace(
            rng=rng, student=student, rollout_batches=state.rollout_batches + 1,
            ppo_batches=state.ppo_batches + trained,
        )
        if not record_metrics:
            return state, scores
        ppo_stats = jnp.where(trained, stats, 0)
        scoring_stats = jnp.zeros_like(stats) if config.method == "dr" else stats
        return state, (scores, (trained, ppo_stats, scoring_stats, branch))

    return step
