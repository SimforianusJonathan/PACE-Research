from functools import partial

import distrax
import jax
import jax.numpy as jnp
from jaxued.environments import Maze
from jaxued.environments.maze import make_level_generator
from jaxued.level_sampler import LevelSampler
import numpy as np
import pytest

from pace import pace_score_one_level, sum_squared_tree_difference
from train.maze_pace import (
    ActorCritic,
    MazeScoringRollout,
    maze_actor_shared_mask,
    maze_gae_policy_gradient_objective,
)


def test_maze_objective_uses_fixed_advantages_carry_and_previous_done():
    obs = jnp.array([1.0, 2.0, -1.0])
    actions = jnp.array([0, 1, 0])
    dones = jnp.array([False, True, False])
    advantages = jnp.array([1.0, 3.0, -2.0])
    carry = jnp.array([4.0])

    def apply_fn(param, inputs, initial_carry):
        observations, resets = inputs
        np.testing.assert_array_equal(resets, [[False], [False], [True]])
        np.testing.assert_array_equal(initial_carry, [[4.0]])
        logits = param * (observations + jnp.where(resets, 0.0, initial_carry))
        return None, distrax.Categorical(logits=jnp.stack((logits, jnp.zeros_like(logits)), -1)), None

    rollout = MazeScoringRollout(obs, actions, dones, advantages)
    objective_fn = partial(maze_gae_policy_gradient_objective, apply_fn)
    score = pace_score_one_level(
        jnp.float32(0.0), rollout, carry,
        objective_fn=objective_fn,
        provisional_alpha=0.1,
        policy_param_mask=True,
    )
    gradient = (5.0 * 0.5 - 3.0 * 6.0 * 0.5 + (-2.0) * (-1.0) * 0.5) / 3
    assert score == pytest.approx(0.1 * gradient**2)


def test_actor_critic_single_level_scoring_uses_real_flax_tree():
    env = Maze()
    key = jax.random.PRNGKey(0)
    level = make_level_generator(13, 13, 0)(key)
    obs, _ = env.reset_to_level(key, level, env.default_params)
    obs = jax.tree_util.tree_map(lambda x: jnp.stack((x, x)), obs)
    done = jnp.array([False, False])
    rollout = MazeScoringRollout(obs, jnp.array([0, 1]), done, jnp.array([1.0, -1.0]))

    model = ActorCritic(env.action_space(env.default_params).n)
    inputs = jax.tree_util.tree_map(lambda x: x[:, None, ...], (obs, done))
    batched_carry = model.initialize_carry((1,))
    variables = model.init(key, inputs, batched_carry)
    carry = jax.tree_util.tree_map(lambda x: x[0], batched_carry)
    mask = maze_actor_shared_mask(variables)
    scopes = mask["params"]
    assert {"actor0", "actor1", "Conv_0", "scalar_embed"}.issubset(scopes)
    assert any("LSTM" in scope or "RNN" in scope for scope in scopes)
    for name, subtree in scopes.items():
        assert set(jax.tree_util.tree_leaves(subtree)) == {name not in {"critic0", "critic1"}}
    assert {"critic0", "critic1"}.issubset(scopes)

    alpha = 0.01
    objective_fn = partial(maze_gae_policy_gradient_objective, model.apply)
    score = jax.jit(partial(
        pace_score_one_level,
        objective_fn=objective_fn,
        provisional_alpha=alpha,
        policy_param_mask=mask,
    ))(variables, rollout, carry)
    gradients = jax.grad(objective_fn)(variables, rollout, carry)
    updated = jax.tree_util.tree_map(
        lambda param, grad, selected: param + alpha * grad if selected else param,
        variables, gradients, mask,
    )
    assert jnp.isfinite(score) and score > 0
    assert score == pytest.approx(sum_squared_tree_difference(updated, variables) / alpha, rel=1e-4)


def test_pinned_sampler_rank_and_staleness_distribution():
    temperature = 0.8
    sampler = LevelSampler(
        capacity=4,
        staleness_coeff=0.6,
        prioritization="rank",
        prioritization_params={"temperature": temperature},
        duplicate_check=False,
    )
    state = sampler.initialize(jnp.array(0))
    state, _ = sampler.insert_batch(state, jnp.array([1, 2, 3]), jnp.array([4.0, 9.0, 2.0]))
    ranks = 1 / np.array([2, 1, 3], dtype=float) ** (1 / temperature)
    rank_weights = np.r_[ranks / ranks.sum(), 0.0]
    staleness_weights = np.array([2 / 3, 1 / 3, 0, 0])

    actual_rank = np.asarray(sampler.score_weights(state))
    actual_mixed = np.asarray(sampler.level_weights(state))
    assert actual_rank[1] > actual_rank[0] > actual_rank[2] > actual_rank[3]
    np.testing.assert_allclose(actual_rank, rank_weights, rtol=1e-6)
    np.testing.assert_allclose(actual_mixed, 0.4 * rank_weights + 0.6 * staleness_weights, rtol=1e-6)
    assert actual_mixed.sum() == pytest.approx(1.0)
    assert np.isfinite(actual_mixed).all() and (actual_mixed >= 0).all()
