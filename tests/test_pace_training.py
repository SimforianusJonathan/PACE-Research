import json
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from train.pace_checkpoint import read_checkpoint, restore_state, save_checkpoint
from train.pace_minigrid import MiniGridConfig, PaceLevelSampler, make_runner, score_levels
from train.pace_ood import PAPER_LEVELS, REFERENCE_PREFABS, load_suite


def assert_tree_equal(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_scoring_preserves_level_axis_and_student_snapshot():
    params = jnp.array(3.0)
    trajectories = jnp.array([[0.0, 1.0], [2.0, 5.0], [4.0, 0.0]])
    carries = jnp.array([2.0, 9.0])

    def score_one(params, rollout, carry):
        return (params - rollout.sum() - carry) ** 2

    expected = jnp.stack([score_one(params, trajectories[:, i], carries[i]) for i in range(2)])
    actual = score_levels(params, trajectories, carries, score_one)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(score_levels(params, trajectories[:, ::-1], carries[::-1], score_one), expected[::-1])


def test_full_buffer_evicts_lowest_raw_score():
    sampler = PaceLevelSampler(
        capacity=2, staleness_coeff=1.0, prioritization="rank",
    )
    buffer = sampler.initialize(jnp.array(0))
    buffer, _ = sampler.insert_batch(buffer, jnp.array([1, 2]), jnp.array([9.0, 1.0]))
    # Make the high-score entry newer so mixed-weight eviction would choose it.
    buffer = {**buffer, "timestamps": jnp.array([2, 1])}
    assert int(sampler.level_weights(buffer).argmin()) == 0
    buffer, idx = sampler.insert(buffer, jnp.array(3), jnp.float32(4.0))
    assert int(idx) == 1
    np.testing.assert_array_equal(buffer["scores"], [9.0, 4.0])


def test_checkpoint_preserves_training_and_replay_state(tmp_path):
    config = MiniGridConfig.smoke()
    initial, advance = make_runner(config)
    explored, scores = advance(initial, 1)
    assert int(explored.ppo_batches) == int(explored.student.step) == 0
    assert int(explored.student.sampler["size"]) == 2
    assert np.isfinite(scores).all() and float(scores.max()) > 0
    assert_tree_equal(explored.student.params, initial.student.params)
    assert_tree_equal(explored.student.opt_state, initial.student.opt_state)
    boundary, _ = advance(explored, 1)
    assert int(boundary.ppo_batches) == int(boundary.student.step) == 1
    assert not np.array_equal(boundary.rng, explored.rng)

    path = tmp_path / "checkpoint.msgpack"
    save_checkpoint(path, asdict(config), boundary)
    payload = read_checkpoint(path)
    assert MiniGridConfig(**payload["config"]) == config
    restored = restore_state(payload, initial)
    assert_tree_equal(restored, boundary)
    expected, expected_scores = advance(boundary, 2)
    actual, scores = advance(restored, 2)
    assert_tree_equal(actual, expected)
    np.testing.assert_array_equal(scores, expected_scores)
    assert (int(actual.rollout_batches), int(actual.ppo_batches), int(actual.student.step)) == (4, 3, 3)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, asdict(config), actual)


def test_paper_ood_suite_requires_the_missing_layouts(tmp_path):
    with pytest.raises(ValueError, match="SimpleCrossing"):
        load_suite("paper")
    layouts = tmp_path / "layouts.json"
    layouts.write_text(json.dumps({
        name: ["###", "#>G", "###"] for name in PAPER_LEVELS if name not in REFERENCE_PREFABS
    }), encoding="utf-8")
    names, levels = load_suite("paper", layouts)
    assert names == PAPER_LEVELS
    assert levels.wall_map.shape == (12, 13, 13)


def test_paper_protocol_matches_table_9():
    config = MiniGridConfig(provisional_alpha=0.01)
    assert config.budget == "ppo"
    assert (config.maze_size, config.n_walls, config.num_updates, config.num_steps, config.num_envs) == (
        15, 100, 30_000, 256, 32,
    )
    assert (config.learning_rate, config.max_grad_norm, config.num_minibatches, config.ppo_epochs) == (
        1e-4, 0.5, 1, 5,
    )
    assert (config.gamma, config.clip_eps, config.gae_lambda, config.entropy_coeff, config.critic_coeff) == (
        0.995, 0.2, 0.98, 0.001, 0.5,
    )
    assert (config.capacity, config.replay_prob, config.staleness_coeff, config.temperature) == (1000, 0.7, 0.6, 0.8)
    assert (config.minimum_fill_ratio, config.prioritization, config.exploratory_grad_updates) == (0.5, "rank", False)
