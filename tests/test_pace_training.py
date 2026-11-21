import json
from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

import train.pace_minigrid as minigrid
from train.pace_checkpoint import read_checkpoint, restore_state, save_checkpoint
from train.pace_minigrid import MiniGridConfig, PaceLevelSampler, interaction_counts, make_runner, score_levels
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
    assert int(initial.ppo_batches) == int(initial.student.step) == int(initial.rollout_batches) == 0
    assert int(initial.student.sampler["size"]) == 2
    assert int(initial.prefill_env_steps) == 16
    boundary, scores = advance(initial, 2)
    assert np.isfinite(scores).all() and float(scores.max()) > 0
    assert int(boundary.ppo_batches) == int(boundary.student.step) == 2
    assert not np.array_equal(boundary.rng, initial.rng)

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
    assert (int(actual.rollout_batches), int(actual.ppo_batches), int(actual.student.step)) == (4, 4, 4)
    assert interaction_counts(config, actual.rollout_batches, actual.ppo_batches, actual.prefill_env_steps) == {
        "env_steps": 144, "prefill_env_steps": 16, "ppo_env_steps": 64,
        "scoring_env_steps": 64, "shared_env_steps": 0, "main_collection_batches": 8,
    }
    with pytest.raises(FileExistsError):
        save_checkpoint(path, asdict(config), actual)


@pytest.fixture
def mock_training(monkeypatch):
    def collect(rng, env, env_params, student, carry, obs, env_state, num_envs, num_steps):
        shape = (num_steps, num_envs)
        observations = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (num_steps, *x.shape)), obs)
        zero = jnp.zeros(shape)
        # Mark rewards with the optimizer step of the policy that collected them.
        rewards = jnp.full(shape, student.step + 1, dtype=jnp.float32)
        trajectory = observations, zero.astype(jnp.int32), rewards, jnp.ones(shape, bool), zero, zero, {}
        return (rng, student, carry, obs, env_state, jnp.zeros(num_envs)), trajectory

    def update(rng, student, *_):
        params = jax.tree_util.tree_map(lambda x: x + 1, student.params)
        return (rng, student.replace(step=student.step + 1, params=params)), None

    def score(params, rollout, carry, **_):
        bias = params["params"]["actor1"]["bias"][0]
        return jnp.abs(30 - rollout.advantages[0] - 10 * bias)

    monkeypatch.setattr(minigrid, "sample_trajectories_rnn", collect)
    monkeypatch.setattr(minigrid, "update_actor_critic_rnn", update)
    monkeypatch.setattr(minigrid, "pace_score_one_level", score)


@pytest.mark.parametrize("replay_prob, expected_score, expected_ppo", [(0.0, 29, 0), (1.0, 18, 1)])
def test_prefill_and_scoring_use_the_expected_policy(mock_training, replay_prob, expected_score, expected_ppo):
    config = replace(MiniGridConfig.smoke(), capacity=6, replay_prob=replay_prob)
    template, _ = make_runner(config, prefill=False)
    initial, advance = make_runner(config)
    # Three levels split into batches of two and one, with no real update.
    assert int(initial.student.sampler["size"]) == 3
    assert int(initial.prefill_env_steps) == 3 * config.num_steps
    assert int(initial.student.step) == int(initial.rollout_batches) == 0
    assert_tree_equal(initial.student.params, template.student.params)
    assert_tree_equal(initial.student.opt_state, template.student.opt_state)

    updated, scores = advance(initial, 1)
    np.testing.assert_array_equal(scores, [[expected_score, expected_score]])
    assert int(updated.ppo_batches) == int(updated.student.step) == expected_ppo
    if not expected_ppo:
        assert_tree_equal(updated.student.params, initial.student.params)
    else:
        # PLR refresh keeps the same levels and accepts a lower score on replay.
        occupied_scores = updated.student.sampler["scores"][:3]
        assert int(updated.student.sampler["size"]) == 3
        assert float(occupied_scores.min()) == expected_score
        assert float(occupied_scores.max()) == 29


def test_legacy_checkpoint_preserves_old_protocol(tmp_path):
    config = replace(MiniGridConfig.smoke(), protocol="legacy")
    initial, _ = make_runner(config, prefill=False)
    config_v1 = asdict(config)
    del config_v1["protocol"]
    state_v1 = serialization.to_state_dict(initial)
    del state_v1["prefill_env_steps"]
    path = tmp_path / "legacy.msgpack"
    path.write_bytes(serialization.msgpack_serialize({
        "version": 1, "config": config_v1, "runner": jax.device_get(state_v1),
    }))

    payload = read_checkpoint(path)
    assert payload["config"]["protocol"] == "legacy"
    assert_tree_equal(restore_state(payload, initial), initial)
    assert interaction_counts(config, 4, 3, 0)["env_steps"] == 64


@pytest.mark.parametrize(
    "protocol, replay_prob, cap, expected",
    [
        ("algorithm1", 1.0, 47, (0, 0, 16, 0)),
        ("algorithm1", 1.0, 49, (1, 1, 48, 2)),
        ("algorithm1", 0.0, 33, (1, 0, 32, 1)),
        ("legacy", 1.0, 49, (3, 2, 48, 3)),
    ],
)
def test_env_budget_counts_all_phases_and_keeps_pending_branch(mock_training, protocol, replay_prob, cap, expected):
    config = replace(
        MiniGridConfig.smoke(), budget="env_steps", max_env_steps=cap,
        protocol=protocol, replay_prob=replay_prob,
    )
    state, advance = make_runner(config)
    for _ in range(10):
        candidate, scores = advance(state, 10)
        if not scores.size:
            assert_tree_equal(candidate, state)
            break
        state = candidate
    else:
        pytest.fail("The interaction budget never stopped the runner")
    counts = interaction_counts(config, state.rollout_batches, state.ppo_batches, state.prefill_env_steps)
    assert (
        int(state.rollout_batches), int(state.ppo_batches),
        counts["env_steps"], counts["main_collection_batches"],
    ) == expected


def test_env_budget_counts_duplicate_prefill_attempts(mock_training, monkeypatch):
    original = minigrid.make_level_generator

    def duplicate_level(*args):
        generate = original(*args)
        return lambda _: generate(jax.random.PRNGKey(0))

    monkeypatch.setattr(minigrid, "make_level_generator", duplicate_level)
    config = replace(MiniGridConfig.smoke(), budget="env_steps", max_env_steps=24)
    with pytest.raises(ValueError, match=r"used 24 environment steps within cap 24 and filled 1/2"):
        make_runner(config)

    with pytest.raises(ValueError, match=r"Prefill needs at least 16"):
        make_runner(replace(config, max_env_steps=15))


def test_env_step_cap_and_counters_survive_resume(mock_training, tmp_path, monkeypatch, capsys):
    # After one replay, 48 are spent and the next replay needs 32.
    args = [
        "train.pace_minigrid", "smoke", "--budget", "env_steps",
        "--max-env-steps", "79", "--stop-after", "1",
    ]
    monkeypatch.setattr("sys.argv", args)
    _, args = minigrid._cli()
    minigrid._run_training(args, tmp_path)
    checkpoint = tmp_path / "checkpoints" / "step_00000001.msgpack"
    payload = read_checkpoint(checkpoint)
    assert payload["config"]["max_env_steps"] == 79
    assert budget_log(capsys)["stop_reason"] == "stop_after"

    monkeypatch.setattr("sys.argv", ["train.pace_minigrid", "resume", "--checkpoint", str(checkpoint)])
    _, args = minigrid._cli()
    minigrid._run_training(args, tmp_path)
    summary = budget_log(capsys)
    assert (summary["rollout_batches"], summary["ppo_batches"], summary["env_steps"]) == (1, 1, 48)
    assert summary["budget_remaining"] == 31 and summary["stop_reason"] == "env_steps_budget"


def budget_log(capsys):
    return json.loads(capsys.readouterr().out.splitlines()[-1])


def test_ppo_budget_counts_real_updates(mock_training, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "train.pace_minigrid", "smoke", "--budget", "ppo", "--protocol", "legacy",
    ])
    _, args = minigrid._cli()
    minigrid._run_training(args, tmp_path)
    summary = budget_log(capsys)
    assert (summary["rollout_batches"], summary["ppo_batches"], summary["env_steps"]) == (5, 4, 80)
    assert summary["stop_reason"] == "ppo_budget"


def test_train_env_budget_defaults_to_reported_total(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "train.pace_minigrid", "train", "--provisional-alpha", "0.01", "--budget", "env_steps",
    ])
    _, args = minigrid._cli()
    assert args.max_env_steps == 245_760_000


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
    assert config.protocol == "algorithm1" and config.initial_buffer_size == 500
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
