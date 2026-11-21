from functools import partial

from flax.core import freeze
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pace import pace_score_one_level, sum_squared_tree_difference


def quadratic_objective(param, target, initial_carry):
    return -(param - target - initial_carry) ** 2


def score_quadratic(param, target, carry, alpha=0.1):
    return pace_score_one_level(
        param,
        target,
        carry,
        objective_fn=quadratic_objective,
        provisional_alpha=alpha,
        policy_param_mask=True,
    )


def test_scalar_score_matches_paper_equations():
    # J(theta) = -(theta - 2)^2 at theta = 0: g = 4, theta_new = 0.4.
    score = score_quadratic(jnp.array(0.0), 2.0, 0.0)
    assert score.shape == ()
    assert jnp.isfinite(score)
    assert score == pytest.approx(1.6)


def test_zero_gradient_gives_exact_zero():
    assert score_quadratic(jnp.array(2.0), 2.0, 0.0) == 0.0


@pytest.mark.parametrize("alpha", [0.0, -0.1, float("nan"), float("inf"), -float("inf")])
def test_must_supply_finite_positive_alpha(alpha):
    with pytest.raises(ValueError, match="finite and positive"):
        score_quadratic(jnp.array(0.0), 2.0, 0.0, alpha)


@pytest.mark.parametrize("mask", [True, {"policy": False}, {"policy": 1}])
def test_mask_must_match_and_select_parameters(mask):
    with pytest.raises(ValueError, match="policy_param_mask"):
        pace_score_one_level(
            {"policy": jnp.array(0.0)},
            None,
            None,
            objective_fn=lambda params, *_: params["policy"],
            provisional_alpha=0.1,
            policy_param_mask=mask,
        )


def test_tree_difference_widens_before_subtraction_and_squaring():
    old = {"matrix": jnp.array([[0.0, 1.0], [2.0, 3.0]]), "pair": (jnp.float16(-40_000),)}
    new = {"matrix": jnp.array([[1.0, 3.0], [5.0, 7.0]]), "pair": (jnp.float16(40_000),)}
    assert jnp.isfinite(sum_squared_tree_difference(new, old))
    assert sum_squared_tree_difference(new, old) == pytest.approx(30 + 80_000**2)
    assert sum_squared_tree_difference(new, new) == 0.0


def test_tree_difference_rejects_broadcasting():
    with pytest.raises(ValueError, match="identical shapes"):
        sum_squared_tree_difference({"actor": jnp.ones(2)}, {"actor": jnp.zeros(1)})


def test_score_measures_rounded_parameter_delta():
    score = pace_score_one_level(
        jnp.float16(1024),
        None,
        None,
        objective_fn=lambda param, *_: param,
        provisional_alpha=0.125,
        policy_param_mask=True,
    )
    assert score == 0.0  # The step rounds to the same float16 parameter.


def test_ascent_direction_matches_paper():
    # Just below 1.0, float32 spacing is 2**-24. Just above it, spacing doubles.
    # The ascent step rounds to 1.0, while a descent step would stay distinct.
    score = pace_score_one_level(
        jnp.float32(1),
        None,
        None,
        objective_fn=lambda param, *_: param,
        provisional_alpha=2.0**-24,
        policy_param_mask=True,
    )
    assert score == 0.0


def test_jit_agrees_with_eager_and_uses_rollout_and_carry():
    compiled = jax.jit(score_quadratic)
    old = jnp.array(0.0)
    assert compiled(old, 2.0, 0.0) == pytest.approx(score_quadratic(old, 2.0, 0.0))
    assert compiled(old, 0.0, 1.0) == pytest.approx(0.4)
    assert compiled(old, 3.0, 1.0) == pytest.approx(6.4)


def test_only_selected_leaves_contribute():
    params = freeze({"actor": jnp.zeros(2), "shared": jnp.array(0.0), "critic": jnp.array(0.0)})
    mask = freeze({"actor": True, "shared": True, "critic": False})

    def objective(p, *_):
        return (p["actor"] * jnp.array([2.0, 3.0])).sum() + 4 * p["shared"] + 99 * p["critic"]

    score = pace_score_one_level(
        params, None, None,
        objective_fn=objective,
        provisional_alpha=0.1,
        policy_param_mask=mask,
    )
    assert score == pytest.approx(0.1 * (2**2 + 3**2 + 4**2))


def test_scoring_preserves_real_train_state_and_next_optimizer_update():
    state = TrainState.create(
        apply_fn=lambda p, x: p["w"] * x,
        params={"w": jnp.array(1.0)},
        tx=optax.adam(1e-3),
    ).apply_gradients(grads={"w": jnp.array(0.5)})
    before = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), state)

    def real_loss(params):
        return (state.apply_fn(params, 2.0) - 7.0) ** 2

    real_grad = jax.grad(real_loss)
    expected = state.apply_gradients(grads=real_grad(state.params))
    score_fn = partial(
        pace_score_one_level,
        objective_fn=lambda p, target, _: -(p["w"] - target) ** 2,
        provisional_alpha=0.1,
        policy_param_mask={"w": True},
    )
    assert jax.jit(score_fn)(state.params, 3.0, None) > 0
    for actual, snapshot in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(before)):
        np.testing.assert_array_equal(actual, snapshot)

    actual = state.apply_gradients(grads=real_grad(state.params))
    for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
