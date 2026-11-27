"""Single-level PACE scoring from a gradient-aligned provisional update."""

import math
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp


def sum_squared_tree_difference(params_new: Any, params_old: Any) -> jax.Array:
    """Return the squared Euclidean distance over matching real-valued PyTrees.

    Widen half-precision leaves before subtraction and reduction. Existing
    float64 precision is preserved when it is enabled in JAX.
    """
    def leaf_distance(new, old):
        if jnp.shape(new) != jnp.shape(old):
            raise ValueError("corresponding parameter leaves must have identical shapes")
        dtype = jnp.result_type(new, old, jnp.float32)
        delta = jnp.asarray(new, dtype) - jnp.asarray(old, dtype)
        return jnp.sum(jnp.square(delta))

    distances = jax.tree_util.tree_map(leaf_distance, params_new, params_old)
    return sum(jax.tree_util.tree_leaves(distances), jnp.zeros((), jnp.float32))


def _provisional_update(
    params_old: Any,
    rollout: Any,
    initial_rnn_state: Any,
    *,
    objective_fn: Callable[[Any, Any, Any], jax.Array],
    provisional_alpha: float,
    policy_param_mask: Any,
) -> tuple[Any, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    alpha = float(provisional_alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("provisional_alpha must be finite and positive")

    old_leaves, structure = jax.tree_util.tree_flatten(params_old)
    mask_leaves, mask_structure = jax.tree_util.tree_flatten(policy_param_mask)
    if mask_structure != structure:
        raise ValueError("policy_param_mask must match the parameter PyTree")
    if any(not isinstance(selected, bool) for selected in mask_leaves) or not any(mask_leaves):
        raise ValueError("policy_param_mask must have Python bool leaves and select parameters")

    selected_old = tuple(leaf for leaf, selected in zip(old_leaves, mask_leaves) if selected)

    def with_selected_leaves(selected_params):
        leaves_to_update = iter(selected_params)
        leaves = [
            next(leaves_to_update) if selected else old
            for old, selected in zip(old_leaves, mask_leaves)
        ]
        return jax.tree_util.tree_unflatten(structure, leaves)

    def selected_objective(selected_params):
        return objective_fn(with_selected_leaves(selected_params), rollout, initial_rnn_state)

    gradients = jax.grad(selected_objective)(selected_old)
    selected_new = tuple(
        old + alpha * gradient for old, gradient in zip(selected_old, gradients)
    )
    return with_selected_leaves(selected_new), selected_old, selected_new


def pace_score_one_level(
    params_old: Any,
    rollout: Any,
    initial_rnn_state: Any,
    *,
    objective_fn: Callable[[Any, Any, Any], jax.Array],
    provisional_alpha: float,
    policy_param_mask: Any,
) -> jax.Array:
    """Measure one temporary ascent step on a pre-collected level rollout.

    The pure objective returns a scalar to maximize from
    `(params, rollout, initial_rnn_state)`. Supply a mask with the same PyTree
    structure as the real floating parameters and Python bool leaves. It
    defines both the differentiated parameters and the norm's scope. Use
    finite inputs and an alpha and score representable in the working dtype.
    Capture the objective, alpha, and mask in a closure when using JIT.
    """
    _, selected_old, selected_new = _provisional_update(
        params_old, rollout, initial_rnn_state, objective_fn=objective_fn,
        provisional_alpha=provisional_alpha, policy_param_mask=policy_param_mask,
    )
    return sum_squared_tree_difference(selected_new, selected_old) / float(provisional_alpha)


def mean_action_distribution_score(
    logits_old: jax.Array,
    logits_new: jax.Array,
    *,
    metric: Literal["kl", "tvd"],
) -> jax.Array:
    """Average forward KL or TVD over a nonempty (time, action) array.

    TVD is the squared W2 cost with zero diagonal and unit off-diagonal
    ground distances. KL is measured in nats. Use -inf to mask actions,
    with at least one finite logit per timestep in each policy.
    """
    dtype = jnp.result_type(logits_old, logits_new, jnp.float32)
    logits_old = jnp.asarray(logits_old, dtype)
    logits_new = jnp.asarray(logits_new, dtype)
    if logits_old.shape != logits_new.shape or logits_old.ndim != 2 or logits_old.size == 0:
        raise ValueError("policy logits must share a nonempty (time, action) shape")
    if metric not in ("kl", "tvd"):
        raise ValueError("metric must be 'kl' or 'tvd'")

    log_p = jax.nn.log_softmax(logits_old, axis=-1)
    log_q = jax.nn.log_softmax(logits_new, axis=-1)
    p, q = jnp.exp(log_p), jnp.exp(log_q)
    if metric == "tvd":
        return (0.5 * jnp.abs(p - q).sum(axis=-1)).mean()

    # Sum p * log(p / q) + q - p. The added terms sum to zero, and
    # expm1 plus a Taylor expansion avoid cancellation for nearby policies.
    p_zero, q_zero = jnp.isneginf(log_p), jnp.isneginf(log_q)
    delta = jnp.where(p_zero | q_zero, 0.0, log_q - log_p)
    positive, negative = jnp.maximum(delta, 0.0), jnp.minimum(delta, 0.0)
    rising = q * (-jnp.expm1(-positive) - positive * jnp.exp(-positive))
    falling = p * (jnp.expm1(negative) - negative)
    small = jnp.clip(delta, -0.01, 0.01)
    nearby = p * small**2 * (0.5 + small * (1/6 + small * (1/24 + small/120)))
    costs = jnp.where(jnp.abs(delta) <= 0.01, nearby, jnp.where(delta > 0, rising, falling))
    costs = jnp.where(p_zero, q, jnp.where(q_zero, jnp.inf, costs))
    return costs.sum(axis=-1).mean()


def pace_policy_score_one_level(
    params_old: Any,
    rollout: Any,
    initial_rnn_state: Any,
    *,
    objective_fn: Callable[[Any, Any, Any], jax.Array],
    policy_logits_fn: Callable[[Any, Any, Any], jax.Array],
    provisional_alpha: float,
    policy_param_mask: Any,
    metric: Literal["kl", "tvd"],
) -> jax.Array:
    """Score the action-distribution change after the PACE provisional step.

    The callback `policy_logits_fn(params, rollout, initial_rnn_state)` must
    replay the history and return all action logits in (time, action) order.
    Both policies start from the supplied carry and compute their own later
    carries. Close over the callbacks, alpha, mask and metric for JIT.
    """
    params_new, _, _ = _provisional_update(
        params_old, rollout, initial_rnn_state, objective_fn=objective_fn,
        provisional_alpha=provisional_alpha, policy_param_mask=policy_param_mask,
    )
    logits_old = policy_logits_fn(params_old, rollout, initial_rnn_state)
    logits_new = policy_logits_fn(params_new, rollout, initial_rnn_state)
    return mean_action_distribution_score(logits_old, logits_new, metric=metric)
