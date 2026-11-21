"""The single-level, gradient-aligned provisional update in PACE Eqs. 8 and 11."""

import math
from collections.abc import Callable
from typing import Any

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

    def selected_objective(selected_params):
        leaves_to_update = iter(selected_params)
        leaves = [
            next(leaves_to_update) if selected else old
            for old, selected in zip(old_leaves, mask_leaves)
        ]
        params = jax.tree_util.tree_unflatten(structure, leaves)
        return objective_fn(params, rollout, initial_rnn_state)

    gradients = jax.grad(selected_objective)(selected_old)
    selected_new = tuple(
        old + alpha * gradient for old, gradient in zip(selected_old, gradients)
    )
    return sum_squared_tree_difference(selected_new, selected_old) / alpha
