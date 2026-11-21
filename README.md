# PACE Research

Independent Phase 0–1 implementation of the single-level PACE score in
Yuan et al., *PACE: Parameter Change for Unsupervised Environment Design*
(ICML 2026). See the [paper-to-code mapping](docs/PAPER_TO_CODE_MAPPING.md)
and [reproduction assumptions](docs/REPRODUCTION_ASSUMPTIONS.md).

## Setup and verification

Use Python 3.11 and Git. The `maze` extra installs JAX-UED directly from
`0f8f1284677375b889e4f13a32c9617cd009f8c4`.

```powershell
python -m venv .venv
$pacePython = ".\.venv\Scripts\python.exe"
& $pacePython -m pip install -c constraints-windows-cpu-py311.txt -e '.[maze,test]'
$env:JAX_PLATFORMS = 'cpu'
& $pacePython -m pip check
& $pacePython -c "import jax, jaxued; print(jax.devices())"
& $pacePython -m pytest -q
```

On Linux or macOS, use `.venv/bin/python` and install with
`python -m pip install -e '.[maze,test]'`. Direct versions are in
[pyproject.toml](pyproject.toml). See the
[implementation log](docs/IMPLEMENTATION_LOG.md) for the Windows CPU results.

To smoke-test the adapted PLR reference with two rollout batches on CPU:

```powershell
$env:WANDB_MODE = 'disabled'
& $pacePython train/maze_pace.py --run_name adapted_plr_smoke --seed 0 --num_updates 2 --eval_freq 2 --num_train_envs 2 --num_steps 8 --epoch_ppo 1 --n_walls 10 --level_buffer_capacity 4 --replay_prob 1 --eval_num_attempts 1 --eval_levels StandardMaze --checkpoint_save_interval 0
```

## Single-level PACE API

`pace.pace_score_one_level` takes immutable parameters, a rollout already
collected on one level, its initial recurrent carry, a maximization objective,
a positive provisional step size, and a boolean parameter-leaf mask. It
returns `sum((theta_new - theta_old)**2) / alpha` after a temporary ascent step.

```python
import jax.numpy as jnp
from pace import pace_score_one_level

score = pace_score_one_level(
    jnp.array(0.0),
    2.0,
    None,
    objective_fn=lambda theta, target, _: -(theta - target) ** 2,
    provisional_alpha=0.1,
    policy_param_mask=True,
)
assert abs(float(score) - 1.6) < 1e-6
```

The Maze adapter exposes `MazeScoringRollout`,
`maze_gae_policy_gradient_objective`, and `maze_actor_shared_mask` in
[`train/maze_pace.py`](train/maze_pace.py). To use them with the scorer, bind
the network's `apply` to the objective with `functools.partial`. Pass the full
Flax variables tree, one level's unbatched recurrent carry, and time-major
observations, actions, terminal flags, and upstream GAE for that level.
The GAE objective and actor/shared mask are candidate interpretations of the
paper. Choose the provisional rate explicitly.

The reference CLI runs upstream PLR. Independent scoring of a batch of
Maze levels, PACE replay integration, and paper evaluation are later phases.
