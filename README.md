# PACE Research

PACE Research studies how to prioritize training levels for replay in
reinforcement learning. It uses MiniGrid to investigate whether the response
of a policy to a provisional update provides a better replay signal than the
average magnitude of generalized advantage estimates (GAE) used in
[Prioritized Level Replay (PLR)](https://proceedings.mlr.press/v139/jiang21b.html).

The main research comparison is between PACE's squared L2 change in policy
parameters and the change in the policy's action distributions. A
policy-distribution score would evaluate the policy before and after the same
provisional update on the same observations and recurrent histories. The
comparison asks which signal leads to better performance on held-out levels
under a matched environment-interaction budget. The PACE scorer and MiniGrid
runner are implemented. The policy-distribution scorer and comparative
experiments are planned.

## PACE reproduction

This repository provides an independent recreation of PACE. The authors’
source code was unavailable when the recreation was developed, so the
implementation follows the published method and documents its interpretation
choices. [PACE paper](https://proceedings.mlr.press/v306/yuan26o.html) ·
[Paper-to-code mapping](docs/PAPER_TO_CODE_MAPPING.md) ·
[Reproduction assumptions](docs/REPRODUCTION_ASSUMPTIONS.md)

For a single level, PACE collects a trajectory and takes one temporary
gradient-ascent step with step size $\alpha$. The replay score is the squared
L2 change in the selected policy parameters, normalized by $\alpha$.

$$
S(\ell) = \frac{\lVert \theta_{\mathrm{new}} - \theta_{\mathrm{old}} \rVert_2^2}{\alpha}
$$

The scorer applies the provisional step to a separate parameter tree. The
MiniGrid implementation uses a GAE-weighted policy objective for this step and
includes actor and shared network parameters in the score. Aggregate GAE
magnitude serves a different role as a direct level-priority signal in PLR.

## Quick start

### Kaggle GPU

Import a notebook in Kaggle, enable a GPU accelerator and Internet
access, and run all cells. The notebooks clone a pinned revision of this
repository, install the GPU dependencies, and generate their MiniGrid levels
within the run.

- [Smoke test](notebooks/kaggle_minigrid.ipynb) checks GPU setup, training, checkpoint resume, and a short Maze3 evaluation.
- [MiniGrid pilot](notebooks/kaggle_minigrid_pilot.ipynb) uses the paper's rollout dimensions and exercises checkpoint resume over a short run.
- [Full PACE scoring runs](notebooks/kaggle_minigrid_pace_policy.ipynb) run L2, KL, or TVD to 30,000 PPO batches and export training and live OOD curves.

See the [Kaggle execution guide](docs/KAGGLE_EXECUTION.md) for the pinned
environment, checkpoint paths, and evaluation options.

### Local CPU

Use Python 3.11 and Git. The `maze` extra installs JAX-UED at
`0f8f1284677375b889e4f13a32c9617cd009f8c4`. Direct dependency versions are
listed in [pyproject.toml](pyproject.toml).

```powershell
python -m venv .venv
$pacePython = ".\.venv\Scripts\python.exe"
& $pacePython -m pip install -c constraints-windows-cpu-py311.txt -e '.[maze,test]'
$env:JAX_PLATFORMS = 'cpu'
& $pacePython -m pip check
& $pacePython -c "import jax, jaxued; print(jax.devices())"
& $pacePython -m pytest -q
& $pacePython -m train.pace_minigrid smoke --device cpu --budget env_steps --max-env-steps 81
```

On Linux or macOS, use `.venv/bin/python` and install with
`.venv/bin/python -m pip install -e '.[maze,test]'`. The
[implementation log](docs/IMPLEMENTATION_LOG.md) includes the smoke command
for the adapted JAX-UED PLR reference.

## MiniGrid training and evaluation

Run `python -m train.pace_minigrid --help` for the CLI. The `smoke` subcommand
runs a small test, `train` uses the paper's MiniGrid configuration with an
explicit `--provisional-alpha`, `resume` continues from a checkpoint, and
`eval` evaluates a checkpoint on held-out layouts.

The default `--protocol reuse` prefills 500 scored levels with the paper
preset and shares each replay rollout between scoring and PPO. Select
`--protocol algorithm1` for a fresh scoring rollout after replay PPO, or
`--protocol legacy` for the original empty-buffer baseline.

- `--budget ppo` counts completed PPO batches. The paper preset targets 30,000.
- `--budget rollout` counts main-loop batches.
- `--budget env_steps` caps actual training interactions, including prefill, PPO rollouts, and scoring rollouts. A shared rollout counts once.

The runner reports PPO batches, rollout batches, and environment steps
independently. The relationship between the paper's reported 245.76 million
interactions and the implemented protocols remains unresolved.
[MiniGrid training budget audit](docs/REPRODUCTION_ASSUMPTIONS.md#minigrid-training-budget-audit)

## Single-level scoring API

`pace.pace_score_one_level` takes the model parameters, a trajectory collected
on one level, the initial recurrent carry, a maximization objective, a
positive provisional step size, and a parameter-leaf mask.

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

The [Maze adapter](train/maze_pace.py) exposes `MazeScoringRollout`,
`maze_gae_policy_gradient_objective`, and `maze_actor_shared_mask`. Bind
the network's `apply` method to the objective with `functools.partial`.
Pass the full Flax variables tree, the level's unbatched initial recurrent
carry, and time-major observations, actions, terminal flags, and GAE.

## Acknowledgments

The MiniGrid integration builds on JAX-UED's `examples/maze_plr.py`.
[Third-party notices](THIRD_PARTY_NOTICES.md)
