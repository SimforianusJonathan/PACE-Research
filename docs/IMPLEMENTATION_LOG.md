# Phase 0–1 implementation log

## Sources and setup

The supplied 22-page PDF was extracted with `pypdf`. Figures 2, 11,
Algorithm 1, Eq. 11, and Table 9 were also inspected from PNGs rendered
with `pdftoppm`. The upstream audit used a local checkout, including
`examples/maze_plr.py`, `src/jaxued/level_sampler.py`, `src/jaxued/linen.py`,
the Maze generator and prefabs, and `pyproject.toml`.

Checkout and environment commands from the workspace and project roots:

```powershell
git clone --depth 1 https://github.com/DramaCow/jaxued.git .pace-audit/jaxued
git -C .pace-audit/jaxued rev-parse HEAD
Set-Location PACE-Research
git init -b main
python -m venv .venv
$pacePython = ".\.venv\Scripts\python.exe"
& $pacePython -m pip install --disable-pip-version-check -e '.[maze,test]'
```

The restricted shell could not resolve GitHub during the first clone
attempt. The clone succeeded with approved network access. The first
editable install started before the `src` directory existed and failed at
the editable build step. It was retried with the source package in place.
The checkout and the installed package use revision
`0f8f1284677375b889e4f13a32c9617cd009f8c4`.

The CPU environment was Windows x86-64 (build 26300), Intel Core i7-1165G7,
CPython 3.11.15, JAX/jaxlib 0.4.35, Flax 0.10.2, Optax 0.2.4, and JAX-UED
0.0.2. JAX returned `[CpuDevice(id=0)]`. No CUDA device was used.
`python -m pip freeze --exclude pace-research --exclude jaxued` supplied
the versions in [the Windows CPU constraints](../constraints-windows-cpu-py311.txt).

## Verification commands

Run from the project root. The two smoke runs use seed 0 and two outer
batches with two environments and eight steps each, for a nominal 32
in-loop Maze transitions per run. The replay probability is set to 1 for
this smoke configuration so that the loop can exercise replay as soon as
the buffer reaches half occupancy. The evaluation prefab is 13×13 to match
the reference renderer.

```powershell
$env:JAX_PLATFORMS = 'cpu'
$env:WANDB_MODE = 'disabled'
& $pacePython -m pip check
& $pacePython -c "import jax, jaxued; print(jax.devices())"
& $pacePython -m pytest -q
& $pacePython ../.pace-audit/jaxued/examples/maze_plr.py --run_name upstream_plr_smoke --seed 0 --num_updates 2 --eval_freq 2 --num_train_envs 2 --num_steps 8 --epoch_ppo 1 --n_walls 10 --level_buffer_capacity 4 --replay_prob 1 --eval_num_attempts 1 --eval_levels StandardMaze --checkpoint_save_interval 0
& $pacePython train/maze_pace.py --run_name adapted_plr_smoke --seed 0 --num_updates 2 --eval_freq 2 --num_train_envs 2 --num_steps 8 --epoch_ppo 1 --n_walls 10 --level_buffer_capacity 4 --replay_prob 1 --eval_num_attempts 1 --eval_levels StandardMaze --checkpoint_save_interval 0
```

The source also passed `python -m compileall -q src train tests`.

## Results

| Check | Exit status | Observed output |
| --- | ---: | --- |
| Dependency installation | 0 | Built and installed `pace-research` and the pinned `jaxued` wheel. |
| Dependency consistency | 0 | `No broken requirements found.` |
| Import, backend, and JAX-UED revision | 0 | `[CpuDevice(id=0)]`. Installed VCS metadata matched the checkout SHA. |
| `pytest -q` | 0 | `20 passed in 67.35s (0:01:07)`. |
| Unmodified upstream PLR smoke | 0 | `Logging update: 2`. Process wall time 73.54 s. |
| Adapted PLR smoke | 0 | `Logging update: 2`. Process wall time 56.21 s. |

Smoke timings include imports, JIT compilation, the training loop, and one
evaluation pass. The smoke runs used the legacy PLR score. Single-level PACE
was verified separately through its analytic, numerical, state-isolation,
and Maze-network tests.

Local raw artifacts in `outputs/phase01/` are ignored by Git:

- `setup-first-attempt.log` and `setup.log` record dependency installation.
- `environment.json` records versions, backend, and installed VCS origin.
- `pytest.log` records the unit suite output.
- `upstream_plr_smoke.log`, `adapted_plr_smoke.log`, and `smoke-runs.json`
  record the smoke outputs, arguments, timings, and exit codes.

The reproduction assumptions and later-phase verification targets are
tracked in [reproduction assumptions](REPRODUCTION_ASSUMPTIONS.md) and the
[paper-to-code mapping](PAPER_TO_CODE_MAPPING.md).
