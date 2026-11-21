# MiniGrid PACE on Kaggle

## Set up the notebook

1. Import [`notebooks/kaggle_minigrid.ipynb`](../notebooks/kaggle_minigrid.ipynb)
   into Kaggle. In notebook settings, enable **Internet** and select a GPU
   accelerator. A two-GPU Kaggle allocation also works with one device exposed
   through `CUDA_VISIBLE_DEVICES=0`.
2. The first cell pins `REPO_REF` to the trainer revision for these
   notebooks. If you are running from a local snapshot, archive that
   commit with `git archive --format=zip --output=PACE-Research.zip HEAD`,
   upload the ZIP to a Kaggle Dataset, attach it, and set `SOURCE` to the
   ZIP under `/kaggle/input/`. You can also specify a directory containing
   the extracted source. Clone checkouts use a workspace directory named
   for the revision. The default clone workflow needs no dataset.
3. Run all cells. The notebook installs the environment, verifies one CUDA
   device with a JIT-compiled matrix multiplication, and starts a four-batch
   PACE smoke run with a 145-step interaction cap. A fresh process resumes
   from batch two and verifies 144 interactions with one step left. A final cell
   evaluates the restored model on two short Maze3 episodes.
4. Use **Save Version** to retain outputs or download the checkpoint files
   before the session ends. In a later notebook, attach the saved outputs
   and pass the checkpoint path to `resume`.

The bootstrap script installs a separate CPython 3.11.15 environment for
Linux x86-64 under `/tmp/pace-kaggle`. It uses the
[CUDA lock](../requirements/kaggle-linux-cuda12-py311.txt) for JAX/jaxlib
0.4.35, JAX CUDA plugin/PJRT 0.4.35, CUDA 12.6 wheel libraries, Flax
0.10.2, Optax 0.2.4, and JAX-UED commit
`0f8f1284677375b889e4f13a32c9617cd009f8c4`. It leaves the notebook
kernel's packages intact. Kaggle supplies the NVIDIA driver. The JAX
0.4.35 wheel's CUDA extra pins jaxlib 0.4.34 due to a release metadata
issue, so this environment selects the CUDA plugin directly and pins
jaxlib to the matching 0.4.35 release. [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) · [JAX 0.4.35 wheel metadata discussion](https://github.com/jax-ml/jax/issues/24826) · [Kaggle runtime images](https://github.com/Kaggle/docker-python)

## Command-line execution

Run these from the repository root inside Kaggle. Use the same interpreter
for setup, training, resuming, and evaluation. The CLI selects one local JAX
device and runs without device replication.

```bash
bash scripts/setup_kaggle.sh
PACE_PYTHON=/tmp/pace-kaggle/venv/bin/python
nvidia-smi -L || true
VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES%%,*}" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
if [[ -r /usr/local/nvidia/lib64/libcuda.so.1 ]]; then
  export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
fi
```

If an imported notebook fails at `nvidia-smi` with
`libnvidia-ml.so` missing or exit code 12, use the updated notebook.
It preserves the original runtime path for the NVML diagnostic and keeps
the NVIDIA driver path for the CUDA and JAX checks. If the CUDA driver
check still fails, select a GPU accelerator and start a fresh session or
save a new GPU-enabled version. [Kaggle GPU runtime definition](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl) · [NVIDIA NVML reference](https://docs.nvidia.com/deploy/nvml-api/latest/nvml-api-reference.html)

The notebook uses `MPLBACKEND=Agg` for its subprocesses so the isolated
environment can import Gymnax and Matplotlib. Setup and GPU probe logs go
under `/kaggle/working/pace/`. Each run saves its training, resume, and
verification output in the run directory.

### Smoke and resume

```bash
"$PACE_PYTHON" -m train.pace_minigrid smoke --device gpu \
  --protocol algorithm1 --budget env_steps --max-env-steps 145 \
  --output-dir /kaggle/working/pace/smoke --checkpoint-interval 2 --stop-after 2

"$PACE_PYTHON" -m train.pace_minigrid resume --device gpu \
  --checkpoint /kaggle/working/pace/smoke/checkpoints/step_00000002.msgpack \
  --output-dir /kaggle/working/pace/smoke --checkpoint-interval 2
```

The smoke preset uses two environments, eight steps per rollout, one PPO
epoch, and a four-level buffer. It uses replay probability 1 after half fill
and a toy provisional alpha of 0.01. It keeps the 15×15 Maze and 100 wall
placement operations. Prefill scores two levels. The four main-loop batches
all replay, for 144 transitions total: 16 prefill, 64 PPO, and 64 scoring.
The 145-step cap leaves one step unused.

### MiniGrid pilot

Import [`notebooks/kaggle_minigrid_pilot.ipynb`](../notebooks/kaggle_minigrid_pilot.ipynb)
into Kaggle, select a GPU accelerator, enable Internet, and run all cells.
The notebook clones the configured trainer revision and generates its own
levels. It first scores 500 distinct levels, then runs 24 main-loop batches,
with checkpoints at batches 8, 16, and 24. Scoring in the main loop accounts
for 196,608 interactions. Prefill and replay PPO add separate interactions
reported by the notebook. A fresh process resumes from batch 16
and verifies that the final checkpoint records replay PPO and the expected
optimizer step count. The pilot uses seed 0 and provisional alpha 0.01 as
an explicit test choice.

### Configure a full training run

Supply and record the provisional alpha you intend to use. The command
below is the launch form for an approved full run. It is separate from the
notebook smoke workflow.

```bash
# Set PACE_ALPHA first to your documented provisional step size.
"$PACE_PYTHON" -m train.pace_minigrid train --device gpu \
  --seed 0 --provisional-alpha "${PACE_ALPHA:?Set a documented provisional alpha}" \
  --output-dir /kaggle/working/pace/paper-seed0 --checkpoint-interval 250
```

The paper preset uses 15×15 maps with 100 wall draws, 32 environments ×
256 steps, one minibatch, five PPO epochs, PPO LR `1e-4`, gradient clip
`0.5`, gamma `0.995`, GAE lambda `0.98`, PPO clip `0.2`, entropy
coefficient `0.001`, and value-loss coefficient `0.5`. It uses a
1,000-level buffer, replay probability `0.7`, rank temperature `0.8`,
staleness weight `0.6`, half-fill replay gating, and no real PPO on new
levels. Train separate seeds in separate output directories.

The default `--budget ppo` counts **30,000 real replay PPO batches** as
stated in Table 9. Use `train --budget rollout` to stop at 30,000 main-loop
batches. For an interaction-matched run, add `--budget env_steps` to the
training command. It caps actual training interactions at 245,760,000 by
default. Supply `--max-env-steps N` for a different cap.

The environment cap includes prefill, PPO, and scoring transitions.
The run stops before a complete next iteration would overshoot, then
reports `budget_remaining` and `stop_reason`. Legacy replay trajectories
used by both PPO and scoring are counted once. Logs separately report
`ppo_batches`, `rollout_batches` (main-loop batches),
`main_collection_batches`, and actual `env_steps`. See the
[training budget audit](REPRODUCTION_ASSUMPTIONS.md#minigrid-training-budget-audit)
for the paper's unresolved interaction accounting.

`--checkpoint-interval` counts rollout batches and accepts 0 to save at
the final or requested stopping boundary only. `--stop-after N` pauses at
absolute rollout batch `N` without changing the stored optimizer schedule.
`--output-dir` selects the run directory. In Kaggle, new runs default to
directories under `/kaggle/working/pace`. To resume from a read-only input,
set `--checkpoint /kaggle/input/.../step_XXXXXXXX.msgpack` and use a writable
`--output-dir` in `/kaggle/working`. When resuming in place, use the run's
latest checkpoint. Use one writer per output directory.

Each checkpoint includes parameters, the complete Optax state and optimizer
step, runner PRNG, replay levels and their scores, replay timestamps and
occupancy, prefill interactions, rollout and PPO counters, and the training config. Files are
published atomically. The runner resets its environments and recurrent carry
at the beginning of each batch, so a boundary checkpoint is sufficient for
continuation. For a reproducibility comparison use the same source, package
lock, and device type. Older checkpoints resume under their original
`legacy` protocol. Select `train --protocol legacy` for a new comparison
with the earlier shared-rollout flow.

### OOD evaluation

```bash
"$PACE_PYTHON" -m train.pace_minigrid eval --device gpu \
  --checkpoint /kaggle/working/pace/smoke/checkpoints/step_00000004.msgpack \
  --suite reference --levels Maze3 --episodes 2 --horizon 16 \
  --output-dir /kaggle/working/pace/smoke/eval
```

For an experimental checkpoint, use `--episodes 100 --horizon 250` and
`--suite paper --levels-json /kaggle/input/.../paper-layouts.json`. The
default `reference` suite covers the seven pinned JAX-UED prefabs that
correspond to paper names. The `paper` suite requires the five additional
assets `SimpleCrossing`, `FourRooms`, `SmallCorridor`, `LargeCorridor`, and
`PerfectMazeMedium`. The CLI fails if any are missing. Supply a JSON
object mapping level names to ASCII rows using `#` (wall), `.` (free),
`G` (goal) and one of `>`, `v`, `<`, `^` (agent and heading). Use
`--suite custom` for other layouts, including a supplied 51×51 PerfectMaze
with `--horizon 3000`. Evaluation records per-level solved rates, raw
returns, and episode lengths using an independent seed and a frozen
checkpoint. The full paper layouts and multi-seed aggregates remain
separate from the reference evaluation. [Pinned JAX-UED prefabs](https://github.com/DramaCow/jaxued/blob/0f8f1284677375b889e4f13a32c9617cd009f8c4/src/jaxued/environments/maze/level.py)

## PACE fidelity

Each level is scored from its own rollout. On new levels, scoring uses the
current policy without real PPO. On replay, the runner performs real PPO,
then resets the same levels for new trajectories with the updated policy.
The scorer makes one temporary actor update and measures the squared
parameter displacement divided by alpha. The buffer evicts by raw score,
samples by mixed rank and staleness, and refreshes replayed scores using
the existing PLR rule.
The paper does not uniquely specify the provisional optimizer
implementation, learning rate, loss, or actor/critic parameter scope. The
runner uses one gradient-ascent step on the fixed, unnormalized GAE actor
objective, over actor and shared leaves. See the [reproduction assumptions](REPRODUCTION_ASSUMPTIONS.md).

## Local verification

- The Linux bootstrap dry run resolved 93 packages with CPython 3.11.15
  and uv 0.13.5. Use `bash scripts/setup_kaggle.sh --dry-run` to check
  dependency resolution without installing the CUDA runtime packages.
- `python -m pytest -q` passed 36 tests on CPU, including the prefill,
  post-PPO scoring, environment cap, and checkpoint tests.
- A CPU prefill check with 32 environments and one step per level scored
  exactly 500 distinct levels before any PPO update.
- The `algorithm1` CPU smoke resumed from batch two to batch four and
  recorded 144 total interactions. The `legacy` checkpoint resumed under
  its stored protocol and matched the previously saved final training state.
- Capped CPU CLI runs used 80 of 81 steps in `algorithm1` after resume,
  and 48 of 49 steps in `legacy`. Both reported one unspent step and
  stopped before another complete iteration.
- The reference OOD CLI completed two 16-step episodes per level across
  all seven bundled environments and wrote the per-episode JSON.
- The local machine had no NVIDIA GPU. The Kaggle notebook performs the
  GPU and CUDA kernel checks before it launches the smoke run.
