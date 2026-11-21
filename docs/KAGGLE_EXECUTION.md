# MiniGrid PACE on Kaggle

## Set up the notebook

1. Import [`notebooks/kaggle_minigrid.ipynb`](../notebooks/kaggle_minigrid.ipynb)
   into Kaggle. In notebook settings, enable **Internet** and select a GPU
   accelerator. A two-GPU Kaggle allocation also works with one device exposed
   through `CUDA_VISIBLE_DEVICES=0`.
2. In the first cell, set `REPO_REF` to a GitHub commit containing the Kaggle
   implementation. If you are running from a local snapshot, archive that
   commit with `git archive --format=zip --output=PACE-Research.zip HEAD`,
   upload the ZIP to a Kaggle Dataset, attach it, and set `SOURCE` to the
   ZIP under `/kaggle/input/`. You can also specify a directory containing
   the extracted source. The first cell uses the existing workspace checkout
   if present. Use a fresh checkout when switching source revisions.
3. Run all cells. The notebook installs the environment, verifies one CUDA
   device with a JIT-compiled matrix multiplication, and starts a four-batch
   PACE smoke run. A fresh process resumes from batch two. A final cell
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
export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH
```

### Smoke and resume

```bash
"$PACE_PYTHON" -m train.pace_minigrid smoke --device gpu \
  --output-dir /kaggle/working/pace/smoke --checkpoint-interval 2 --stop-after 2

"$PACE_PYTHON" -m train.pace_minigrid resume --device gpu \
  --checkpoint /kaggle/working/pace/smoke/checkpoints/step_00000002.msgpack \
  --output-dir /kaggle/working/pace/smoke --checkpoint-interval 2
```

The smoke preset uses two environments, eight steps per rollout, one PPO
epoch, and a four-level buffer. It uses replay probability 1 after half fill
and a toy provisional alpha of 0.01. It keeps the 15×15 Maze and 100 wall
placement operations. With seed 0, the first batch explores and the next
three replay. Batch four represents 64 environment transitions, three
replay PPO batches, and three optimizer steps.

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
stated in Table 9. Exploration also collects trajectories, so this can
exceed the 245.76M interaction budget in Table 2. Use
`train --budget rollout` with the other training arguments to stop at
30,000 total rollout batches, or 245.76M interactions, and report the
actual number of replay PPO batches. Each mode tracks both counters.

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
occupancy, rollout and PPO counters, and the training config. Files are
published atomically. The runner resets its environments and recurrent carry
at the beginning of each batch, so a boundary checkpoint is sufficient for
continuation. For a reproducibility comparison use the same source, package
lock, and device type.

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

Each level is scored from its own rollout and the same pre-update student.
The scorer makes one temporary actor update, then measures the squared
parameter displacement divided by alpha. Replay reuses those trajectories
for real PPO, and the scorer performs no additional environment rollout.
The buffer evicts by raw score and samples by mixed rank and staleness.
New levels score into the buffer until the half-fill gate is reached.
The paper does not uniquely specify the provisional optimizer
implementation, learning rate, loss, or actor/critic parameter scope. The
runner uses one gradient-ascent step on the fixed, unnormalized GAE actor
objective, over actor and shared leaves. See the [reproduction assumptions](REPRODUCTION_ASSUMPTIONS.md).

## Local verification

- The Linux bootstrap dry run resolved 93 packages with CPython 3.11.15
  and uv 0.13.5. Use `bash scripts/setup_kaggle.sh --dry-run` to check
  dependency resolution without installing the CUDA runtime packages.
- `python -m pytest -q` passed 25 tests on CPU.
- A CPU smoke run stopped at batch two. A fresh process restored that
  checkpoint and finished at batch four with 64 interactions and three
  PPO updates. Its final checkpoint was byte-identical to a run that
  completed without interruption.
- The reference OOD CLI completed two 16-step episodes per level across
  all seven bundled environments and wrote the per-episode JSON.
- The local machine had no NVIDIA GPU. The Kaggle notebook performs the
  GPU and CUDA kernel checks before it launches the smoke run.
