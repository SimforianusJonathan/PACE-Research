# MiniGrid PACE on Kaggle

## Set up the notebook

1. Import [`notebooks/kaggle_minigrid.ipynb`](../notebooks/kaggle_minigrid.ipynb)
   into Kaggle. In notebook settings, enable **Internet** and select a GPU
   accelerator. A two-GPU Kaggle allocation also works with one device exposed
   through `CUDA_VISIBLE_DEVICES=0`.
2. The first code cell pins `REPO_REF` to the trainer commit with the
   `reuse` baseline. Keep that pin to reproduce the smoke and pilot runs.
   The clone workflow needs no dataset. To test a
   local snapshot, attach its source ZIP as a Kaggle Dataset and set `SOURCE`
   to the ZIP path under `/kaggle/input/`. `SOURCE` also accepts a directory
   containing the extracted files. A source snapshot uses its own checkout
   directory. For a committed snapshot, create the ZIP with
   `git archive --format=zip --output=PACE-Research.zip HEAD`.
3. Run all cells. The notebook installs the environment, verifies one CUDA
   device with a JIT-compiled matrix multiplication, and starts a four-batch
   PACE smoke run with an 81-step interaction cap. A fresh process resumes
   from batch two and verifies 80 interactions with one step left. A final cell
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
  --protocol reuse --budget env_steps --max-env-steps 81 \
  --output-dir /kaggle/working/pace/smoke --checkpoint-interval 2 --stop-after 2

"$PACE_PYTHON" -m train.pace_minigrid resume --device gpu \
  --checkpoint /kaggle/working/pace/smoke/checkpoints/step_00000002.msgpack \
  --output-dir /kaggle/working/pace/smoke --checkpoint-interval 2
```

The smoke preset uses two environments, eight steps per rollout, one PPO
epoch, and a four-level buffer. It uses replay probability 1 after half fill
and a toy provisional alpha of 0.01. It keeps the 15×15 Maze and 100 wall
placement operations. Prefill scores two levels. The four main-loop batches
all replay. The 64 scoring steps also serve PPO, so the total is 80
transitions including 16 prefill steps. The cap leaves one step unused.

### MiniGrid pilot

Import [`notebooks/kaggle_minigrid_pilot.ipynb`](../notebooks/kaggle_minigrid_pilot.ipynb)
into Kaggle, select a GPU accelerator, enable Internet, and run all cells.
The notebook clones the configured trainer revision and generates its own
levels. It first scores 500 distinct levels, then runs 24 main-loop batches,
with checkpoints at batches 8, 16, and 24. The main loop collects
196,608 transitions and prefill contributes at least 128,000. PPO reuses
the scoring trajectories on replay, so those transitions count once in
the total. A fresh process resumes from batch 16 and verifies the
PPO counter, optimizer step count, and interaction accounting. The pilot uses seed 0 and provisional alpha 0.01 as
an explicit test choice.

### Full PACE training with live evaluation

Import [the full-run PACE scoring notebook](../notebooks/kaggle_minigrid_pace_policy.ipynb)
into Kaggle. Enable **GPU** and **Internet**. The notebook fetches its pinned
trainer revision from GitHub. Pick `SCORE_METRIC = "l2"`, `"kl"`, or
`"tvd"` and set `SEED`. The default `algorithm1` protocol reaches 30,000
actual PPO batches in a single run. Set `SMOKE_TEST = True` for a four-batch
trial.

The trainer records rollout rows in `training_curve.csv` and measures OOD
performance in memory around each 4,096,000 training interactions. It records
the actual interaction count at each evaluation. The seven reference layouts
use 20 episodes per layout at intermediate evaluations and 100 at the end.
The ZIP contains training and OOD curves, raw OOD episodes, CV data, and
final aggregates. `ood_metrics_by_eval.csv` includes mean return, solve rate,
IQM, and optimality gap (1 − mean OOD solve rate). The workflow stores metrics
instead of model state. Restarting an interrupted run starts from its seed.

Use `python -m train.pace_plot --run-dirs RUN_L2 RUN_KL RUN_TVD --output-dir PLOTS`
to plot completed runs together. Use matching seeds, protocols, and evaluation
settings for the comparison.

### Full training across Kaggle sessions

Import [the full-run notebook](../notebooks/kaggle_minigrid_full.ipynb), select
a GPU and enable Internet. It defaults to session 1, seed 0, `algorithm1`,
and 30,000 real PPO batches. The provisional alpha of `0.01` is a documented
reproduction choice. Use **Save & Run All** to save the session output.

1. Session 1 pauses at main-loop batch 21,000 and saves checkpoints every 250
   main-loop batches. `run_summary.json` gives the latest checkpoint path
   and the PPO and interaction counts at that checkpoint.
2. In the next session, attach the previous notebook output via **Add
   Input**. Set `SESSION = 2` and `RESUME_CHECKPOINT` to the attached
   checkpoint path. Keep the same seed and alpha. Session 2 checks those
   fields and resumes to the stored PPO budget in a new output directory.

Each session caps training at ten hours. If the cap is reached, the notebook
reports the most recent fully written checkpoint that appears in the
training log. Use that checkpoint to continue in session 2. Keep the
training logs and summaries from each session to retain the complete history.

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
levels. The runner makes one replay decision per batch and decays the PPO
learning rate linearly by completed PPO batches. Train separate seeds in
separate output directories.

The default `--budget ppo` counts **30,000 real replay PPO batches** as
stated in Table 9. Use `train --budget rollout` to stop at 30,000 main-loop
batches. For an interaction-matched run, add `--budget env_steps` to the
training command. It caps actual training interactions at 245,760,000 by
default. Supply `--max-env-steps N` for a different cap.

The environment cap includes prefill, PPO, and scoring transitions.
The run stops before a complete next iteration would overshoot, then
reports `budget_remaining` and `stop_reason`. Shared replay trajectories
in `reuse` and `legacy` count once. Logs separately report
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
lock, and device type. Checkpoints resume with their stored protocol.
Version 1 checkpoints resume under `legacy`. Select
`train --protocol algorithm1` to compare fresh replay scoring after PPO,
or `train --protocol legacy` to compare the original empty-buffer flow.

### OOD evaluation

The [MiniGrid baseline guide](MINIGRID_BASELINES.md) covers the DR, PLR,
PLR⊥, and ACCEL notebook, presets, and shared analysis outputs.

```bash
"$PACE_PYTHON" -m train.pace_minigrid eval --device gpu \
  --checkpoint /kaggle/working/pace/smoke/checkpoints/step_00000004.msgpack \
  --suite reference --levels Maze3 --episodes 2 --horizon 16 \
  --output-dir /kaggle/working/pace/smoke/eval
```

For an experimental checkpoint, use `--episodes 100 --horizon 250` and
`--suite paper --levels-json /kaggle/input/.../paper-layouts.json`. The
default `reference` suite covers seven pinned JAX-UED prefabs whose wall
geometry and start/goal tiles match Figure 11. The `paper` suite requires the five additional
assets `SimpleCrossing`, `FourRooms`, `SmallCorridor`, `LargeCorridor`, and
`PerfectMazeMedium`. The CLI fails if any are missing. Supply a JSON
object mapping level names to ASCII rows using `#` (wall), `.` (free),
`G` (goal) and one of `>`, `v`, `<`, `^` (agent and heading). Use
`--suite custom` for other layouts, including a supplied 51×51 PerfectMaze
with `--horizon 3000`. Evaluation records per-level solved rates, raw
returns, and episode lengths using an independent seed and a frozen
checkpoint. The full paper layouts and multi-seed aggregates remain
separate from the reference evaluation. [Figure 11 layout audit](PAPER_TO_CODE_MAPPING.md#figure-11-layout-audit)

### Data to retain for the paper comparison

The full-run notebook records `learning_curve.csv` once per main-loop
batch (8,192 or 16,384 transitions at the paper rollout size). Its columns
include actual interactions, PPO and loop counts, total rewards, completed
episode counts and returns, and PACE scores. PPO and scoring rollouts are
kept in separate columns. Episode-return means leave incomplete tails out
and remain empty for batches with no terminal episode. Use the episode counts
as weights when smoothing rewards. Logging adds zero environment transitions.

After training, attach all session outputs for a seed and select
`SESSION = 3`. Fill `ANALYSIS_RUN_DIRS` with the folders that contain
`checkpoints/` and `learning_curve.csv`. The analysis merges saved
training rows and evaluates OOD every 500 main-loop batches by default
(roughly 2 checkpoint intervals). Use `OOD_EVERY = 250` to evaluate every
periodic checkpoint saved at the default interval. Curve checkpoints use
20 episodes per level and the final checkpoint uses 100, with horizon 250.

- `training_curve.csv` combines the dense training curves for the seed.
- `ood_episodes.csv` keeps raw returns, lengths, and solved flags.
  `ood_curve.csv` summarizes them by checkpoint and layout.
- `cv_scores.csv` keeps all repeated PACE scores and `cv_summary.csv`
  summarizes their sample standard deviations and CVs by checkpoint and
  layout. The defaults are up to five evenly spaced saved checkpoints and
  30 repeats per pair. Zero means have undefined CV. The reference CV set
  uses `16Rooms` as a proxy for the paper's unresolved `6Rooms` label.
  Supply `CV_LAYOUTS_JSON` and matching `CV_LEVELS` to use identified grids.
- `final_ood.json` provides one final score per seed and layout. Session 4
  combines matching final reports into `seed_layout_solve_rates.csv`,
  `ood_bar_chart.csv`, and `ood_aggregate.json`. The latter contains IQM
  and Optimality Gap. With multiple seeds it also records 95% percentile
  intervals from a bootstrap stratified by layout.

With `OOD_LAYOUTS_JSON` empty, the analysis evaluates the seven verified
reference layouts and labels the outputs accordingly. Supply the extra
verified grids to evaluate all 12. The full notebook clones its pinned Git revision by default.
For an attached source archive, set `SOURCE` to the archive path.

- Keep the final checkpoint for each training seed and the run configuration
  and log. Record the actual interactions, PPO batch count, seed, trainer
  revision, dependency lock, and provisional alpha.
- Evaluate each final checkpoint on the same 12 Figure 11 layouts with
  100 episodes per layout and a 250-step horizon. Keep the layout keys and
  input grids, evaluation seed and settings, per-episode return and length,
  and the resulting solve rate per layout and training seed. The OOD CLI
  records per-episode returns and lengths and per-layout solve rates.
- Use the seed-by-layout solve rates to report mean and standard deviation,
  IQM, and optimality gap with 95% confidence intervals for the aggregates.
  The paper reports ten training seeds. The seven verified layouts support
  reference checks while the five remaining assets are being verified.

## PACE fidelity

Each level is scored from its own trajectory. The default `reuse`
baseline uses the current policy for scoring. On replay it passes the
same rollout to real PPO. The `algorithm1` comparison recollects scoring
trajectories under the updated policy. Both protocols score the initial
prefill with the initial policy.

The scorer makes one temporary, unclipped ascent step on the actor’s
fixed, unnormalized GAE objective over the shared CNN/LSTM and actor head,
then measures the squared parameter displacement divided by explicit
alpha. Only real PPO uses the Table 9 gradient clip. The buffer evicts by
raw score, samples by rank and staleness, and refreshes replay scores with
PLR’s `update_batch()`. [Reproduction assumptions](REPRODUCTION_ASSUMPTIONS.md)

## Local verification

- The Linux bootstrap dry run resolved 93 packages with CPython 3.11.15
  and uv 0.13.5. Use `bash scripts/setup_kaggle.sh --dry-run` to check
  dependency resolution without installing the CUDA runtime packages.
- `python -m pytest -q` passed 38 tests on CPU, including both scoring
  protocols, prefill, environment caps, and checkpoint restoration.
- The `reuse` smoke workflow resumed from batch two to batch four, used
  80 of 81 allowed transitions, and completed the two-episode OOD check.
- The seed-0 `reuse` pilot resumed from batch 16 to 24. The final
  checkpoint recorded 15 PPO batches, 75 optimizer steps, and 324,608
  actual transitions, including 128,000 prefill transitions.
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
