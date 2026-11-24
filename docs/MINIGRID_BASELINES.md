# MiniGrid baselines

The baseline trainer follows PACE §4.1 and Table 9 for the experiment preset
and uses the pinned JAX-UED examples for rollouts, replay, and editing.
The PACE provisional optimizer is specific to PACE.
[Paper](https://proceedings.mlr.press/v306/yuan26o.html) ·
[JAX-UED DR](https://github.com/DramaCow/jaxued/blob/0f8f1284677375b889e4f13a32c9617cd009f8c4/examples/maze_dr.py) ·
[JAX-UED replay and ACCEL](https://github.com/DramaCow/jaxued/blob/0f8f1284677375b889e4f13a32c9617cd009f8c4/examples/maze_plr.py)

| Method | New levels | Level score | Real PPO batches |
| --- | --- | --- | --- |
| `dr` | 100 random wall placement attempts | None | All new batches |
| `plr` | 100 random wall placement attempts | Time mean of absolute GAE, PACE Eq. 4 | New and replay |
| `plr_perp` | 100 random wall placement attempts | JAX-UED MaxMC | Replay |
| `accel` | Empty rooms and 5 mutation attempts on the previous replay batch | JAX-UED MaxMC | Replay |

The shared preset is a 15×15 Maze, 32 environments, 256 rollout steps, a
250-step episode limit, and PPO with one minibatch, five epochs, learning
rate `1e-4`, max gradient norm `0.5`, gamma `0.995`, GAE lambda `0.98`,
clip `0.2`, entropy coefficient `0.001`, and value coefficient `0.5`.
The learning rate follows the reproduction's linear schedule by actual PPO
batches. All methods use the same CNN/LSTM policy and initialization
procedure.

## Replay choices

The paper's buffer parameters appear under PACE. Baselines start with an
empty buffer and use the standard JAX-UED replay sampler and eviction.
For PLR and PLR⊥ we use Jiang et al.'s DCD reference replay settings
(capacity 4,000, replay 0.5, temperature 0.1, fill threshold 0.5, rank,
staleness 0.3). ACCEL uses the empty-room Parker-Holder reference replay
settings (capacity 4,000, replay 0.8, temperature 0.3, fill threshold 0.5,
rank) and the pinned JAX-UED staleness default 0.3. These are reproduction
choices because the exact PACE baseline replay settings are unpublished.
The maze sizes, wall placement, PPO preset, and scores follow the PACE
paper for this experiment. [PLR settings][plr] · [Robust PLR settings][robust] · [ACCEL settings][accel]

ACCEL follows the pinned branch sequence. After PPO on replayed levels,
the next batch edits those parents, scores the children, and admits them
through replay insertion. An edit can flip a wall, move the goal, or leave
the level unchanged according to the upstream mutator. ACCEL keeps the
parent levels and pending mutation flag in training state.

Run progress counts every unique rollout and reports separate new, replay,
and mutation batches and transitions. Training targets 30,000 completed PPO
batches for the paper's PPO-matched evaluation. A second budget can cap actual
interactions via `--budget env_steps`. This distinguishes Table 2's
245,760,000 reported interactions from the actual simulator cost of each
implemented protocol. Reward metrics distinguish PPO and scoring rollouts.

## Kaggle and evaluation

Use [the baseline notebook](../notebooks/kaggle_minigrid_baselines.ipynb)
with GPU, Internet, and **Save & Run All**. Attach the updated source archive
via **Add Input** as a ZIP or extracted directory. Choose one method
and seed per run. Training starts directly with the 30,000-PPO target.

The notebook invokes `--final-only` and produces no checkpoint or model
file. An interrupted run restarts from its seed. `learning_curve.csv`
records one row per main-loop batch. Baseline curves carry generic level
score columns and label the score type.

The trainer evaluates the current policy in memory at batch zero and every
500 main-loop batches by default, with 20 episodes per layout. The final
evaluation uses 100 episodes per layout. OOD data use an independent
evaluation seed. CV uses up to five evenly spaced PPO milestones and 30
rollouts per checkpoint-equivalent policy and layout. OOD and CV
interactions sit outside the training counters, and their wall time is
reported separately in `run_summary.json`.

DR has OOD results and no level-score CV. For MaxMC CV, each held-out rollout
uses its own best completed return as the reference because the paper
does not specify a historical reference on held-out layouts. No-completion
scores are missing under the pinned MaxMC rule. CV is reported when at
least two valid scores and a positive mean are available. This convention
and the existing reference CV layout choice remain documented assumptions.

`MODE = 'aggregate'` groups complete, matching reports by method. It
supports legacy PACE reports and both PACE rollout protocols. Hold the OOD
layouts and evaluation settings fixed and use matching training seeds. The
default suite has seven verified layouts. Provide the five additional
verified grids to evaluate the paper's full suite.

[plr]: https://github.com/facebookresearch/dcd/blob/main/train_scripts/grid_configs/minigrid/25_blocks/mg_25b_plr.json
[robust]: https://github.com/facebookresearch/dcd/blob/main/train_scripts/grid_configs/minigrid/25_blocks/mg_25b_robust_plr.json
[accel]: https://github.com/facebookresearch/dcd/blob/main/train_scripts/grid_configs/minigrid/60_blocks_uniform/mg_60b_uni_accel_empty.json
