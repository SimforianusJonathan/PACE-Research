# PACE paper-to-code mapping

Primary source: Yuan et al., *PACE: Parameter Change for Unsupervised
Environment Design* (ICML 2026), supplied 22-page `PACE_FINAL.pdf`.
PDF page numbers below count from the start of the file. The researcher's
`PACE_REPRODUCTION_CODEX_HANDOFF.md`, §9, defines the phase gates.

Attachment SHA-256 digests:

- Paper `070c51fad3aa419561992cd90233f1c901df146473d352feeebb816020d03cb7`
- Handoff `0bad40f3776b614f72ce1b6f0c8269f4ec51374bd3dad8c0417e4d9773240af4`

Upstream audited at `0f8f1284677375b889e4f13a32c9617cd009f8c4`
(`git rev-parse HEAD` on the cloned checkout). Source line references point
to that revision. [Pinned example][example] · [Pinned framework][upstream]

| Paper requirement | Pinned JAX-UED location | Required change | Verification | Open fidelity question |
| --- | --- | --- | --- | --- |
| §3.1–3.2, Eqs. 8 and 11, Fig. 2 (pp. 4–5) and App. B.1–B.2 (pp. 15–16). One temporary update on a single level and squared parameter displacement divided by provisional alpha. | `examples/maze_plr.py:389` `compute_score` handles PLR proxies; `:208` `update_actor_critic_rnn` is the real optimizer path. | Phase 1 implements Eq. 8 ascent and literal Eq. 11 in `src/pace/scoring.py`. Receive params, rollout, carry, objective, alpha and a leaf mask explicitly. | Analytic scalar, zero, PyTree reduction, alpha validation, precision, and real train-state/optimizer isolation tests. | The baseline uses one unclipped SGD-style ascent step. Empirical alpha and gradient clipping remain unconfirmed. |
| App. B.3, Eq. 24 (p. 16) uses a GAE-based policy-gradient estimator; §4 and Table 9 describe student PPO. | `examples/maze_plr.py:42` `compute_gae`, `:79` `sample_trajectories_rnn`, `:208` real PPO. `src/jaxued/linen.py` resets recurrent carry from previous transition's done flag. | Phase 1 provides an explicit Maze GAE objective callback for previously collected data, using fixed advantages. MiniGrid uses that actor-only objective and slices `(T, N, ...)` rollouts per level while preserving carry and time axes. | Compare analytic gradients on a stored one-level trajectory, test recurrent reset/carry handling, then compare Phase 2 batch results and permutations against the single-level oracle. | Advantage standardization, use of clipping/entropy/value loss, and rollout objective reduction. |
| §3.1 names policy parameters; Fig. 2 presents separate scoring and student updates. | `examples/maze_plr.py:292` `ActorCritic` shares convolution, direction embedding, and recurrent features. It branches into `actor0`/`actor1` and `critic0`/`critic1`. | Phase 1 scores and updates leaves selected by the caller. The Maze adapter offers an explicit actor-and-shared selector that excludes the critic heads. | Test real Flax parameter names, actor/shared inclusion, critic exclusion, and a synthetic objective with gradients on both selected and excluded leaves. | Paper does not name the parameter subtrees used in the experiment. |
| §3.2/Algorithm 1 (p. 6) and Table 9. New levels score only in MiniGrid. Replay trains the actual student, using a distinct update. | `examples/maze_plr.py:515` `train_step`, `on_new_levels`, `on_replay_levels`, and `on_mutate_levels`. New and replay paths compute MaxMC/PVL before real PPO. | New levels score under the current student. The default `reuse` protocol scores and trains replay on one rollout. The `algorithm1` comparison collects a fresh scoring rollout after replay PPO. | Check the policy snapshot used to generate the scoring rollout, compare optimizer counters, and verify the interaction count across both branches. | Exact rollout use in the authors' experiments. Algorithm 1 puts replay PPO before scoring, and App. A/B describe reuse of existing trajectories. |
| §3.2, Eq. 12 and Table 9. `K=1000`, replay `p=0.7`, rank temperature `0.8`, staleness `0.6`, fill ratio `0.5`. | `src/jaxued/level_sampler.py:67,96,145,276,292,327,375` (initialize, gate, insert, weight mix, rank, staleness, victim). | Prefill 500 distinct scored levels before the main loop. Draw one replay decision per batch. Sample with rank and staleness, evict by raw score, and refresh existing replay entries with JAX-UED PLR's `update_batch()`. | Check exact prefill occupancy including a partial batch, ordering and normalization, replay refresh, and full-buffer eviction. | Initial scoring is unspecified. The initial policy provides these scores in the implementation. Algorithm 1 does not detail refresh of existing replay entries. |
| §4.1 and Table 9. MiniGrid 15×15, 100 wall placement operations, 32 environments, 256 steps, five PPO epochs, one minibatch, no exploratory PPO. | `examples/maze_plr.py:456` `Maze` setup is 13×13. `src/jaxued/environments/maze/util.py:10` draws wall locations with replacement and places agent and goal on free tiles. | Preserve reference settings for smoke tests. Phase 2 configures the paper map size and wall draw count and preserves PPO with linearly decayed learning rate starting at `1e-4`, norm clip `0.5`, gamma `0.995`, GAE lambda `0.98`, clip `0.2`, entropy `0.001`, and value coefficient `0.5`. | Tiny reference sanity run in Phase 0, then assert shape and generator semantics with instrumented branch tests. | Paper does not publish the exact upstream revision or actual number of distinct walls. |
| §4.1, Table 2 (p. 18), and Table 9 (p. 22). 30,000 × 32 × 256 = 245,760,000 reported MiniGrid interactions. | `examples/maze_plr.py:420` logs loop count × environments × rollout length, and `:820` loops over `num_updates // eval_freq`. | Count prefill, PPO, and scoring transitions separately. `--budget env_steps` caps the total actual interactions. `--budget ppo` and `--budget rollout` target their own counters. | Test shared and separate rollouts, the legacy checkpoint path, checkpoint resume, and stopping before an iteration would overshoot the cap. | How the paper reconciled real PPO batches with outer iterations, prefill, scoring, and its reported total. |
| §4.1, App. C.1, Tables 3–4 and Figs. 5, 11–12. Final checkpoints, 12 OOD levels, horizon 250, 100 episodes per seed, 10 seeds. Separate 51×51 PerfectMaze, horizon 3,000. | `examples/maze_plr.py:720,782,848` evaluation, checkpoint evaluation, and eight default prefabs. `src/jaxued/environments/maze/level.py:273` has 11 total keys. | The JAX-UED reference suite provides seven layouts. The paper suite requires five additional assets. See the [Figure 11 layout audit](#figure-11-layout-audit). | Match rendered geometry and start/goal tiles for the seven available prefabs. | Five evaluation instances or generation settings, validation/test overlap (App. C.4), and bootstrap and aggregation details. |
| §4.3, App. C.3, Tables 1, 7–8. Fixed-policy/level CV with 30 independently drawn rollouts per pair. | No equivalent in the Maze reference. | Later reconstruct `SD / mean` per pair and average across pairs (MiniGrid: five checkpoints and five levels). | Repeated rollout oracle and an explicit near-zero mean policy. | Table 7 says `6Rooms` while Fig. 11 names `16Rooms` variants. Some displayed CVs round to zero. |
| §4.2, Tables 2, 6 and 9. Craftax-Symbolic has a 4,096-transition scoring horizon per level and uses exploratory PPO. | Separate Craftax examples. | Deferred. | Future protocol verification. | Main text reports Craftax PACE IQM/gap `0.694/0.341`; Table 6 reports `0.750/0.268`. |

## Figure 11 layout audit

Compared the embedded images on PDF p. 18 with the pinned [JAX-UED prefabs][prefabs].
For each available prefab, the 13×13 interior wall mask and agent/goal tiles
match the 15×15 panel after removing its one-tile border. This verifies the
displayed geometry and endpoints.

| Figure 11 name | Pinned prefab or evidence | Panel size with border |
| --- | --- | ---: |
| 16Rooms | `SixteenRooms`, matched | 15×15 |
| 16RoomsFewerDoors | `SixteenRooms2`, matched | 15×15 |
| SimpleCrossing | Generator family identified, instance unverified | 11×11 |
| FourRooms | Generator family identified, instance unverified | 19×19 |
| Labyrinth | `Labyrinth`, matched | 15×15 |
| Labyrinth2 | `Labyrinth2`, matched | 15×15 |
| SmallCorridor | Generator family identified, instance unverified | 15×15 |
| LargeCorridor | Generator family identified, instance unverified | 21×21 |
| Maze | `StandardMaze`, matched | 15×15 |
| Maze2 | `StandardMaze2`, matched | 15×15 |
| Maze3 | `StandardMaze3`, matched | 15×15 |
| PerfectMazeMedium | Generator family identified, instance unverified | 21×21 |

The five unverified panels have procedural counterparts in the upstream
benchmark family. For example, the cited ACCEL benchmark defines
`PerfectMazeMedium` at 21×21 and regenerates the maze and the endpoints
when it builds a grid. Figure 11 shows one instance per panel. Appendix C.1
does not give generation seeds or establish whether every evaluation episode
used that same instance. The CLI keeps the seven prefabs in `--suite reference`
and requires caller-supplied grids for the other five in `--suite paper`.
That mode evaluates a fixed set of grids. Verify the supplied assets and the
paper's per-episode reset protocol before comparing aggregate results.
[Upstream maze benchmark documentation][minimax] · [ACCEL PerfectMaze implementation][accel-maze]

[prefabs]: https://github.com/DramaCow/jaxued/blob/0f8f1284677375b889e4f13a32c9617cd009f8c4/src/jaxued/environments/maze/level.py
[accel-maze]: https://github.com/facebookresearch/dcd/blob/main/envs/multigrid/mst_maze.py
[minimax]: https://github.com/facebookresearch/minimax/blob/main/docs/envs/maze.md
[example]: https://github.com/DramaCow/jaxued/blob/0f8f1284677375b889e4f13a32c9617cd009f8c4/examples/maze_plr.py
[upstream]: https://github.com/DramaCow/jaxued/tree/0f8f1284677375b889e4f13a32c9617cd009f8c4
