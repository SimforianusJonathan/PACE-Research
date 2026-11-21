# Reproduction assumptions and fidelity gaps

The evidence comes from the supplied `PACE_FINAL.pdf`. Section and equation
numbers refer to that version. The Phase 1 API has no Maze training defaults
for provisional alpha, objective, or parameter scope.

| Choice | Paper evidence | Phase 0–1 decision and effect |
| --- | --- | --- |
| Provisional learning rate `alpha` | Eq. 8 defines a step size, Eq. 11 divides the squared displacement by it, and App. B.1 says the implementation uses a small rate. Table 9 puts MiniGrid's `1e-4` under **PPO**. | Require a finite positive Python scalar explicitly, static when compiling. Unit-test alphas are synthetic. In exact arithmetic a common positive alpha scales all single-SGD scores equally, but floating-point rounding can affect ties and a different update rule can change rankings. |
| Provisional update rule | Eq. 8 and Fig. 2 show gradient ascent. App. B.2 says the scoring phase uses one update. §3.1 calls gradient alignment a local idealization and App. B.1 mentions PPO clipping during training. | Implement one gradient-aligned ascent step on the supplied objective and measure its literal parameter delta. PPO Adam, clipping, scheduling, epoch count, and optimizer state stay in the reference training path. Empirical use of gradient clipping, Adam, or optimizer state in the scorer is unconfirmed. |
| Scoring objective | §3.1 defines an abstract per-level \(J\). App. B.3 Eq. 24 illustrates a mean of policy log-probability gradients weighted by GAE. §4 and Table 9 specify PPO for student training. | Require the caller to supply the objective. The explicit Maze candidate is the time mean of action log probability times fixed, unnormalized GAE, with no value-loss or entropy term. This follows the displayed estimator at the logged policy. Per-level normalization, entropy, clipping, or a critic term could change rankings. A minimization loss must be negated before passing it to the scorer. |
| Parameter scope | §3.1 and Eq. 11 name policy parameters but do not specify the leaves in a shared actor–critic. | Require a boolean leaf mask. The explicit Maze candidate includes convolution, direction embedding, recurrent trunk, and both actor layers. It excludes `critic0` and `critic1`. A critic-only gradient cannot contribute directly, and a joint objective could still move shared leaves through its critic term. |
| Numerics and purity | Eq. 11 measures actual displacement and Fig. 2 discards the provisional parameters. | Form temporary leaves using their parameter dtype and accumulate differences at least in float32. Smaller updates can round to zero. Use finite real floating parameters/gradients and a representable score. The scorer consumes pre-collected data and receives no environment, optimizer, training counter, or RNG. |
| Replay ordering and data reuse | Algorithm 1 lists replay PPO before trajectory collection, then scores and updates the buffer. Fig. 2 separates scoring from training. App. A and B.4 describe reuse and no additional evaluation rollout. | The `algorithm1` protocol collects an on-policy PPO rollout, updates the real student, then resets the same levels and collects a scoring rollout under the updated policy. The exact rollout use in the authors' experiments remains unconfirmed. |
| Initial fill | Algorithm 1 samples `K·rho` initial levels. Table 9 calls `rho` minimum fill ratio. | Generate and score exactly 500 distinct levels by default before the main loop. Score with the initial policy. The paper leaves the initial scores unspecified. |
| Eviction and score refresh | §3.2 describes admission and eviction by raw score. Algorithm 1 does not specify a separate refresh rule for entries already in the buffer. | Evict by lowest raw score. Refresh existing slots with the JAX-UED PLR update rule, even when the new score is lower. Sampling weights are `0.4 * normalized_rank + 0.6 * normalized_staleness` at paper settings. |
| Rollout and update accounting | §4.1 and Table 2 report 245.76M interactions and 30,000 PPO updates, while Table 9 disables exploratory gradient updates for MiniGrid. | Count prefill, scoring, and PPO trajectories separately. The three budget modes target different counters. See the [training budget audit](#minigrid-training-budget-audit). |
| Evaluation protocol | §4.1 and App. C.1 report zero-shot transfer. App. C.4 names the same set of 12 layouts as validation environments. | Phase 3 will verify the map assets and document the overlap before reporting OOD statistics. |
| Upstream and hardware provenance | §4 reports JAX-UED and an RTX 5090 for MiniGrid. The paper does not give a framework revision or software lock. | The project pins the inspected JAX-UED revision stated in the mapping. The local CPU setup and smoke are engineering checks. |

## MiniGrid data flow

```text
prefill: generate levels -> collect trajectories -> score with initial policy -> insert
new:     generate levels -> collect trajectories -> provisional score -> insert
replay:  sample levels -> collect PPO trajectories -> real PPO update
         -> reset the same levels -> collect scoring trajectories with updated policy
         -> provisional score -> refresh replay slots
```

New levels and prefill leave the real student unchanged. Replay carries the
real PPO parameters and optimizer state forward. Each scoring batch computes
one temporary update per level from its corresponding policy snapshot.

## MiniGrid training and Kaggle decisions

- Use the GAE actor objective, actor/shared parameter mask and gradient-aligned
  ascent described above. Full training requires an explicitly chosen
  provisional alpha. The smoke preset's `0.01` serves only as a test value.
- Prefill targets `ceil(K * rho)` distinct entries. The final prefill rollout
  batch shrinks to fit the remaining slots. Duplicate candidates are refreshed
  and the run draws replacements until the target occupancy is reached.
- Evict the lowest raw score, as specified in §3.2. Retain JAX-UED's rank and
  staleness mixture for sampling and its existing duplicate refresh behavior.
  If a batch samples the same replay slot more than once, the last score for
  that slot wins via JAX-UED's sequential update.
- Version 1 checkpoints resume with the `legacy` ordering and shared replay
  trajectories. Version 2 records the protocol and prefill count.
- Use the available JAX-UED prefabs for the reference OOD suite. Require
  explicit assets for missing paper layouts and log the suite and evaluation
  seed. The full 12-level asset mapping and tuning/test overlap still need
  verification before making a scientific reproduction claim.

## MiniGrid training budget audit

Section 4.1 and Table 2 associate 30,000 PPO updates with 245,760,000
environment interactions. Section 4.1 and Table 9 give 32 environments and
256 steps per rollout, so one full collection costs 8,192 transitions.
Multiplying by 30,000 gives the reported total. Algorithm 1 places real PPO
on the replay branch and prefill before the loop. Table 9 sets replay
probability to 0.7 and disables PPO on new levels. The paper does not
reconcile its total with the number of exploration batches, how initial
levels are scored, or whether replay PPO and scoring share trajectories.

Let `B` be completed main-loop batches, `U` be completed replay PPO
batches, and `F` be actual prefill transitions. At the Table 9 rollout size:

| Protocol | Main-loop collections | Actual training interactions |
| --- | ---: | ---: |
| `legacy`, reusing the replay trajectory for scoring and PPO | `B` | `F + 8192 × B`, with `F = 0` in this protocol |
| `algorithm1`, collecting a fresh replay scoring trajectory after PPO | `B + U` | `F + 8192 × (B + U)` |

Under `legacy`, 30,000 **main-loop batches** yield 245.76M interactions,
but exploration occupies some of those batches, so there are fewer than
30,000 real PPO batches. Under `algorithm1`, prefill alone takes at least
128,000 transitions when scoring 500 distinct initial levels. Reaching
30,000 real PPO batches in that protocol requires at least 491,648,000
transitions even before counting any main-loop exploration. Exact agreement
with both published totals cannot be inferred for either implemented
protocol. A literal interaction-cap comparison should label PPO counts
separately.

- `--budget ppo` targets 30,000 real replay PPO batches. Each has five
  optimizer steps at the Table 9 settings. The learning rate decays by
  actual PPO batches using that 30,000-batch horizon and stays at zero
  once the horizon is reached.
- `--budget rollout` targets 30,000 main-loop batches. It keeps the
  prototype's explicit outer-loop budget available.
- `--budget env_steps` caps actual training interactions at 245,760,000
  unless `--max-env-steps` supplies a different cap. Prefill, including
  duplicate candidates, spends this budget too. Before the next iteration,
  the runner checks the cost of the branch selected by the current PRNG
  state. If the whole iteration does not fit, it stops and reports the
  unspent budget. It keeps the selected branch and its full rollout size.

Each log reports `ppo_batches`, `rollout_batches` (outer batches),
`main_collection_batches` (actual collection batches in the main loop),
and `env_steps`. The total is
`prefill_env_steps + ppo_env_steps + scoring_env_steps - shared_env_steps`.
The legacy overlap records trajectories used for both PPO and scoring once.
Prefill counts every simulated step even when a candidate is a duplicate.
Separate OOD evaluation uses a frozen checkpoint and sits outside the
training-interaction budget. The checkpoint stores the selected budget
and the counters needed to continue under the same cap.
