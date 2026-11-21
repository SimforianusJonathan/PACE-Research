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
| Replay ordering and data reuse | Algorithm 1 lists replay PPO before trajectory collection, then scores and updates the buffer. Fig. 2 separates scoring from training, with scoring using the old policy. App. A and B.4 describe reuse and no additional evaluation rollout. | Unresolved for PACE training. The Phase 2 candidate below follows the existing JAX-UED order and uses a shared pre-update snapshot. Scoring after a real PPO step could change gradients and ranks, especially when using data collected under the older policy. |
| Initial fill | Algorithm 1 samples `K·rho` initial levels. Table 9 calls `rho` minimum fill ratio. | The reference begins empty and accumulates scored levels with replay gated until occupancy reaches half capacity. A distinct presampling or scored initialization phase would affect the score distribution and budget. This is open for Phase 2. |
| Eviction and duplicates | §3.2 and Algorithm 1 describe a raw-score threshold. Rank and staleness both affect replay. | The pinned sampler preserves duplicate refresh and uses the minimum mixed-weight slot as its eviction candidate, then compares raw scores. Sampling weights are `0.4 * normalized_rank + 0.6 * normalized_staleness` at paper settings. Staleness is elapsed sampler episode-count since a visit, normalized over occupied slots, with uniform fallback for a populated buffer of zero total staleness. Phase 2 must decide eviction independently of replay weights. |
| Rollout and update accounting | §4.1 and Table 2 report 245.76M interactions and 30,000 PPO updates, while Table 9 disables exploratory gradient updates for MiniGrid. | The reference counts outer rollout batches. Actual PPO applications and replay batches are separate quantities. Phase 2 must track these explicitly and account for any initialization or evaluation steps. |
| Evaluation protocol | §4.1 and App. C.1 report zero-shot transfer. App. C.4 names the same set of 12 layouts as validation environments. | Phase 3 will verify the map assets and document the overlap before reporting OOD statistics. |
| Upstream and hardware provenance | §4 reports JAX-UED and an RTX 5090 for MiniGrid. The paper does not give a framework revision or software lock. | The project pins the inspected JAX-UED revision stated in the mapping. The local CPU setup and smoke are engineering checks. |

## Candidate Phase 2 data flow

```text
select new or replay levels -> collect one batch using theta_old
                            -> compute GAE and keep each sequence's initial carry
                            -> independently score each sequence from theta_old
                            -> insert or refresh scores at the corresponding slots
                            -> on replay, real PPO on the collected batch
                            -> carry forward the real params and optimizer state
```

On the new-level branch the real student stays at `theta_old`. This ordering
is an implementation hypothesis to verify against author code or clarification.
