"""Single-device MiniGrid PACE using the pinned JAX-UED PPO reference."""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from jaxued.environments import Maze
from jaxued.environments.maze import make_level_generator
from jaxued.level_sampler import LevelSampler
from jaxued.wrappers import AutoReplayWrapper

from pace import pace_score_one_level
from train.maze_pace import (
    ActorCritic,
    MazeScoringRollout,
    compute_gae,
    maze_actor_shared_mask,
    maze_gae_policy_gradient_objective,
    sample_trajectories_rnn,
    update_actor_critic_rnn,
)
from train.pace_checkpoint import checkpoint_path, read_checkpoint, restore_state, save_checkpoint


PAPER_ENV_STEPS = 245_760_000


@dataclass(frozen=True)
class MiniGridConfig:
    """Paper Table 9 plus the recorded implementation choices."""

    provisional_alpha: float
    seed: int = 0
    num_updates: int = 30_000
    budget: str = "ppo"
    num_steps: int = 256
    num_envs: int = 32
    num_minibatches: int = 1
    ppo_epochs: int = 5
    learning_rate: float = 1e-4
    max_grad_norm: float = 0.5
    gamma: float = 0.995
    gae_lambda: float = 0.98
    clip_eps: float = 0.2
    entropy_coeff: float = 1e-3
    critic_coeff: float = 0.5
    capacity: int = 1000
    replay_prob: float = 0.7
    staleness_coeff: float = 0.6
    temperature: float = 0.8
    minimum_fill_ratio: float = 0.5
    prioritization: str = "rank"
    exploratory_grad_updates: bool = False
    maze_size: int = 15
    n_walls: int = 100
    agent_view_size: int = 5
    scoring_objective: str = "gae_policy_gradient"
    parameter_scope: str = "actor_and_shared"
    provisional_update: str = "gradient_ascent"
    protocol: str = "algorithm1"
    max_env_steps: int | None = None

    def __post_init__(self):
        if not math.isfinite(self.provisional_alpha) or self.provisional_alpha <= 0:
            raise ValueError("provisional_alpha must be finite and positive")
        if min(self.num_updates, self.num_steps, self.num_envs, self.num_minibatches, self.ppo_epochs) <= 0:
            raise ValueError("Training counts must be positive")
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches")
        if self.capacity < 1 or not 0 < self.minimum_fill_ratio <= 1:
            raise ValueError("Replay capacity and minimum fill ratio must be positive")
        if self.budget not in {"ppo", "rollout", "env_steps"}:
            raise ValueError("The budget must count PPO batches, main-loop batches, or environment steps")
        if self.max_env_steps is not None and (self.budget != "env_steps" or self.max_env_steps <= 0):
            raise ValueError("A positive max_env_steps is only valid with the env_steps budget")
        if self.protocol not in {"algorithm1", "legacy"}:
            raise ValueError("Unknown training protocol")
        if (
            self.prioritization != "rank"
            or self.exploratory_grad_updates
            or self.provisional_update != "gradient_ascent"
            or self.scoring_objective != "gae_policy_gradient"
            or self.parameter_scope != "actor_and_shared"
        ):
            raise ValueError("Unsupported MiniGrid PACE policy or score configuration")

    @property
    def initial_buffer_size(self) -> int:
        return math.ceil(self.capacity * self.minimum_fill_ratio)

    @property
    def budget_limit(self) -> int:
        if self.budget == "env_steps":
            return self.max_env_steps if self.max_env_steps is not None else PAPER_ENV_STEPS
        return self.num_updates

    @classmethod
    def smoke(cls, seed: int = 0):
        return cls(
            provisional_alpha=0.01, seed=seed, num_updates=4, num_steps=8, num_envs=2,
            budget="rollout", ppo_epochs=1, capacity=4, replay_prob=1.0,
        )


class PaceTrainState(TrainState):
    sampler: dict


@struct.dataclass
class RunnerState:
    rng: jax.Array
    student: PaceTrainState
    rollout_batches: jax.Array
    ppo_batches: jax.Array
    prefill_env_steps: jax.Array


class PaceLevelSampler(LevelSampler):
    """Retain the highest raw scores while sampling with rank and staleness."""

    def _get_next_idx(self, sampler):
        # _insert_new compares the candidate's score with the chosen slot.
        return jnp.where(sampler["size"] < self.capacity, sampler["size"], sampler["scores"].argmin())


def score_levels(params, rollout, initial_carries, score_one):
    """Score each trajectory from the same student snapshot with bounded memory."""
    level_rollouts = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), rollout)
    return jax.lax.map(
        lambda data: score_one(params, data[0], data[1]), (level_rollouts, initial_carries),
    )


def interaction_counts(config, rollout_batches, ppo_batches, prefill_env_steps):
    """Account for collected transitions and count shared trajectories once."""
    rollout_batches = int(rollout_batches)
    ppo_batches = int(ppo_batches)
    batch_steps = config.num_envs * config.num_steps
    scoring = rollout_batches * batch_steps
    ppo = ppo_batches * batch_steps
    prefill = int(prefill_env_steps)
    shared = ppo if config.protocol == "legacy" else 0
    return {
        "env_steps": prefill + scoring + ppo - shared,
        "prefill_env_steps": prefill,
        "ppo_env_steps": ppo,
        "scoring_env_steps": scoring,
        "shared_env_steps": shared,
        "main_collection_batches": rollout_batches + (ppo_batches if config.protocol == "algorithm1" else 0),
    }


def budget_progress(config, state, env_steps):
    if config.budget == "ppo":
        return int(state.ppo_batches)
    if config.budget == "rollout":
        return int(state.rollout_batches)
    return env_steps


def make_runner(config: MiniGridConfig, *, prefill=True):
    """Return the seeded state and a compiled function for advancing it."""
    env_step_limit = config.budget_limit if config.budget == "env_steps" else None
    needs_prefill = prefill and config.protocol == "algorithm1"
    if needs_prefill and env_step_limit is not None:
        minimum_prefill_steps = config.initial_buffer_size * config.num_steps
        if env_step_limit < minimum_prefill_steps:
            raise ValueError(f"Prefill needs at least {minimum_prefill_steps} environment steps")

    env = AutoReplayWrapper(Maze(
        max_width=config.maze_size, max_height=config.maze_size,
        agent_view_size=config.agent_view_size, normalize_obs=True,
    ))
    env_params = env.default_params
    generate = make_level_generator(config.maze_size, config.maze_size, config.n_walls)
    sampler = PaceLevelSampler(
        capacity=config.capacity, replay_prob=config.replay_prob, staleness_coeff=config.staleness_coeff,
        minimum_fill_ratio=config.minimum_fill_ratio, prioritization=config.prioritization,
        prioritization_params={"temperature": config.temperature}, duplicate_check=True,
    )
    rng, init_key, level_key, reset_key = jax.random.split(jax.random.PRNGKey(config.seed), 4)
    example_level = generate(level_key)
    obs, _ = env.reset_to_level(reset_key, example_level, env_params)
    net = ActorCritic(env.action_space(env_params).n)
    inputs = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
    params = net.init(init_key, (inputs, jnp.zeros((1, 1), dtype=bool)), ActorCritic.initialize_carry((1,)))

    def schedule(step):
        ppo_batch = step // (config.ppo_epochs * config.num_minibatches)
        return config.learning_rate * jnp.maximum(0.0, 1.0 - ppo_batch / config.num_updates)

    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(schedule, eps=1e-5))
    zero = jnp.zeros((), dtype=jnp.int32)
    student = PaceTrainState.create(
        apply_fn=net.apply, params=params, tx=optimizer,
        sampler=jax.tree_util.tree_map(jnp.asarray, sampler.initialize(example_level)),
    ).replace(step=zero)
    initial = RunnerState(rng, student, zero, zero, zero)
    score_one = partial(
        pace_score_one_level,
        objective_fn=partial(maze_gae_policy_gradient_objective, net.apply),
        provisional_alpha=config.provisional_alpha, policy_param_mask=maze_actor_shared_mask(params),
    )

    def new_levels(buffer, key):
        levels = jax.vmap(generate)(jax.random.split(key, config.num_envs))
        return buffer, jnp.zeros(config.num_envs, jnp.int32), levels

    def collect(rng, reset_key, student, levels, num_envs):
        obs, env_state = jax.vmap(env.reset_to_level, (0, 0, None))(
            jax.random.split(reset_key, num_envs), levels, env_params,
        )
        carry = ActorCritic.initialize_carry((num_envs,))
        (rng, _, _, _, _, last_value), trajectory = sample_trajectories_rnn(
            rng, env, env_params, student, carry, obs, env_state, num_envs, config.num_steps,
        )
        obs, actions, rewards, dones, log_probs, values, _ = trajectory
        advantages, targets = compute_gae(config.gamma, config.gae_lambda, last_value, values, rewards, dones)
        return rng, carry, (obs, actions, dones, log_probs, values, targets, advantages)

    def score(student, carry, batch):
        obs, actions, dones, _, _, _, advantages = batch
        rollout = MazeScoringRollout(obs, actions, dones, advantages)
        return score_levels(student.params, rollout, carry, score_one)

    def ppo(rng, student, carry, batch):
        updated, _ = update_actor_critic_rnn(
            rng, student, carry, batch, config.num_envs, config.num_steps,
            config.num_minibatches, config.ppo_epochs, config.clip_eps,
            config.entropy_coeff, config.critic_coeff,
        )
        return updated

    @partial(jax.jit, static_argnames=("num_levels",))
    def fill_batch(state, num_levels):
        rng, levels_key, reset_key = jax.random.split(state.rng, 3)
        levels = jax.vmap(generate)(jax.random.split(levels_key, num_levels))
        rng, carry, batch = collect(rng, reset_key, state.student, levels, num_levels)
        scores = score(state.student, carry, batch)
        buffer, _ = sampler.insert_batch(state.student.sampler, levels, scores)
        return state.replace(
            rng=rng, student=state.student.replace(sampler=buffer),
            prefill_env_steps=state.prefill_env_steps + num_levels * config.num_steps,
        ), scores

    def plan_step(state):
        rng, decision_key, levels_key, reset_key = jax.random.split(state.rng, 4)
        replay = sampler.sample_replay_decision(state.student.sampler, decision_key)
        return rng, levels_key, reset_key, replay

    def step(state, _):
        rng, levels_key, reset_key, replay = plan_step(state)
        student = state.student

        def replay_levels(buffer, key):
            buffer, (indices, levels) = sampler.sample_replay_levels(buffer, key, config.num_envs)
            return buffer, indices, levels

        buffer, indices, levels = jax.lax.cond(replay, replay_levels, new_levels, student.sampler, levels_key)
        rng, carry, batch = collect(rng, reset_key, student, levels, config.num_envs)

        if config.protocol == "algorithm1":
            def collect_after_ppo(rng, student, carry, batch):
                rng, student = ppo(rng, student, carry, batch)
                rng, scoring_reset_key = jax.random.split(rng)
                rng, carry, batch = collect(rng, scoring_reset_key, student, levels, config.num_envs)
                return rng, student, carry, batch

            rng, student, carry, batch = jax.lax.cond(
                replay, collect_after_ppo, lambda *args: args, rng, student, carry, batch,
            )

        scores = score(student, carry, batch)

        def refresh(buffer):
            # PLR refreshes sampled slots in order, even when new scores are lower.
            return sampler.update_batch(buffer, indices, scores)

        def insert(buffer):
            return sampler.insert_batch(buffer, levels, scores)[0]

        buffer = jax.lax.cond(replay, refresh, insert, buffer)

        if config.protocol == "legacy":
            rng, student = jax.lax.cond(
                replay, ppo, lambda rng, student, *_: (rng, student), rng, student, carry, batch,
            )

        state = state.replace(
            rng=rng, student=student.replace(sampler=buffer),
            rollout_batches=state.rollout_batches + 1, ppo_batches=state.ppo_batches + replay,
        )
        return state, scores

    @partial(jax.jit, static_argnames=("batches",))
    def advance_batches(state, batches: int):
        return jax.lax.scan(step, state, None, length=batches)

    @jax.jit
    def next_is_replay(state):
        return plan_step(state)[-1]

    def advance(state, batches: int):
        if env_step_limit is not None:
            counts = interaction_counts(config, state.rollout_batches, state.ppo_batches, state.prefill_env_steps)
            remaining = env_step_limit - counts["env_steps"]
            batch_steps = config.num_envs * config.num_steps
            max_cost = 2 * batch_steps if config.protocol == "algorithm1" else batch_steps
            safe_batches = remaining // max_cost
            if safe_batches > 0:
                batches = min(batches, safe_batches)
            else:
                # Peek without consuming the key. A pending replay is never
                # substituted with a cheaper exploration to fit the budget.
                if remaining < batch_steps or bool(next_is_replay(state)):
                    return state, jnp.empty((0, config.num_envs), dtype=jnp.float32)
                batches = min(batches, 1)
        return advance_batches(state, batches)

    if needs_prefill:
        while (remaining := config.initial_buffer_size - int(initial.student.sampler["size"])) > 0:
            num_levels = min(config.num_envs, remaining)
            if env_step_limit is not None:
                used = int(initial.prefill_env_steps)
                num_levels = min(num_levels, (env_step_limit - used) // config.num_steps)
                if not num_levels:
                    occupied = config.initial_buffer_size - remaining
                    raise ValueError(
                        f"Prefill used {used} environment steps within cap {env_step_limit} "
                        f"and filled {occupied}/{config.initial_buffer_size} slots"
                    )
            initial, scores = fill_batch(initial, num_levels)
            check_scores(scores, f"during prefill after {int(initial.prefill_env_steps)} environment steps")
    return initial, advance


def check_scores(scores, phase):
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise FloatingPointError(f"Invalid PACE score {phase}")
    return scores


def _read_runner(path):
    payload = read_checkpoint(path)
    config = MiniGridConfig(**payload["config"])
    template, advance = make_runner(config, prefill=False)
    state = restore_state(payload, template)
    batches, ppo, optimizer_steps = map(int, (state.rollout_batches, state.ppo_batches, state.student.step))
    counts = interaction_counts(config, batches, ppo, state.prefill_env_steps)
    progress = budget_progress(config, state, counts["env_steps"])
    if (
        not 0 <= ppo <= batches
        or progress > config.budget_limit
        or optimizer_steps != ppo * config.ppo_epochs * config.num_minibatches
    ):
        raise ValueError("Checkpoint contains inconsistent training counters")
    return config, state, advance


def _run_training(args, output_dir):
    checkpoints = sorted((output_dir / "checkpoints").glob("step_*.msgpack"))
    if checkpoints and (args.command != "resume" or checkpoints[-1].resolve() != args.checkpoint.resolve()):
        raise ValueError("Use an empty output directory or resume from its latest checkpoint")
    if args.command == "resume":
        config, state, advance = _read_runner(args.checkpoint)
    else:
        settings = {"budget": args.budget, "protocol": args.protocol, "max_env_steps": args.max_env_steps}
        config = (
            replace(MiniGridConfig.smoke(args.seed), **settings)
            if args.command == "smoke"
            else MiniGridConfig(provisional_alpha=args.provisional_alpha, seed=args.seed, **settings)
        )
        if config.protocol == "algorithm1":
            print(json.dumps({"prefill_target": config.initial_buffer_size}), flush=True)
        state, advance = make_runner(config)

    current = int(state.rollout_batches)
    ppo_batches = int(state.ppo_batches)
    counts = interaction_counts(config, current, ppo_batches, state.prefill_env_steps)
    progress = budget_progress(config, state, counts["env_steps"])
    if args.command == "resume" and progress >= config.budget_limit:
        raise ValueError("The checkpoint has already reached its training budget")
    if args.stop_after is not None and (
        args.stop_after <= current or (config.budget == "rollout" and args.stop_after > config.budget_limit)
    ):
        raise ValueError("--stop-after must advance the rollout counter within the configured budget")
    print(json.dumps({
        "config": asdict(config), "budget_limit": config.budget_limit, "start_batch": current,
        "stop_after": args.stop_after, "buffer_size": int(state.student.sampler["size"]), **counts,
    }), flush=True)
    while True:
        count = args.chunk_size
        if config.budget != "env_steps":
            # Each outer iteration can advance the PPO counter by at most one.
            count = min(count, config.budget_limit - progress)
        if args.stop_after is not None:
            count = min(count, args.stop_after - current)
        if args.checkpoint_interval:
            count = min(count, args.checkpoint_interval - current % args.checkpoint_interval)
        state, scores = advance(state, count)
        current = int(state.rollout_batches)
        ppo_batches = int(state.ppo_batches)
        counts = interaction_counts(config, current, ppo_batches, state.prefill_env_steps)
        scores = check_scores(scores, f"after batch {current} and {counts['env_steps']} environment steps")
        progress = budget_progress(config, state, counts["env_steps"])
        stop_reason = None
        if not scores.size or progress == config.budget_limit:
            stop_reason = f"{config.budget}_budget"
        elif current == args.stop_after:
            stop_reason = "stop_after"
        checkpoint = None
        if stop_reason or (args.checkpoint_interval and current % args.checkpoint_interval == 0):
            checkpoint = checkpoint_path(output_dir, current)
            if scores.size or not checkpoint.exists():
                save_checkpoint(checkpoint, asdict(config), state)
        print(json.dumps({
            "rollout_batches": current, "ppo_batches": ppo_batches,
            "optimizer_steps": int(state.student.step), "buffer_size": int(state.student.sampler["size"]),
            **counts,
            "budget_remaining": config.budget_limit - progress, "stop_reason": stop_reason,
            "pace_score_mean": float(scores.mean()) if scores.size else None,
            "pace_score_max": float(scores.max()) if scores.size else None,
            "checkpoint": str(checkpoint) if checkpoint else None,
        }), flush=True)
        if stop_reason:
            break


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "train", "resume", "eval"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
        command.add_argument("--output-dir", type=Path)
        if name in {"resume", "eval"}:
            command.add_argument("--checkpoint", type=Path, required=True)
        if name == "eval":
            command.add_argument("--suite", choices=("reference", "paper", "custom"), default="reference")
            command.add_argument("--levels-json", type=Path)
            command.add_argument("--levels", nargs="+")
            command.add_argument("--episodes", type=int, default=100)
            command.add_argument("--horizon", type=int, default=250)
            command.add_argument("--eval-seed", type=int, default=10000)
        else:
            command.add_argument("--stop-after", type=int, help="Stop at this absolute rollout batch")
            command.add_argument("--checkpoint-interval", type=int, default=100)
            command.add_argument("--chunk-size", type=int, default=10)
            if name != "resume":
                command.add_argument("--seed", type=int, default=0)
                command.add_argument("--budget", choices=("ppo", "rollout", "env_steps"),
                                     default="ppo" if name == "train" else "rollout",
                                     help="Count PPO updates, main-loop batches, or all training environment steps")
                command.add_argument("--max-env-steps", type=int,
                                     help="Cap actual training interactions (train default: 245760000)")
                command.add_argument("--protocol", choices=("algorithm1", "legacy"), default="algorithm1",
                                     help="Use prefill and post-PPO scoring or the original prototype flow")
            if name == "train":
                command.add_argument("--provisional-alpha", type=float, required=True)
    args = parser.parse_args()
    if args.command in {"train", "smoke"}:
        if args.max_env_steps is not None and (args.budget != "env_steps" or args.max_env_steps <= 0):
            parser.error("--max-env-steps requires --budget env_steps and a positive cap")
        if args.budget == "env_steps" and args.max_env_steps is None:
            if args.command == "smoke":
                parser.error("--budget env_steps requires an explicit --max-env-steps for smoke")
            args.max_env_steps = PAPER_ENV_STEPS
    return parser, args


def main():
    parser, args = _cli()
    kaggle = Path("/kaggle/working")
    on_kaggle = sys.platform == "linux" and kaggle.is_dir()
    if args.output_dir is None:
        root = kaggle / "pace" if on_kaggle else Path("outputs/pace")
        if args.command == "resume":
            checkpoint_dir = args.checkpoint.resolve().parent
            if checkpoint_dir.name == "checkpoints" and (
                not on_kaggle or checkpoint_dir.is_relative_to(kaggle)
            ):
                args.output_dir = checkpoint_dir.parent
            else:
                args.output_dir = root / f"resume-{args.checkpoint.stem}"
        else:
            run_name = "eval" if args.command == "eval" else f"{args.command}-{args.seed}"
            args.output_dir = root / run_name
    if args.command != "eval" and (args.checkpoint_interval < 0 or args.chunk_size <= 0):
        parser.error("Checkpoint interval must be nonnegative and chunk size must be positive")
    try:
        device = jax.local_devices(backend=args.device)[0]
    except RuntimeError as error:
        parser.error(f"{args.device} device unavailable: {error}")
    print(f"Using JAX device {device}", flush=True)
    with jax.default_device(device):
        if args.command == "eval":
            from train.pace_ood import evaluate_checkpoint

            config, state, _ = _read_runner(args.checkpoint)
            evaluate_checkpoint(args, config, state)
        else:
            _run_training(args, args.output_dir)


if __name__ == "__main__":
    main()
