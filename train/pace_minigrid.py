"""Single-device MiniGrid PACE using the pinned JAX-UED PPO reference."""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
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

    def __post_init__(self):
        if not math.isfinite(self.provisional_alpha) or self.provisional_alpha <= 0:
            raise ValueError("provisional_alpha must be finite and positive")
        if min(self.num_updates, self.num_steps, self.num_envs, self.num_minibatches, self.ppo_epochs) <= 0:
            raise ValueError("Training counts must be positive")
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches")
        if self.capacity < 1 or not 0 < self.minimum_fill_ratio <= 1:
            raise ValueError("Replay capacity and minimum fill ratio must be positive")
        if self.budget not in {"ppo", "rollout"}:
            raise ValueError("The update budget must count PPO or rollout batches")
        if (
            self.prioritization != "rank"
            or self.exploratory_grad_updates
            or self.provisional_update != "gradient_ascent"
            or self.scoring_objective != "gae_policy_gradient"
            or self.parameter_scope != "actor_and_shared"
        ):
            raise ValueError("Unsupported MiniGrid PACE policy or score configuration")

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


def make_runner(config: MiniGridConfig):
    """Return the seeded state and a compiled function for advancing it."""
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
        return config.learning_rate * (1.0 - ppo_batch / config.num_updates)

    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(schedule, eps=1e-5))
    zero = jnp.zeros((), dtype=jnp.int32)
    student = PaceTrainState.create(
        apply_fn=net.apply, params=params, tx=optimizer,
        sampler=jax.tree_util.tree_map(jnp.asarray, sampler.initialize(example_level)),
    ).replace(step=zero)
    initial = RunnerState(rng, student, zero, zero)
    score_one = partial(
        pace_score_one_level,
        objective_fn=partial(maze_gae_policy_gradient_objective, net.apply),
        provisional_alpha=config.provisional_alpha, policy_param_mask=maze_actor_shared_mask(params),
    )

    def new_levels(buffer, key):
        levels = jax.vmap(generate)(jax.random.split(key, config.num_envs))
        return buffer, jnp.zeros(config.num_envs, jnp.int32), levels

    def step(state, _):
        rng, decision_key, levels_key, reset_key = jax.random.split(state.rng, 4)
        student = state.student
        replay = sampler.sample_replay_decision(student.sampler, decision_key)

        def replay_levels(buffer, key):
            buffer, (indices, levels) = sampler.sample_replay_levels(buffer, key, config.num_envs)
            return buffer, indices, levels

        buffer, indices, levels = jax.lax.cond(replay, replay_levels, new_levels, student.sampler, levels_key)
        obs, env_state = jax.vmap(env.reset_to_level, (0, 0, None))(
            jax.random.split(reset_key, config.num_envs), levels, env_params,
        )
        carry = ActorCritic.initialize_carry((config.num_envs,))
        (rng, _, _, _, _, last_value), trajectory = sample_trajectories_rnn(
            rng, env, env_params, student, carry, obs, env_state, config.num_envs, config.num_steps,
        )
        obs, actions, rewards, dones, log_probs, values, _ = trajectory
        advantages, targets = compute_gae(config.gamma, config.gae_lambda, last_value, values, rewards, dones)
        scores = score_levels(student.params, MazeScoringRollout(obs, actions, dones, advantages), carry, score_one)

        def refresh(buffer):
            return sampler.update_batch(buffer, indices, scores)

        def insert(buffer):
            return sampler.insert_batch(buffer, levels, scores)[0]

        buffer = jax.lax.cond(replay, refresh, insert, buffer)

        def ppo(rng, student):
            (rng, student), _ = update_actor_critic_rnn(
                rng, student, carry, (obs, actions, dones, log_probs, values, targets, advantages),
                config.num_envs, config.num_steps, config.num_minibatches, config.ppo_epochs,
                config.clip_eps, config.entropy_coeff, config.critic_coeff,
            )
            return rng, student

        rng, student = jax.lax.cond(replay, ppo, lambda rng, student: (rng, student), rng, student)
        state = RunnerState(rng, student.replace(sampler=buffer), state.rollout_batches + 1, state.ppo_batches + replay)
        return state, scores

    @partial(jax.jit, static_argnames=("batches",))
    def advance(state, batches: int):
        return jax.lax.scan(step, state, None, length=batches)

    return initial, advance


def _read_runner(path):
    payload = read_checkpoint(path)
    config = MiniGridConfig(**payload["config"])
    template, advance = make_runner(config)
    state = restore_state(payload, template)
    batches, ppo, optimizer_steps = map(int, (state.rollout_batches, state.ppo_batches, state.student.step))
    progress = ppo if config.budget == "ppo" else batches
    if (
        not 0 <= ppo <= batches
        or progress > config.num_updates
        or optimizer_steps != ppo * config.ppo_epochs * config.num_minibatches
    ):
        raise ValueError("Checkpoint contains inconsistent training counters")
    return config, state, advance


def _run_training(args, output_dir):
    if args.command == "resume":
        config, state, advance = _read_runner(args.checkpoint)
    else:
        config = MiniGridConfig.smoke(args.seed) if args.command == "smoke" else MiniGridConfig(
            provisional_alpha=args.provisional_alpha, seed=args.seed, budget=args.budget,
        )
        state, advance = make_runner(config)

    current = int(state.rollout_batches)
    ppo_batches = int(state.ppo_batches)
    progress = ppo_batches if config.budget == "ppo" else current
    if progress >= config.num_updates:
        raise ValueError("The checkpoint has already reached its training budget")
    if args.stop_after is not None and (
        args.stop_after <= current or (config.budget == "rollout" and args.stop_after > config.num_updates)
    ):
        raise ValueError("--stop-after must advance the rollout counter within the configured budget")
    checkpoints = sorted((output_dir / "checkpoints").glob("step_*.msgpack"))
    if checkpoints and (args.command != "resume" or checkpoints[-1].resolve() != args.checkpoint.resolve()):
        raise ValueError("Use an empty output directory or resume from its latest checkpoint")

    print(json.dumps({"config": asdict(config), "start_batch": current, "stop_after": args.stop_after}), flush=True)
    while True:
        # A rollout can advance the PPO counter by at most one. Limiting
        # the chunk to the remaining budget prevents overshooting either mode.
        count = min(args.chunk_size, config.num_updates - progress)
        if args.stop_after is not None:
            count = min(count, args.stop_after - current)
        if args.checkpoint_interval:
            count = min(count, args.checkpoint_interval - current % args.checkpoint_interval)
        state, scores = advance(state, count)
        scores = np.asarray(scores)
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise FloatingPointError(f"Invalid PACE score after rollout batch {current}")
        current = int(state.rollout_batches)
        ppo_batches = int(state.ppo_batches)
        progress = ppo_batches if config.budget == "ppo" else current
        finished = progress == config.num_updates or current == args.stop_after
        checkpoint = None
        if finished or (args.checkpoint_interval and current % args.checkpoint_interval == 0):
            checkpoint = checkpoint_path(output_dir, current)
            save_checkpoint(checkpoint, asdict(config), state)
        print(json.dumps({
            "rollout_batches": current, "ppo_batches": ppo_batches,
            "optimizer_steps": int(state.student.step), "buffer_size": int(state.student.sampler["size"]),
            "env_steps": current * config.num_envs * config.num_steps,
            "pace_score_mean": float(scores.mean()), "pace_score_max": float(scores.max()),
            "checkpoint": str(checkpoint) if checkpoint else None,
        }), flush=True)
        if finished:
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
            if name == "train":
                command.add_argument("--provisional-alpha", type=float, required=True)
                command.add_argument("--budget", choices=("ppo", "rollout"), default="ppo",
                                     help="Count Table 9 PPO updates or match Table 2 rollout interactions")
    return parser, parser.parse_args()


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
