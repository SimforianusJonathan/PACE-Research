"""Zero-shot MiniGrid evaluation for fixed, named Maze layouts."""

import json

import jax
import jax.numpy as jnp
import numpy as np
from jaxued.environments import Maze
from jaxued.environments.maze import Level

from train.maze_pace import ActorCritic, evaluate_rnn


REFERENCE_PREFABS = {
    "16Rooms": "SixteenRooms",
    "16RoomsFewerDoors": "SixteenRooms2",
    "Labyrinth": "Labyrinth",
    "Labyrinth2": "Labyrinth2",
    "Maze": "StandardMaze",
    "Maze2": "StandardMaze2",
    "Maze3": "StandardMaze3",
}
PAPER_LEVELS = (
    "16Rooms", "16RoomsFewerDoors", "SimpleCrossing", "FourRooms",
    "Labyrinth", "Labyrinth2", "SmallCorridor", "LargeCorridor",
    "Maze", "Maze2", "Maze3", "PerfectMazeMedium",
)


def _load_layout(name, rows):
    if isinstance(rows, str):
        rows = rows.strip().splitlines()
    if (
        not isinstance(rows, list) or not rows
        or any(not isinstance(row, str) or len(row) != len(rows[0]) for row in rows)
    ):
        raise ValueError(f"Layout {name} must be a nonempty rectangular grid")
    chars = "".join(rows)
    if set(chars) - set("#.G<>^v") or chars.count("G") != 1 or sum(chars.count(c) for c in "<>^v") != 1:
        raise ValueError(f"Layout {name} needs exactly one goal, one agent, and only Maze tile symbols")
    return Level.from_str("\n".join(rows))


def load_suite(suite, layout_path=None, selected=None):
    supplied = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path else {}
    if not isinstance(supplied, dict) or any(not name for name in supplied):
        raise ValueError("The layout JSON must map nonempty names to ASCII grids")
    if suite != "custom" and set(supplied) & set(REFERENCE_PREFABS):
        raise ValueError("Use the custom suite to override a JAX-UED prefab")
    prefabs = {} if suite == "custom" else REFERENCE_PREFABS
    names = PAPER_LEVELS if suite == "paper" else selected or tuple(prefabs or supplied)
    if suite == "paper" and selected:
        raise ValueError("Use the reference or custom suite to evaluate a subset")
    missing = [name for name in names if name not in prefabs and name not in supplied]
    if missing:
        raise ValueError(f"Provide these layouts via --levels-json: {', '.join(missing)}")
    if not names or len(names) != len(set(names)):
        raise ValueError("Select at least one level with no duplicate names")
    levels = [
        Level.load_prefabs([prefabs[name]]) if name in prefabs else _load_layout(name, supplied[name])
        for name in names
    ]
    # load_prefabs returns a leading level axis even for one name.
    levels = [
        jax.tree_util.tree_map(lambda x: x[0], level) if name in prefabs else level
        for name, level in zip(names, levels)
    ]
    return names, Level.stack(levels)


def evaluate_checkpoint(args, config, state):
    if args.episodes <= 0 or args.horizon <= 0:
        raise ValueError("Evaluation episode count and horizon must be positive")
    names, levels = load_suite(args.suite, args.levels_json, args.levels)
    height, width = levels.wall_map.shape[-2:]
    env = Maze(max_height=height, max_width=width, agent_view_size=config.agent_view_size, normalize_obs=True)
    env_params = env.default_params.replace(max_steps_in_episode=args.horizon)
    carry = ActorCritic.initialize_carry((len(names),))

    def episode(key, student):
        reset_key, rollout_key = jax.random.split(key)
        observations, initial = jax.vmap(env.reset_to_level, (0, 0, None))(
            jax.random.split(reset_key, len(names)), levels, env_params,
        )
        _, rewards, lengths = evaluate_rnn(
            rollout_key, env, env_params, student, carry, observations, initial, args.horizon,
        )
        mask = jnp.arange(args.horizon)[:, None] < lengths
        return (rewards * mask).sum(axis=0), lengths

    run_episodes = jax.jit(jax.vmap(episode, (0, None)))
    keys = jax.random.split(jax.random.PRNGKey(args.eval_seed), args.episodes)
    results = [jax.device_get(run_episodes(keys[i:i + 8], state.student)) for i in range(0, args.episodes, 8)]
    returns, lengths = map(np.concatenate, zip(*results))
    solved = returns > 0
    summary = {
        "checkpoint": str(args.checkpoint),
        "training_seed": config.seed, "rollout_batches": int(state.rollout_batches),
        "eval_seed": args.eval_seed, "suite": args.suite, "episodes": args.episodes, "horizon": args.horizon,
        "mean_solve_rate": float(solved.mean()),
        "levels": {
            name: {
                "solve_rate": float(solved[:, i].mean()),
                "returns": returns[:, i].tolist(), "episode_lengths": lengths[:, i].tolist(),
            } for i, name in enumerate(names)
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"ood-{args.suite}-seed{config.seed}-step{int(state.rollout_batches)}-eval{args.eval_seed}.json"
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evaluation": str(path), "mean_solve_rate": summary["mean_solve_rate"]}), flush=True)
