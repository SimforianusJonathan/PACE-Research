"""Export MiniGrid OOD curves, repeated PACE scores, and final aggregates."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import jax
import numpy as np
from jaxued.environments import Maze
from jaxued.utils import compute_max_returns
from jaxued.wrappers import AutoReplayWrapper
from scipy.stats import trim_mean

from train.maze_baselines import level_score
from train.maze_pace import ActorCritic, MazeScoringRollout, compute_gae, sample_trajectories_rnn
from train.pace_checkpoint import read_checkpoint
from train.pace_minigrid import MiniGridConfig, interaction_counts, make_pace_level_scorer, score_levels
from train.pace_ood import FrozenPolicy, load_suite, make_evaluator


DEFAULT_CV_LEVELS = ("16Rooms", "Labyrinth", "Labyrinth2", "Maze", "Maze2")


def write_csv(path, rows, *, append=False):
    with path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint_paths(run_dirs):
    by_batch = {}
    for folder in run_dirs:
        candidates = sorted((folder / "checkpoints").glob("step_*.msgpack"))
        summary_path = folder / "run_summary.json"
        if summary_path.is_file():
            boundary = json.loads(summary_path.read_text(encoding="utf-8"))["counts"]["rollout_batches"]
            candidates = [path for path in candidates if int(path.stem.removeprefix("step_")) <= boundary]
        if not candidates:
            raise ValueError(f"No checkpoints in {folder / 'checkpoints'}")
        for path in candidates:
            batch = int(path.stem.removeprefix("step_"))
            if batch in by_batch and by_batch[batch].resolve() != path.resolve():
                raise ValueError(f"Conflicting checkpoints at batch {batch}. Use one history per training seed.")
            by_batch[batch] = path
    return sorted(by_batch.items())


def curve_checkpoints(checkpoints, interval):
    selected = [checkpoints[0]]
    for point in checkpoints[1:-1]:
        if point[0] - selected[-1][0] >= interval:
            selected.append(point)
    if selected[-1] != checkpoints[-1]:
        selected.append(checkpoints[-1])
    return selected


def collect_training_curve(run_dirs, seed):
    by_batch = {}
    for folder in run_dirs:
        curve_path = folder / "learning_curve.csv"
        if not curve_path.is_file():
            continue
        boundary, _ = checkpoint_paths([folder])[-1]
        with curve_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                value = row.get("rollout_batches")
                if not (value and value.isdecimal()) or int(value) > boundary:
                    continue
                batch = int(value)
                if int(row["seed"]) != seed or batch in by_batch and by_batch[batch] != row:
                    raise ValueError(f"Training curve conflicts with the checkpoint history: {curve_path}")
                by_batch[batch] = row
    return [by_batch[batch] for batch in sorted(by_batch)]


def layout_fingerprint(names, levels):
    digest = hashlib.sha256(json.dumps(list(names)).encode())
    for leaf in jax.tree_util.tree_leaves(levels):
        array = np.asarray(leaf)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_cv_sampler(config, levels, initial_params):
    count, height, width = levels.wall_map.shape
    env = AutoReplayWrapper(Maze(
        max_height=height, max_width=width, agent_view_size=config.agent_view_size, normalize_obs=True,
    ))
    env_params = env.default_params.replace(max_steps_in_episode=config.episode_horizon)
    net = ActorCritic(env.action_space(env_params).n)
    carry = ActorCritic.initialize_carry((count,))
    if config.method == "pace":
        score_one = make_pace_level_scorer(config, initial_params, net.apply)

    def rollout_score(params, key):
        reset_key, rollout_key = jax.random.split(key)
        obs, states = jax.vmap(env.reset_to_level, (0, 0, None))(
            jax.random.split(reset_key, count), levels, env_params,
        )
        student = FrozenPolicy(apply_fn=net.apply, params=params)
        (_, _, _, _, _, last_value), data = sample_trajectories_rnn(
            rollout_key, env, env_params, student, carry, obs, states, count, config.num_steps,
        )
        obs, actions, rewards, dones, _, values, _ = data
        advantages, _ = compute_gae(config.gamma, config.gae_lambda, last_value, values, rewards, dones)
        if config.method != "pace":
            returns = compute_max_returns(dones, rewards) if config.score_type == "maxmc" else None
            return level_score(config.method, dones, values, advantages, returns)
        return score_levels(params, MazeScoringRollout(obs, actions, dones, advantages), carry, score_one)

    @jax.jit
    def sample(params, keys):
        return jax.lax.map(partial(rollout_score, params), keys)

    return sample


class MiniGridAnalysis:
    """Collect evaluation data from saved or live policies without changing training state."""

    def __init__(self, args, config, params):
        self.args, self.config = args, config
        self.names, self.levels = load_suite(args.suite, args.levels_json)
        self.evaluate = make_evaluator(self.levels, config.agent_view_size, args.horizon)
        self.cv_names = ()
        self.cv_fingerprint = None
        self.ood_rows, self.curve_rows, self.ood_metric_rows = [], [], []
        self.cv_rows, self.cv_summary = [], []
        if config.score_type is not None:
            suite = "custom" if args.cv_levels_json else "reference"
            self.cv_names, levels = load_suite(suite, args.cv_levels_json, selected=args.cv_levels)
            self.cv_fingerprint = layout_fingerprint(self.cv_names, levels)
            self.sample_scores = make_cv_sampler(config, levels, params)

    def record(self, params, point, *, ood, cv, final, live_output=None):
        args, config = self.args, self.config
        if ood:
            episodes = args.final_episodes if final else args.curve_episodes
            returns, lengths = self.evaluate(params, episodes, args.eval_seed)
            curve_start, episode_start = len(self.curve_rows), len(self.ood_rows)
            for i, name in enumerate(self.names):
                self.curve_rows.append({
                    **point, "level": name, "final": int(final), "episodes": episodes,
                    "solve_rate": float((returns[:, i] > 0).mean()),
                    "mean_return": float(returns[:, i].mean()),
                })
                for episode, (reward, length) in enumerate(zip(returns[:, i], lengths[:, i]), 1):
                    self.ood_rows.append({
                        **point, "suite": args.suite, "level": name, "final": int(final),
                        "eval_seed": args.eval_seed, "horizon": args.horizon, "episode": episode,
                        "return": float(reward), "length": int(length), "solved": int(reward > 0),
                    })
            solve_rates = (returns > 0).mean(axis=0)
            aggregates = aggregate_metrics(solve_rates[None, :])
            metrics = {
                **point, "suite": args.suite, "final": int(final), "layout_count": len(self.names),
                "episodes_per_layout": episodes, "solve_rate_mean": float(solve_rates.mean()),
                "mean_return": float(returns.mean()),
                "iqm": aggregates["iqm"]["estimate"], "optimality_gap": aggregates["optimality_gap"]["estimate"],
            }
            self.ood_metric_rows.append(metrics)
            if live_output is not None:
                write_csv(live_output / "ood_curve.csv", self.curve_rows[curve_start:], append=True)
                write_csv(live_output / "ood_episodes.csv", self.ood_rows[episode_start:], append=True)
                write_csv(live_output / "ood_metrics_by_eval.csv", [metrics], append=True)
        if cv and self.cv_names:
            batch = point["rollout_batches"]
            key = jax.random.fold_in(jax.random.PRNGKey(args.cv_seed), batch)
            scores = np.asarray(self.sample_scores(params, jax.random.split(key, args.cv_repeats)))
            bad_scores = np.isnan(scores) | np.isposinf(scores) if config.score_type == "maxmc" else ~np.isfinite(scores)
            if bad_scores.any():
                raise ValueError(f"Non-finite CV score at batch {batch}")
            for i, name in enumerate(self.cv_names):
                raw = scores[:, i]
                valid = raw[np.isfinite(raw)]
                mean = float(valid.mean()) if valid.size else None
                sd = float(valid.std(ddof=1)) if valid.size > 1 else None
                self.cv_summary.append({
                    **point, "level": name, "score_type": config.score_type,
                    "repeats": args.cv_repeats, "valid_repeats": int(valid.size),
                    "mean_score": mean, "score_sd": sd, "cv": sd / mean if sd is not None and mean > 0 else None,
                })
                for repeat, score in enumerate(raw, 1):
                    self.cv_rows.append({
                        **point, "level": name, "repeat": repeat, "cv_seed": args.cv_seed,
                        "rollout_steps": config.num_steps, "score_type": config.score_type,
                        "score": float(score) if np.isfinite(score) else None,
                    })
        print(json.dumps({"analysis_batch": point["rollout_batches"], "train_env_steps": point["env_steps"]}), flush=True)

    def finish(self, output_dir, final_point, training_curve_rows):
        args, config = self.args, self.config
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "ood_episodes.csv", self.ood_rows)
        write_csv(output_dir / "ood_curve.csv", self.curve_rows)
        write_csv(output_dir / "ood_metrics_by_eval.csv", self.ood_metric_rows)
        if self.cv_rows:
            write_csv(output_dir / "cv_scores.csv", self.cv_rows)
            write_csv(output_dir / "cv_summary.csv", self.cv_summary)
        defined = [row["cv"] for row in self.cv_summary if row["cv"] is not None]
        ppo = final_point["ppo_batches"]
        report = {
            "method": config.method, "experiment": config.experiment_name, "seed": config.seed, "suite": args.suite,
            "layout_sha256": layout_fingerprint(self.names, self.levels),
            "training_config": {key: value for key, value in asdict(config).items() if key != "seed"},
            "final_checkpoint": final_point["checkpoint"], "ppo_batches": ppo,
            "training_complete": ppo == config.num_updates and config.budget == "ppo",
            "horizon": args.horizon, "episodes": args.final_episodes, "eval_seed": args.eval_seed,
            "training_curve_rows": training_curve_rows,
            "levels": {row["level"]: row["solve_rate"] for row in self.curve_rows if row["final"]},
            "cv_levels": list(self.cv_names),
            "cv_checkpoints": len({row["rollout_batches"] for row in self.cv_summary}),
            "cv_repeats": args.cv_repeats if self.cv_summary else 0, "cv_score_type": config.score_type,
            "cv_layout_sha256": self.cv_fingerprint,
            "cv_mean_over_defined_pairs": float(np.mean(defined)) if defined else None,
            "cv_undefined_pairs": len(self.cv_summary) - len(defined),
        }
        write_json(output_dir / "final_ood.json", report)
        return report


def collect_analysis(args):
    checkpoints = checkpoint_paths(args.run_dirs)
    final_payload = read_checkpoint(checkpoints[-1][1])
    config = MiniGridConfig(**final_payload["config"])
    analysis = MiniGridAnalysis(args, config, final_payload["runner"]["student"]["params"])
    cv_batches = set()
    if analysis.cv_names:
        cv_indices = np.linspace(0, len(checkpoints) - 1, min(args.cv_checkpoints, len(checkpoints)), dtype=int)
        cv_batches = {checkpoints[i][0] for i in cv_indices}
    curve_points = dict(curve_checkpoints(checkpoints, args.ood_every))
    selected = {**curve_points, **{batch: path for batch, path in checkpoints if batch in cv_batches}}
    for batch, path in sorted(selected.items()):
        final = batch == checkpoints[-1][0]
        payload = final_payload if final else read_checkpoint(path)
        if MiniGridConfig(**payload["config"]) != config:
            raise ValueError(f"Checkpoint settings do not match the final checkpoint: {path}")
        state = payload["runner"]
        if int(state["rollout_batches"]) != batch:
            raise ValueError(f"Checkpoint filename disagrees with its batch counter: {path}")
        ppo = int(state["ppo_batches"])
        counts = interaction_counts(
            config, batch, ppo, state["prefill_env_steps"],
            replay=int(state["student"].get("replay_batches", 0)),
            mutations=int(state["student"].get("mutation_batches", 0)),
        )
        point = {
            "method": config.experiment_name, "seed": config.seed, "checkpoint": str(path),
            "rollout_batches": batch, "ppo_batches": ppo, "env_steps": counts["env_steps"],
        }
        analysis.record(state["student"]["params"], point, ood=batch in curve_points, cv=batch in cv_batches, final=final)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_rows = collect_training_curve(args.run_dirs, config.seed)
    if training_rows:
        write_csv(args.output_dir / "training_curve.csv", training_rows)
    analysis.finish(args.output_dir, point, len(training_rows))


def aggregate_metrics(scores):
    metrics = {
        "iqm": {"estimate": float(trim_mean(scores, 0.25, axis=None)), "ci_95": None},
        "optimality_gap": {"estimate": float((1 - scores).mean()), "ci_95": None},
    }
    if scores.shape[0] > 1:
        generator = np.random.default_rng(0)
        indices = generator.integers(scores.shape[0], size=(10_000, *scores.shape))
        bootstrap = scores[indices, np.arange(scores.shape[1])].reshape(10_000, -1)
        metrics["iqm"]["ci_95"] = np.quantile(trim_mean(bootstrap, 0.25, axis=1), [0.025, 0.975]).tolist()
        metrics["optimality_gap"]["ci_95"] = np.quantile((1 - bootstrap).mean(axis=1), [0.025, 0.975]).tolist()
    return metrics


def aggregate_reports(paths, out):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not all(report["training_complete"] for report in reports):
        raise ValueError("Aggregate completed PPO runs only")
    settings = ("suite", "layout_sha256", "horizon", "episodes", "eval_seed", "ppo_batches")
    if not all(all(key in report for key in settings + ("training_config",)) for report in reports):
        raise ValueError("Use final_ood.json reports from the analysis command")
    shared_settings = {key: reports[0][key] for key in settings}
    if any(any(report[key] != value for key, value in shared_settings.items()) for report in reports[1:]):
        raise ValueError("OOD reports must have the same suite, layouts, evaluation settings, and PPO budget")

    cohorts = {}
    for report in reports:
        config = MiniGridConfig(**report["training_config"])
        method = config.experiment_name
        cohorts.setdefault(method, []).append((report, config))

    names = tuple(reports[0]["levels"])
    score_rows, chart_rows, methods = [], [], {}
    for method, cohort in cohorts.items():
        seeds = [report["seed"] for report, _ in cohort]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Supply one final OOD report per seed for {method}")
        config = cohort[0][1]
        if any(other != config for _, other in cohort[1:]):
            raise ValueError(f"Training settings differ across seeds for {method}")
        scores = np.array([[report["levels"][name] for name in names] for report, _ in cohort])
        score_rows.extend(
            {"method": method, "seed": seed, "level": level, "solve_rate": float(score)}
            for seed, row in zip(seeds, scores) for level, score in zip(names, row)
        )
        chart_rows.extend(
            {"method": method, "level": level, "seeds": len(seeds), "mean_solve_rate": float(column.mean()),
             "sd_across_seeds": float(column.std(ddof=1)) if len(seeds) > 1 else None}
            for level, column in zip(names, scores.T)
        )
        overall = scores.mean(axis=1)
        chart_rows.append({
            "method": method, "level": "", "seeds": len(seeds), "mean_solve_rate": float(overall.mean()),
            "sd_across_seeds": float(overall.std(ddof=1)) if len(seeds) > 1 else None,
        })
        methods[method] = {
            "training_config": {key: value for key, value in asdict(config).items() if key != "seed"},
            "training_seeds": seeds, **aggregate_metrics(scores),
        }

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "seed_layout_solve_rates.csv", score_rows)
    write_csv(out / "ood_bar_chart.csv", chart_rows)
    single_method = next(iter(methods.values())) if len(methods) == 1 else {}
    write_json(out / "ood_aggregate.json", {
        **shared_settings, **single_method, "by_method": methods,
        "bootstrap": "percentile over 10000 stratified seed resamples within each layout",
    })


def add_analysis_options(parser):
    parser.add_argument("--suite", choices=("reference", "paper", "custom"), default="reference")
    parser.add_argument("--levels-json", type=Path)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--ood-every", type=int, default=500)
    parser.add_argument("--curve-episodes", type=int, default=20)
    parser.add_argument("--final-episodes", type=int, default=100)
    parser.add_argument("--cv-levels", nargs="+", default=DEFAULT_CV_LEVELS)
    parser.add_argument("--cv-levels-json", type=Path)
    parser.add_argument("--cv-checkpoints", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=30)
    parser.add_argument("--cv-seed", type=int, default=20_000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("evaluate")
    analyze.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    add_analysis_options(analyze)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--reports", nargs="+", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate_reports(args.reports, args.output_dir)
    else:
        if min(args.horizon, args.ood_every, args.curve_episodes, args.final_episodes, args.cv_checkpoints) < 1 or args.cv_repeats < 2:
            parser.error("Evaluation requires positive counts and at least two repeats for CV")
        with jax.default_device(jax.local_devices(backend=args.device)[0]):
            collect_analysis(args)


if __name__ == "__main__":
    main()
