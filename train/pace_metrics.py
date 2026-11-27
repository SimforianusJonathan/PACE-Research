"""Dense rollout statistics for MiniGrid training curves."""

import csv

import jax
import jax.numpy as jnp
import numpy as np


ROLLOUT_FIELDS = ("episodes", "solved", "return_sum", "episode_length_sum", "reward_sum")


def rollout_metrics(rewards, dones):
    def finish_episode(carry, transition):
        reward, done = transition
        total_return, length = carry
        total_return, length = total_return + reward, length + 1
        return (jnp.where(done, 0, total_return), jnp.where(done, 0, length)), (
            jnp.where(done, total_return, 0), jnp.where(done, length, 0),
        )

    initial = (jnp.zeros_like(rewards[0]), jnp.zeros(dones.shape[1], jnp.int32))
    _, (returns, lengths) = jax.lax.scan(finish_episode, initial, (rewards, dones))
    return jnp.stack((dones.sum(), (returns > 0).sum(), returns.sum(), lengths.sum(), rewards.sum()))


def _write_rows(path, fields, rows, mode):
    with path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def prepare_curve(path, checkpoint_batch):
    """Discard any rows recorded after the restored checkpoint."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            return
        fields = reader.fieldnames
        rows = [
            row for row in reader
            if (row.get("rollout_batches") or "").isdecimal() and int(row["rollout_batches"]) <= checkpoint_batch
        ]
    _write_rows(path, fields, rows, "w")


def score_summary(scores, score_type):
    usable = scores[np.isfinite(scores)] if score_type else np.empty(0)
    prefix = "pace_score" if score_type == "pace" else "level_score"
    return {
        f"{prefix}_mean": float(usable.mean()) if usable.size else None,
        f"{prefix}_max": float(usable.max()) if usable.size else None,
        **({"score_type": score_type, "valid_level_scores": int(usable.size)} if score_type != "pace" else {}),
    }


def append_curve(path, seed, current, ppo_batches, counts, scores, metrics, *, method, score_type, final_counts,
                 experiment=None):
    trained, ppo_metrics, scoring_metrics, *branches = jax.device_get(metrics)
    ppo_batches -= int(trained.sum())
    branch_ids = trained.astype(int) if method == "pace" else branches[0]
    replay = final_counts.get("replay_batches", 0) - int((branch_ids == 1).sum())
    mutations = final_counts.get("mutation_batches", 0) - int((branch_ids == 2).sum())
    rows = []
    for batch, (updated, ppo, scoring, level_scores, branch) in enumerate(
        zip(trained, ppo_metrics, scoring_metrics, scores, branch_ids), start=current - len(scores) + 1,
    ):
        ppo_batches += int(updated)
        replay += int(branch == 1)
        mutations += int(branch == 2)
        row = {
            "seed": seed, "rollout_batches": batch, "ppo_batches": ppo_batches,
            **counts(batch, ppo_batches, replay=replay, mutations=mutations), "replay": int(branch == 1),
            **score_summary(level_scores, score_type),
        }
        if method != "pace" or experiment is not None:
            row.update(method=experiment or method, branch=("new", "replay", "mutation")[branch],
                       ppo_updated=int(updated))
        for phase, stats in (("ppo", ppo), ("scoring", scoring)):
            values = dict(zip(ROLLOUT_FIELDS, stats.tolist()))
            row.update({f"{phase}_{name}": value for name, value in values.items()})
            row[f"{phase}_mean_episode_return"] = (
                values["return_sum"] / values["episodes"] if values["episodes"] else None
            )
        rows.append(row)
    if rows:
        _write_rows(path, rows[0].keys(), rows, "a")
