"""Versioned checkpoints for a complete MiniGrid rollout boundary."""

import os
import tempfile
from pathlib import Path

import jax
import numpy as np
from flax import serialization


CHECKPOINT_VERSION = 2


def checkpoint_path(output_dir: Path, batches: int) -> Path:
    return output_dir / "checkpoints" / f"step_{batches:08d}.msgpack"


def save_checkpoint(path: Path, config: dict, state) -> None:
    """Publish the checkpoint only after its payload has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    payload = serialization.msgpack_serialize({
        "version": CHECKPOINT_VERSION,
        "config": config,
        "runner": jax.device_get(serialization.to_state_dict(state)),
    })
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as fh:
        temporary = Path(fh.name)
        try:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        except BaseException:
            fh.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path: Path) -> dict:
    payload = serialization.msgpack_restore(path.read_bytes())
    if payload.get("version") == 1:
        # Old checkpoints shared the pre-PPO rollout and filled during the main loop.
        payload["config"]["protocol"] = "legacy"
        payload["runner"]["prefill_env_steps"] = np.int32(0)
        payload["version"] = CHECKPOINT_VERSION
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {path}")
    return payload


def restore_state(payload: dict, template):
    """Restore arrays into a trusted JAX/Flax structure, including Optax state."""
    state = serialization.from_state_dict(template, payload["runner"])
    for restored, expected in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(template)):
        if restored.shape != expected.shape or restored.dtype != expected.dtype:
            raise ValueError("Checkpoint array shapes or dtypes differ from the training configuration")
    return jax.device_put(state)
