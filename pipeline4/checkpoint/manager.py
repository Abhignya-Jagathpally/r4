"""Checkpoint manager for model training traceability."""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Save, load, and manage model checkpoints with manifest tracking."""

    def __init__(self, checkpoints_dir: str = "checkpoints", max_keep: int = 5):
        self.root = Path(checkpoints_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep

    def _run_dir(self, model_name: str, run_id: str) -> Path:
        d = self.root / model_name / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _manifest_path(self, model_name: str, run_id: str) -> Path:
        return self._run_dir(model_name, run_id) / "manifest.json"

    def _load_manifest(self, model_name: str, run_id: str) -> Dict:
        mp = self._manifest_path(model_name, run_id)
        if mp.exists():
            with open(mp) as f:
                return json.load(f)
        return {
            "run_id": run_id,
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "config_hash": None,
            "checkpoints": [],
            "best_epoch": None,
            "best_metric": None,
        }

    def _save_manifest(self, model_name: str, run_id: str, manifest: Dict) -> None:
        mp = self._manifest_path(model_name, run_id)
        with open(mp, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    @staticmethod
    def compute_config_hash(config: Dict) -> str:
        """Compute SHA256 hash of config dict for integrity checking."""
        raw = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def save(
        self,
        model: Any,
        model_name: str,
        run_id: str,
        epoch: int,
        metrics: Dict[str, float],
        config_hash: str,
        optimizer: Any = None,
    ) -> str:
        """Save a model checkpoint and update manifest."""
        run_dir = self._run_dir(model_name, run_id)
        ckpt_name = f"checkpoint_epoch_{epoch:04d}.pt"
        ckpt_path = run_dir / ckpt_name

        # Determine save method based on model type
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "config_hash": config_hash,
                    "timestamp": datetime.now().isoformat(),
                }
                if optimizer is not None:
                    state["optimizer_state_dict"] = optimizer.state_dict()
                torch.save(state, ckpt_path)
            else:
                self._save_sklearn(model, ckpt_path, epoch, metrics, config_hash)
        except ImportError:
            self._save_sklearn(model, ckpt_path, epoch, metrics, config_hash)

        # Update manifest
        manifest = self._load_manifest(model_name, run_id)
        manifest["config_hash"] = config_hash
        manifest["checkpoints"].append({
            "epoch": epoch,
            "path": str(ckpt_path),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        })

        # Track best checkpoint (by c_index or first available metric)
        primary = metrics.get("c_index", metrics.get("val_c_index",
                   next(iter(metrics.values()), 0.0)))
        if manifest["best_metric"] is None or primary > manifest["best_metric"]:
            manifest["best_epoch"] = epoch
            manifest["best_metric"] = primary

        self._save_manifest(model_name, run_id, manifest)
        self.prune(model_name, run_id)

        logger.info(f"Saved checkpoint: {model_name}/{run_id} epoch={epoch} metrics={metrics}")
        return str(ckpt_path)

    def _save_sklearn(
        self, model: Any, path: Path, epoch: int,
        metrics: Dict, config_hash: str,
    ) -> None:
        """Save sklearn/lifelines model via joblib."""
        import joblib
        state = {
            "model": model,
            "epoch": epoch,
            "metrics": metrics,
            "config_hash": config_hash,
            "timestamp": datetime.now().isoformat(),
        }
        joblib.dump(state, path)

    def load(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a checkpoint and return its contents."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            import torch
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                return state
        except Exception:
            pass

        import joblib
        return joblib.load(path)

    def get_best(
        self, model_name: str, run_id: str, metric: str = "c_index"
    ) -> Optional[str]:
        """Return path to best checkpoint for a run."""
        manifest = self._load_manifest(model_name, run_id)
        best_epoch = manifest.get("best_epoch")
        if best_epoch is None:
            return None
        for ckpt in manifest["checkpoints"]:
            if ckpt["epoch"] == best_epoch:
                return ckpt["path"]
        return None

    def prune(self, model_name: str, run_id: str) -> int:
        """Keep only top max_keep checkpoints by metric. Returns number removed."""
        manifest = self._load_manifest(model_name, run_id)
        checkpoints = manifest["checkpoints"]
        if len(checkpoints) <= self.max_keep:
            return 0

        # Sort by primary metric descending, keep top max_keep
        def _metric_val(ckpt: Dict) -> float:
            m = ckpt.get("metrics", {})
            return m.get("c_index", m.get("val_c_index", 0.0))

        ranked = sorted(checkpoints, key=_metric_val, reverse=True)
        keep = set(c["epoch"] for c in ranked[: self.max_keep])
        removed = 0

        new_checkpoints = []
        for ckpt in checkpoints:
            if ckpt["epoch"] in keep:
                new_checkpoints.append(ckpt)
            else:
                p = Path(ckpt["path"])
                if p.exists():
                    p.unlink()
                removed += 1

        manifest["checkpoints"] = new_checkpoints
        self._save_manifest(model_name, run_id, manifest)
        if removed:
            logger.info(f"Pruned {removed} checkpoints for {model_name}/{run_id}")
        return removed

    def list_runs(self, model_name: Optional[str] = None) -> List[Dict]:
        """List all runs, optionally filtered by model name."""
        runs = []
        search = [self.root / model_name] if model_name else self.root.iterdir()
        for model_dir in search:
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.iterdir():
                mp = run_dir / "manifest.json"
                if mp.exists():
                    with open(mp) as f:
                        runs.append(json.load(f))
        return runs
