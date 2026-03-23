"""MLflow experiment tracking with graceful fallback."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow wrapper with fallback to file-based logging."""

    def __init__(self, experiment_name: str = "r4_clinical", tracking_uri: str = "mlruns"):
        self._available = False
        self._run = None
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            self._available = True
            logger.info(f"MLflow tracking: {tracking_uri}/{experiment_name}")
        except ImportError:
            logger.info("MLflow not available, using file-based logging")

    def start_run(self, run_name: str, tags: Optional[Dict] = None):
        """Start an MLflow run (context manager)."""
        if self._available:
            return self._mlflow.start_run(run_name=run_name, tags=tags)
        return _NoopContext()

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._available:
            for k, v in params.items():
                self._mlflow.log_param(k, v)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._available:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._mlflow.log_metric(k, v, step=step)

    def log_artifact(self, path: str) -> None:
        if self._available:
            self._mlflow.log_artifact(path)


class _NoopContext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
