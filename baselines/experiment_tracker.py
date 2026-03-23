"""MLflow experiment tracker (M31: log ALL metrics, HPs, artifacts)."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track experiments with full MLflow logging (M31)."""

    def __init__(self, experiment_name: str = "mm_baselines", tracking_uri: str = "mlruns"):
        self._mlflow = None
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
        except ImportError:
            logger.info("MLflow not available")

    def log_run(
        self, metrics: Dict[str, float], params: Dict[str, Any],
        artifacts: Optional[Dict[str, str]] = None, run_name: Optional[str] = None,
    ) -> None:
        """Log ALL metrics, HPs, and artifacts to MLflow (M31)."""
        if self._mlflow is None:
            logger.info(f"[no-mlflow] metrics={metrics}, params={params}")
            return

        with self._mlflow.start_run(run_name=run_name or f"run_{datetime.now().strftime('%H%M%S')}"):
            # Log ALL params (M31)
            for k, v in params.items():
                self._mlflow.log_param(k, v)

            # Log ALL metrics (M31)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._mlflow.log_metric(k, v)

            # Log ALL artifacts (M31)
            if artifacts:
                for name, path in artifacts.items():
                    if Path(path).exists():
                        self._mlflow.log_artifact(path)

        logger.info(f"Logged run: {len(metrics)} metrics, {len(params)} params")
