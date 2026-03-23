"""MLflow configuration (M87: schema enforcement, M88: timestamped artifacts)."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REQUIRED_METRICS = ["primary_metric", "loss"]  # M87


class MLflowManager:
    """MLflow with metric schema enforcement and timestamped artifacts."""

    def __init__(self, experiment_name: str = "mm_pipeline", tracking_uri: str = "mlruns"):
        self._mlflow = None
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
        except ImportError:
            logger.info("MLflow not available")

    def log_run(self, metrics: Dict, params: Dict, artifacts: Optional[Dict[str, str]] = None) -> None:
        """Log with schema enforcement (M87) and timestamped artifacts (M88)."""
        # M87: Enforce required metric schema
        missing = [m for m in REQUIRED_METRICS if m not in metrics]
        if missing:
            logger.warning(f"Missing required metrics: {missing}. Logging anyway with NaN.")
            for m in missing:
                metrics[m] = float("nan")

        if self._mlflow is None:
            logger.info(f"[no-mlflow] {metrics}")
            return

        with self._mlflow.start_run():
            for k, v in params.items():
                self._mlflow.log_param(k, v)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._mlflow.log_metric(k, v)

            # M88: Timestamped artifact paths
            if artifacts:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                for name, path in artifacts.items():
                    if Path(path).exists():
                        timestamped = Path(path).parent / f"{ts}_{Path(path).name}"
                        import shutil
                        shutil.copy2(path, timestamped)
                        self._mlflow.log_artifact(str(timestamped))
