"""HPO with trial-specific val splits (M41: no shared val leakage)."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class HPOptimizer:
    """Optuna-based HPO with trial-specific validation splits (M41)."""

    def __init__(self, config: Dict):
        self.config = config

    def search(self, X: np.ndarray, y: np.ndarray, n_trials: int = 30) -> Dict:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not available")
            return {"best_params": {}}

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            hidden = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            n_layers = trial.suggest_int("n_layers", 2, 5)

            # M41: Trial-specific val split (NOT shared)
            rng = np.random.RandomState(trial.number)
            perm = rng.permutation(len(X))
            split = int(0.8 * len(X))
            train_idx, val_idx = perm[:split], perm[split:]

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Simple proxy: linear model for speed
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0 / (lr * 100))
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            return -score  # minimize

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=self.config.get("timeout", 600))

        return {"best_params": study.best_params, "best_value": study.best_value}
