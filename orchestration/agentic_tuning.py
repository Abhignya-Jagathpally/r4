"""Agentic tuning with Karpathy pattern (M72, M85, M86)."""

import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AgenticTuner:
    """Hyperparameter tuning with hill-climbing + random restarts + convergence (M72)."""

    def __init__(self, config: Dict):
        # M85: Budget validation
        budget_seconds = config.get("budget_seconds")
        budget_experiments = config.get("budget_experiments")
        if budget_seconds is None and budget_experiments is None:
            raise ValueError("At least one of budget_seconds or budget_experiments must be set (M85)")

        self.budget_seconds = budget_seconds or float("inf")
        self.budget_experiments = budget_experiments or float("inf")
        self.convergence_window = config.get("convergence_window", 10)
        self.convergence_threshold = config.get("convergence_threshold", 0.001)
        self.n_restarts = config.get("n_restarts", 3)
        self.history: List[Dict] = []

    def tune(self, objective_fn: Callable, param_space: Dict) -> Dict:
        """Full Karpathy pattern: hill-climb + random restarts + convergence (M72)."""
        start_time = time.time()
        best_score = float("-inf")
        best_params = {}
        n_experiments = 0

        for restart in range(self.n_restarts):
            # Random restart point
            current_params = self._random_sample(param_space)
            current_score = objective_fn(current_params)
            n_experiments += 1
            self.history.append({"restart": restart, "params": current_params,
                                "score": current_score, "type": "random"})

            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()

            # Hill climbing from this point
            while True:
                if n_experiments >= self.budget_experiments:
                    logger.info(f"Budget exhausted: {n_experiments} experiments")
                    break
                if time.time() - start_time > self.budget_seconds:
                    logger.info(f"Time budget exhausted: {time.time()-start_time:.0f}s")
                    break

                # M86: Convergence check
                if self._check_convergence():
                    logger.info(f"Converged at restart {restart}, experiment {n_experiments}")
                    break

                # Perturb current params
                neighbor = self._perturb(current_params, param_space)
                neighbor_score = objective_fn(neighbor)
                n_experiments += 1
                self.history.append({"restart": restart, "params": neighbor,
                                    "score": neighbor_score, "type": "hill_climb"})

                if neighbor_score > current_score:
                    current_params = neighbor
                    current_score = neighbor_score
                    if current_score > best_score:
                        best_score = current_score
                        best_params = current_params.copy()

        logger.info(f"Tuning complete: {n_experiments} experiments, best={best_score:.4f}")
        return {"best_params": best_params, "best_score": best_score,
                "n_experiments": n_experiments, "history": self.history}

    def _check_convergence(self) -> bool:
        """M86: Convergence tracking in stopping criteria."""
        if len(self.history) < self.convergence_window:
            return False
        recent = [h["score"] for h in self.history[-self.convergence_window:]]
        improvement = max(recent) - min(recent)
        return improvement < self.convergence_threshold

    def _random_sample(self, space: Dict) -> Dict:
        params = {}
        for k, v in space.items():
            if isinstance(v, list):
                params[k] = np.random.choice(v)
            elif isinstance(v, tuple) and len(v) == 2:
                params[k] = np.random.uniform(v[0], v[1])
            else:
                params[k] = v
        return params

    def _perturb(self, params: Dict, space: Dict) -> Dict:
        result = params.copy()
        key = np.random.choice(list(space.keys()))
        v = space[key]
        if isinstance(v, list):
            result[key] = np.random.choice(v)
        elif isinstance(v, tuple):
            current = result[key]
            delta = (v[1] - v[0]) * 0.1 * np.random.randn()
            result[key] = np.clip(current + delta, v[0], v[1])
        return result
