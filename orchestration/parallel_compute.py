"""Parallel compute with seed propagation (M82, M89)."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ParallelCompute:
    """Distributed computation with reproducibility."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def ray_map(self, fn: Callable, items: List, n_workers: int = 4) -> List:
        """Map with Ray, propagating seeds to workers (M82)."""
        try:
            import ray
            if not ray.is_initialized():
                ray.init(num_cpus=n_workers, runtime_env={
                    "env_vars": {
                        "PYTHONHASHSEED": str(self.seed),  # M82
                        "NUMBA_DISABLE_JIT": "1",
                    }
                })

            @ray.remote
            def _worker(item, worker_seed):
                import numpy as np
                np.random.seed(worker_seed)
                return fn(item)

            futures = [_worker.remote(item, self.seed + i) for i, item in enumerate(items)]
            return ray.get(futures)
        except ImportError:
            logger.warning("Ray not available, falling back to sequential")
            return [fn(item) for item in items]

    def dask_map(self, fn: Callable, items: List, n_workers: int = 4) -> List:
        """Map with Dask using processes scheduler (M89)."""
        try:
            import dask
            delayed_items = [dask.delayed(fn)(item) for item in items]
            # M89: scheduler='processes' to avoid GIL
            results = dask.compute(*delayed_items, scheduler="processes", num_workers=n_workers)
            return list(results)
        except ImportError:
            logger.warning("Dask not available, falling back to sequential")
            return [fn(item) for item in items]
