"""Benchmark leaderboard tracking."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class Leaderboard:
    """Track benchmark results across runs."""

    def __init__(self, path: str = "results/leaderboard.json"):
        self.path = Path(path)
        self.entries: List[Dict] = []
        if self.path.exists():
            with open(self.path) as f:
                self.entries = json.load(f)

    def add(self, model_name: str, metrics: Dict, metadata: Dict = None) -> None:
        self.entries.append({
            "model": model_name, "metrics": metrics,
            "metadata": metadata or {}, "timestamp": datetime.now().isoformat(),
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def get_top(self, metric: str = "pearson_r", n: int = 10) -> List[Dict]:
        return sorted(self.entries, key=lambda e: e["metrics"].get(metric, 0), reverse=True)[:n]
