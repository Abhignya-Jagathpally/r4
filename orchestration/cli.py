"""CLI entry point for pipeline orchestration."""

import argparse
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MM Proteomics Pipeline Orchestration")
    parser.add_argument("--mode", choices=["full", "tune", "benchmark"], default="full")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from orchestration.reproducibility import ReproducibilityManager
    rm = ReproducibilityManager()
    rm.set_seeds(args.seed)
    logger.info(f"Pipeline orchestration: mode={args.mode}")
