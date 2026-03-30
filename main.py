#!/usr/bin/env python3
"""
R4-MM-Clinical: Multi-Modal Clinical Outcome Prediction for Multiple Myeloma.

Entry point for the end-to-end pipeline. Orchestrates 8 stages:
  ingest -> cohort -> features -> train -> evaluate -> interpret -> report -> autotune

Usage:
    python main.py --stages all
    python main.py --stages ingest features cohort train evaluate
    python main.py --stages train --resume checkpoints/deepsurv/run_001/checkpoint_epoch_0050.pt
    python main.py --dry-run
"""

import argparse
import logging
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

PIPELINE_STAGES = [
    "ingest", "cohort", "features", "train",
    "evaluate", "interpret", "report", "autotune",
]


def _get_stage_map() -> OrderedDict:
    """Lazy-import stage runners to avoid heavy imports on --help."""
    from pipeline4.stages import (
        run_ingest, run_features, run_cohort, run_train,
        run_evaluate, run_interpret, run_report, run_autotune,
    )
    return OrderedDict([
        ("ingest", run_ingest),
        ("cohort", run_cohort),
        ("features", run_features),
        ("train", run_train),
        ("evaluate", run_evaluate),
        ("interpret", run_interpret),
        ("report", run_report),
        ("autotune", run_autotune),
    ])


def main() -> None:
    """Main entry point for the R4 clinical outcome prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="R4-MM-Clinical: Multi-Modal Clinical Outcome Prediction for Multiple Myeloma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config-dir", default="configs",
        help="Directory containing JSON config files (default: configs)",
    )
    parser.add_argument(
        "--stages", nargs="+", default=["all"],
        choices=PIPELINE_STAGES + ["all"],
        help="Pipeline stages to run (default: all)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint path to resume training from",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override compute device (cpu, cuda)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show planned stages without executing",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )

    args = parser.parse_args()

    # Load configuration
    from pipeline4.config import PipelineConfig
    config = PipelineConfig.from_dir(args.config_dir)

    # Apply CLI overrides
    if args.device:
        config.base.device = args.device
    if args.seed is not None:
        config.base.seed = args.seed

    # Setup logging and reproducibility
    from pipeline4.utils import setup_logging, set_all_seeds
    setup_logging(config.base)
    set_all_seeds(config.base.seed)

    logger.info("=" * 80)
    logger.info(f"R4-MM-Clinical Pipeline v{config.base.pipeline_version}")
    logger.info(f"Config: {args.config_dir}")
    logger.info(f"Device: {config.base.device}")
    logger.info(f"Seed: {config.base.seed}")
    logger.info("=" * 80)

    # Resolve stages
    if "all" in args.stages:
        stages_to_run = list(PIPELINE_STAGES)
    else:
        stages_to_run = [s for s in PIPELINE_STAGES if s in args.stages]

    logger.info(f"Stages to run: {stages_to_run}")

    if args.dry_run:
        logger.info("[DRY RUN] Would execute stages:")
        for s in stages_to_run:
            logger.info(f"  -> {s}")
        return

    # Build stage dispatcher
    stage_map = _get_stage_map()

    # Shared context passed between stages
    context: Dict[str, Any] = {
        "resume_path": args.resume,
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timings": {},
    }

    # Ensure output directories exist
    Path(config.base.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.base.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(config.base.data_dir).mkdir(parents=True, exist_ok=True)

    # Execute stages
    pipeline_start = time.time()
    failed = False

    for stage_name in stages_to_run:
        runner = stage_map[stage_name]
        logger.info("-" * 60)
        logger.info(f"STAGE: {stage_name.upper()}")
        logger.info("-" * 60)

        t0 = time.time()
        try:
            context = runner(config, context)
            elapsed = time.time() - t0
            context["timings"][stage_name] = elapsed
            logger.info(f"Stage {stage_name} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            context["timings"][stage_name] = elapsed
            logger.error(f"Stage {stage_name} FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            failed = True
            break

    # Summary
    total_time = time.time() - pipeline_start
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    for stage, elapsed in context["timings"].items():
        status = "OK" if not (failed and stage == stages_to_run[-1]) else "FAILED"
        logger.info(f"  {stage:20s} {elapsed:8.1f}s  [{status}]")
    logger.info(f"  {'TOTAL':20s} {total_time:8.1f}s")

    if failed:
        logger.error("Pipeline finished with errors.")
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully.")

    # Save run summary
    from pipeline4.utils.io import write_json
    write_json(
        {
            "run_id": context["run_id"],
            "stages": stages_to_run,
            "timings": context["timings"],
            "total_seconds": total_time,
            "status": "failed" if failed else "success",
            "timestamp": datetime.now().isoformat(),
        },
        str(Path(config.base.results_dir) / "pipeline_summary.json"),
    )


if __name__ == "__main__":
    main()
