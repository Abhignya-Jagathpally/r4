"""Stage 7: Report generation — HTML report with figures and tables."""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def run_report(config: Any, context: Dict) -> Dict:
    """Generate HTML report with all results."""
    from pipeline4.reporting.generator import ReportGenerator

    generator = ReportGenerator(config, config.base.results_dir)

    try:
        report_path = generator.generate(context)
        context["report_path"] = report_path
        logger.info(f"Report generated: {report_path}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()

    return context
