"""Benchmark runner (M60: real run() method)."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def run(config: Dict = None) -> Dict:
    """Run full DepMap benchmark (M60)."""
    from .depmap_loader import DepMapLoader
    from .benchmark_definition import BenchmarkSuite
    from .cell_line_profiles import CellLineProfiler
    from .drug_target_validation import DrugTargetValidator
    from .mm_pathway_oracle import MMPathwayOracle

    config = config or {}
    loader = DepMapLoader(release=config.get("release", "24Q4"))
    suite = BenchmarkSuite()
    profiler = CellLineProfiler()
    validator = DrugTargetValidator()
    oracle = MMPathwayOracle()

    # Load data
    data = loader.download()
    mm_data = {k: loader.filter_to_mm_lines(v) for k, v in data.items()}

    # Build profiles
    profiles = {}
    for name, df in mm_data.items():
        profiles[name] = profiler.build_profile(df)

    # Run benchmark
    results = {"n_cell_lines": sum(len(df) for df in mm_data.values())}

    # Pathway recovery
    all_genes = []
    for df in mm_data.values():
        all_genes.extend(df.columns[:100].tolist())
    results["pathway_recovery"] = oracle.full_recovery_report(all_genes)

    logger.info(f"Benchmark complete: {results['n_cell_lines']} cell lines evaluated")
    return results
