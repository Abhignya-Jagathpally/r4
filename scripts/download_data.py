#!/usr/bin/env python3
"""Download required datasets for the R4-MM-Clinical pipeline.

This script fetches expression and clinical data from GEO so the pipeline
can run end-to-end without pre-existing data files.

Usage:
    python scripts/download_data.py                  # download all
    python scripts/download_data.py --demo           # generate synthetic data only
    python scripts/download_data.py --accessions GSE24080
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def ensure_dirs():
    """Create data directory structure."""
    for subdir in ["raw", "merged", "features", "splits"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directories ready at {DATA_DIR}")


def download_geo_data(accessions: list[str], cache_dir: Path):
    """Download expression and clinical data from GEO."""
    try:
        import GEOparse
    except ImportError:
        logger.error("GEOparse not installed. Run: pip install GEOparse")
        sys.exit(1)

    for acc in accessions:
        logger.info(f"Downloading {acc} from GEO...")
        try:
            gse = GEOparse.get_GEO(geo=acc, destdir=str(cache_dir), silent=True)
            logger.info(
                f"  {acc}: {len(gse.gsms)} samples, "
                f"{len(gse.phenotype_data)} phenotype records"
            )
        except Exception as e:
            logger.error(f"  Failed to download {acc}: {e}")
            continue

    logger.info("GEO download complete. Data cached in data/raw/")


def generate_demo_data(n_patients: int = 500, seed: int = 42):
    """Generate synthetic demo data for testing the pipeline."""
    sys.path.insert(0, str(PROJECT_ROOT))

    from pipeline4.data.geo_loader import GEOClinicalLoader
    from pipeline4.data.expression_loader import ExpressionLoader

    loader = GEOClinicalLoader(cache_dir=str(DATA_DIR / "raw"))
    expr_loader = ExpressionLoader()

    logger.info(f"Generating synthetic clinical data ({n_patients} patients)...")
    clinical = loader.generate_synthetic_clinical(n_patients, seed)

    logger.info(f"Generating synthetic expression data ({n_patients} patients)...")
    expression = expr_loader.generate_synthetic_expression(
        clinical.index, n_genes=2000, seed=seed,
    )

    merged_dir = DATA_DIR / "merged"
    clinical.to_parquet(merged_dir / "clinical.parquet")
    expression.to_parquet(merged_dir / "expression.parquet")

    logger.info(f"Demo data saved to {merged_dir}/")
    logger.info(f"  clinical: {clinical.shape[0]} patients x {clinical.shape[1]} columns")
    logger.info(f"  expression: {expression.shape[0]} patients x {expression.shape[1]} genes")


def main():
    parser = argparse.ArgumentParser(description="Download data for R4-MM-Clinical pipeline")
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate synthetic demo data instead of downloading real data",
    )
    parser.add_argument(
        "--accessions", nargs="+", default=None,
        help="GEO accession IDs to download (default: from configs/ingest.json)",
    )
    parser.add_argument(
        "--n-patients", type=int, default=500,
        help="Number of synthetic patients for demo mode (default: 500)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ensure_dirs()

    if args.demo:
        generate_demo_data(args.n_patients, args.seed)
    else:
        # Load accessions from config if not specified
        accessions = args.accessions
        if accessions is None:
            config_path = PROJECT_ROOT / "configs" / "ingest.json"
            if config_path.exists():
                with open(config_path) as f:
                    accessions = json.load(f).get("geo_accessions", [])
            else:
                accessions = ["GSE24080", "GSE136337"]

        download_geo_data(accessions, DATA_DIR / "raw")

    logger.info("Done. Run the pipeline with: python main.py --stages all")


if __name__ == "__main__":
    main()
