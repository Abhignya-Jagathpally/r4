"""Stage 1: Data ingestion — load expression and clinical data."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_ingest(config: Any, context: Dict) -> Dict:
    """Load expression + clinical data, align patients, save merged dataset."""
    from pipeline4.data.geo_loader import GEOClinicalLoader
    from pipeline4.data.expression_loader import ExpressionLoader

    data_dir = Path(config.base.data_dir)
    merged_dir = data_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    loader = GEOClinicalLoader(cache_dir=str(data_dir / "raw"))
    expr_loader = ExpressionLoader()

    # Load clinical data
    if config.ingest.demo_mode:
        logger.info("Demo mode: generating synthetic clinical + expression data")
        clinical = loader.generate_synthetic_clinical(
            n_patients=config.ingest.demo_n_patients, seed=config.base.seed,
        )
        expression = expr_loader.generate_synthetic_expression(
            patient_ids=clinical.index,
            n_genes=config.features.n_top_genes,
            seed=config.base.seed,
        )
    else:
        # Try loading from R3 output
        expr_path = Path(config.ingest.expression_path)
        if expr_path.exists():
            if str(expr_path).endswith(".h5ad"):
                expression = expr_loader.load_r3_pseudobulk(str(expr_path))
            else:
                expression = expr_loader.load_parquet(str(expr_path))
        else:
            raise FileNotFoundError(
                f"Expression data not found at {expr_path}. "
                f"Provide real data or set demo_mode=true for synthetic."
            )

        # Load clinical from GEO
        clinical_parts = []
        for acc in config.ingest.geo_accessions:
            df = loader.fetch_clinical(acc)
            if not df.empty:
                clinical_parts.append(df)
        if clinical_parts:
            clinical = pd.concat(clinical_parts)
        else:
            raise RuntimeError(
                f"No clinical data fetched from GEO accessions {config.ingest.geo_accessions}. "
                f"Provide real data or set demo_mode=true for synthetic."
            )

    # Align patients
    expression, clinical = expr_loader.align_patients(expression, clinical)

    # Validate
    valid, issues = expr_loader.validate_expression(expression)
    if not valid:
        logger.warning(f"Expression issues: {issues}")

    # Save
    expression.to_parquet(merged_dir / "expression.parquet")
    clinical.to_parquet(merged_dir / "clinical.parquet")

    context["expression_path"] = str(merged_dir / "expression.parquet")
    context["clinical_path"] = str(merged_dir / "clinical.parquet")
    context["n_patients"] = len(expression)
    context["n_genes"] = expression.shape[1]

    logger.info(
        f"Ingest complete: {context['n_patients']} patients, {context['n_genes']} genes, "
        f"event rate={clinical['event'].mean():.2f}"
    )
    return context
