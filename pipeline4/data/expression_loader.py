"""Expression data loader for R3 pseudobulk and GEO expression matrices."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExpressionLoader:
    """Load and validate expression data from various sources."""

    def load_r3_pseudobulk(self, path: str) -> pd.DataFrame:
        """Load pseudobulk h5ad from R3 pipeline output."""
        import anndata
        adata = anndata.read_h5ad(path)
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
        logger.info(f"Loaded R3 pseudobulk: {df.shape}")
        return df

    def load_parquet(self, path: str) -> pd.DataFrame:
        """Load expression matrix from parquet."""
        df = pd.read_parquet(path)
        logger.info(f"Loaded expression parquet: {df.shape}")
        return df

    def validate_expression(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate expression matrix for common issues."""
        issues = []
        if df.isna().any().any():
            n_na = df.isna().sum().sum()
            issues.append(f"{n_na} NaN values found")
        if (df < 0).any().any():
            issues.append("Negative values found in expression matrix")
        if df.shape[1] < 100:
            issues.append(f"Only {df.shape[1]} genes (expected >= 100)")
        if df.shape[0] < 10:
            issues.append(f"Only {df.shape[0]} patients (expected >= 10)")
        valid = len(issues) == 0
        if valid:
            logger.info(f"Expression validation passed: {df.shape}")
        else:
            for issue in issues:
                logger.warning(f"Expression validation: {issue}")
        return valid, issues

    def align_patients(
        self, expression: pd.DataFrame, clinical: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align patient IDs between expression and clinical data."""
        common = expression.index.intersection(clinical.index)
        if len(common) == 0:
            raise ValueError(
                "No overlapping patient IDs between expression and clinical data. "
                f"Expression IDs (first 5): {list(expression.index[:5])}. "
                f"Clinical IDs (first 5): {list(clinical.index[:5])}. "
                "Check that ID formats match across data sources."
            )

        if len(common) < len(expression) * 0.5:
            logger.warning(
                f"Only {len(common)}/{len(expression)} patients overlap — "
                f"check for ID format mismatches"
            )

        logger.info(
            f"Aligned {len(common)} patients "
            f"(expression={len(expression)}, clinical={len(clinical)})"
        )
        return expression.loc[common], clinical.loc[common]

    def generate_synthetic_expression(
        self, patient_ids: pd.Index, n_genes: int = 2000, seed: int = 42
    ) -> pd.DataFrame:
        """Generate realistic synthetic gene expression for demo/testing."""
        rng = np.random.RandomState(seed)
        n_patients = len(patient_ids)

        # Simulate log-normalized counts with gene-specific means
        gene_means = rng.lognormal(2.0, 1.5, n_genes)
        gene_stds = gene_means * rng.uniform(0.3, 0.8, n_genes)

        X = np.zeros((n_patients, n_genes))
        for g in range(n_genes):
            X[:, g] = rng.normal(gene_means[g], gene_stds[g], n_patients)
        X = np.clip(X, 0, None)

        gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
        # Insert known MM marker genes
        mm_genes = [
            "MYC", "CCND1", "FGFR3", "MAF", "MAFB", "TP53", "KRAS",
            "NRAS", "BRAF", "DIS3", "FAM46C", "TRAF3", "IRF4", "XBP1",
            "PRDM1", "SDC1", "CD38", "CD138", "TNFRSF17", "BCMA",
        ]
        for i, gene in enumerate(mm_genes[:min(len(mm_genes), n_genes)]):
            gene_names[i] = gene

        df = pd.DataFrame(X, index=patient_ids, columns=gene_names)
        logger.info(f"Generated synthetic expression: {df.shape}")
        return df
