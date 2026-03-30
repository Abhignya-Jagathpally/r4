"""Transcriptomic feature engineering."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# MM-relevant Hallmark pathway gene sets (representative genes)
HALLMARK_GENE_SETS: Dict[str, List[str]] = {
    "MYC_TARGETS_V1": ["MYC", "NPM1", "NCL", "RPS3", "RPL3", "EIF4A1", "LDHA", "PKM", "ENO1"],
    "MTORC1_SIGNALING": ["MTOR", "RPS6KB1", "EIF4EBP1", "SLC7A5", "SLC3A2", "VEGFA", "HIF1A"],
    "IL6_JAK_STAT3_SIGNALING": ["IL6", "IL6R", "JAK1", "JAK2", "STAT3", "SOCS3", "BCL2L1", "MCL1"],
    "INTERFERON_GAMMA_RESPONSE": ["IFNG", "IRF1", "STAT1", "GBP1", "IDO1", "CXCL10", "HLA-A"],
    "UNFOLDED_PROTEIN_RESPONSE": ["XBP1", "ATF4", "ATF6", "DDIT3", "HSPA5", "CALR", "PDIA4"],
    "HYPOXIA": ["HIF1A", "VEGFA", "LDHA", "PGK1", "ENO1", "SLC2A1", "BNIP3", "CA9"],
    "P53_PATHWAY": ["TP53", "CDKN1A", "MDM2", "BAX", "BBC3", "PMAIP1", "GADD45A", "FAS"],
    "TNFA_SIGNALING_VIA_NFKB": ["NFKB1", "RELA", "TNF", "TNFAIP3", "NFKBIA", "CXCL8", "IL1B"],
    "PI3K_AKT_MTOR_SIGNALING": ["PIK3CA", "AKT1", "MTOR", "PTEN", "TSC1", "TSC2", "RHEB"],
    "OXIDATIVE_PHOSPHORYLATION": ["NDUFA1", "SDHA", "UQCRC1", "COX5A", "ATP5F1A", "CYCS"],
}


class TranscriptomicFeatures:
    """Build transcriptomic features from gene expression data."""

    def variance_filter(
        self, X: pd.DataFrame, threshold: float = 0.01,
        train_idx: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Remove near-zero variance genes. Variance computed on train set only."""
        if train_idx is not None:
            variances = X.iloc[train_idx].var(axis=0)
        else:
            variances = X.var(axis=0)
        mask = variances > threshold
        filtered = X.loc[:, mask]
        logger.info(f"Variance filter: {X.shape[1]} -> {filtered.shape[1]} genes (threshold={threshold})")
        return filtered

    def top_variable_genes(
        self, X: pd.DataFrame, n: int = 2000,
        train_idx: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Select top-n most variable genes. Variance computed on train set only."""
        if train_idx is not None:
            variances = X.iloc[train_idx].var(axis=0).sort_values(ascending=False)
        else:
            variances = X.var(axis=0).sort_values(ascending=False)
        top_genes = variances.head(n).index
        result = X[top_genes]
        logger.info(f"Selected top {n} variable genes")
        return result

    def pathway_scoring(
        self, X: pd.DataFrame, gene_sets: Optional[Dict[str, List[str]]] = None,
        train_idx: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Score samples against gene sets using mean z-score approach.

        Scaler is fit on training data only to prevent test leakage.
        """
        if gene_sets is None:
            gene_sets = HALLMARK_GENE_SETS

        scaler = StandardScaler()
        if train_idx is not None:
            scaler.fit(X.iloc[train_idx])
        else:
            scaler.fit(X)
        X_scaled = pd.DataFrame(
            scaler.transform(X), index=X.index, columns=X.columns,
        )

        scores = {}
        for pathway, genes in gene_sets.items():
            overlap = [g for g in genes if g in X_scaled.columns]
            if len(overlap) >= 2:
                scores[pathway] = X_scaled[overlap].mean(axis=1)
            else:
                logger.debug(f"Skipping {pathway}: only {len(overlap)} genes overlap")

        if not scores:
            logger.warning("No pathway scores computed (insufficient gene overlap)")
            return pd.DataFrame(index=X.index)

        result = pd.DataFrame(scores, index=X.index)
        logger.info(f"Computed {result.shape[1]} pathway scores")
        return result

    def pca_embedding(
        self, X: pd.DataFrame, n_components: int = 50,
        train_idx: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, PCA]:
        """Compute PCA embedding. Fit on training data only to prevent leakage."""
        n_components = min(n_components, X.shape[0], X.shape[1])
        scaler = StandardScaler()

        if train_idx is not None:
            scaler.fit(X.iloc[train_idx])
        else:
            scaler.fit(X)
        X_scaled = scaler.transform(X)

        pca = PCA(n_components=n_components, random_state=42)
        if train_idx is not None:
            pca.fit(X_scaled[train_idx])
        else:
            pca.fit(X_scaled)
        embeddings = pca.transform(X_scaled)

        var_explained = pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA: {X.shape[1]} features -> {n_components} components "
            f"({var_explained:.1%} variance explained)"
        )
        return embeddings, pca
