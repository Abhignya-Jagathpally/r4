"""Transcriptomic feature engineering."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# MM-relevant Hallmark pathway gene sets (expanded from MSigDB Hallmark collection)
# Shared canonical source — biomarker_discovery.py imports from here.
HALLMARK_GENE_SETS: Dict[str, List[str]] = {
    "MYC_TARGETS_V1": [
        "MYC", "NPM1", "NCL", "RPS3", "RPL3", "EIF4A1", "LDHA", "PKM", "ENO1",
        "NOP56", "NOP58", "BYSL", "MRTO4", "RRP12", "IMP4", "SRM", "UTP20",
        "HSPD1", "PHB", "PHB2", "CCT2", "CCT3", "CCT5", "CCT7", "CCT8",
        "PA2G4", "EIF2S1", "EIF2S2", "EIF3B", "EIF3D", "EIF4G1", "AIMP2",
        "DDX18", "DDX21", "EXOSC5", "FARSA", "GNL3", "IARS1", "IPO4",
        "MPHOSPH10", "MRPS18B", "NOLC1", "POLR1B", "POLR2E", "RAN",
        "RCL1", "RPL22", "RPL6", "RPS2", "RPS5", "RPS6", "SNRPD1",
    ],
    "MTORC1_SIGNALING": [
        "MTOR", "RPS6KB1", "EIF4EBP1", "SLC7A5", "SLC3A2", "VEGFA", "HIF1A",
        "RPTOR", "MLST8", "AKT1", "RHEB", "TSC1", "TSC2", "DDIT4",
        "RPS6", "EIF4E", "EIF4G1", "SCD", "ACLY", "FASN", "HMGCR",
        "SLC1A5", "SLC38A2", "PSMA1", "PSMD1", "SQSTM1", "LAMP1",
        "ATF4", "MTHFD2", "PSAT1", "PHGDH", "GOT1", "CDK4", "CCND1",
    ],
    "IL6_JAK_STAT3_SIGNALING": [
        "IL6", "IL6R", "JAK1", "JAK2", "STAT3", "SOCS3", "BCL2L1", "MCL1",
        "IL6ST", "TYK2", "STAT1", "IL10", "IL10RA", "IL10RB", "IL21",
        "IL21R", "OSMR", "LIFR", "LIF", "CNTF", "CNTFR", "CSF3R",
        "PIM1", "JUNB", "IRF1", "MYC", "CCND1", "VEGFA", "HIF1A",
        "IL2RA", "CXCL9", "CCL2", "IL4R", "IL13RA1", "TNFRSF1B",
        "CD44", "CD38", "FAS", "IFNGR1", "IFNGR2", "CSF2RB",
    ],
    "INTERFERON_GAMMA_RESPONSE": [
        "IFNG", "IRF1", "STAT1", "GBP1", "IDO1", "CXCL10", "HLA-A",
        "STAT2", "IRF2", "IRF7", "IRF9", "JAK2", "IFIT1", "IFIT2",
        "IFIT3", "IFITM1", "MX1", "OAS1", "OAS2", "ISG15", "ISG20",
        "CXCL9", "CXCL11", "CCL5", "GBP2", "GBP4", "TAP1", "TAP2",
        "PSMB8", "PSMB9", "HLA-B", "HLA-C", "HLA-E", "B2M",
        "CIITA", "CD74", "ICAM1", "WARS1", "TRIM14", "TRIM21",
    ],
    "UNFOLDED_PROTEIN_RESPONSE": [
        "XBP1", "ATF4", "ATF6", "DDIT3", "HSPA5", "CALR", "PDIA4",
        "ERN1", "EIF2AK3", "EIF2S1", "DNAJB9", "DNAJC3", "HERPUD1",
        "SEC61A1", "SEC61B", "SSR1", "CANX", "PDIA3", "PDIA6",
        "ERO1A", "HYOU1", "MANF", "SDF2L1", "VIMP", "DERL1",
        "OS9", "SEL1L", "EDEM1", "UGGT1", "PPP1R15A", "TRIB3",
        "GADD45A", "BBC3", "SERP1", "STC2", "WIPI1", "CEBPB",
    ],
    "HYPOXIA": [
        "HIF1A", "VEGFA", "LDHA", "PGK1", "ENO1", "SLC2A1", "BNIP3", "CA9",
        "EPAS1", "EGLN1", "EGLN3", "VHL", "ARNT", "EDN1", "HMOX1",
        "ADM", "ALDOA", "ALDOC", "GPI", "HK2", "PFKFB3", "PKM",
        "PDK1", "MXI1", "SERPINE1", "CITED2", "BHLHE40", "NDRG1",
        "P4HA1", "P4HA2", "LOX", "PLOD2", "MIF", "TPI1", "GAPDH",
    ],
    "P53_PATHWAY": [
        "TP53", "CDKN1A", "MDM2", "BAX", "BBC3", "PMAIP1", "GADD45A", "FAS",
        "MDM4", "CDKN2A", "RB1", "ATM", "ATR", "CHEK1",
        "CHEK2", "TP53BP1", "TP53BP2", "PERP", "SESN1", "SESN2",
        "TIGAR", "SCO2", "DRAM1", "ZMAT3", "DDB2", "XPC", "POLK",
        "RRM2B", "FDXR", "GLS2", "STEAP3", "PIDD1", "LRDD",
        "APAF1", "CASP9", "CASP3", "PTEN", "TSC2", "RPRM",
    ],
    "TNFA_SIGNALING_VIA_NFKB": [
        "NFKB1", "RELA", "TNF", "TNFAIP3", "NFKBIA", "CXCL8", "IL1B",
        "NFKB2", "RELB", "REL", "IKBKB", "IKBKG", "CHUK", "TRAF2",
        "TRAF3", "TRAF6", "RIPK1", "BIRC2", "BIRC3", "XIAP",
        "BCL2A1", "BCL2L1", "CFLAR", "CCL5", "CXCL1", "CXCL2",
        "IL6", "ICAM1", "VCAM1", "SELE", "MMP9", "PTGS2",
        "CSF2", "CSF3", "LTA", "LTB", "CD40", "CD80", "CD86",
    ],
    "PI3K_AKT_MTOR_SIGNALING": [
        "PIK3CA", "AKT1", "MTOR", "PTEN", "TSC1", "TSC2", "RHEB",
        "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2", "AKT2", "AKT3",
        "PDK1", "RPTOR", "RICTOR", "MLST8", "RPS6KB1", "EIF4EBP1",
        "GSK3B", "FOXO1", "FOXO3", "BAD", "CASP9", "CDKN1B",
        "RPS6", "EIF4E", "EIF4G1", "ULK1", "TFEB", "PPARG",
        "INSR", "IGF1R", "ERBB2", "ERBB3", "EGFR", "VEGFA",
    ],
    "OXIDATIVE_PHOSPHORYLATION": [
        "NDUFA1", "SDHA", "UQCRC1", "COX5A", "ATP5F1A", "CYCS",
        "NDUFA2", "NDUFA4", "NDUFB3", "NDUFB5", "NDUFB8", "NDUFS1",
        "NDUFS2", "NDUFS3", "NDUFS7", "NDUFS8", "SDHB", "SDHC",
        "UQCRB", "UQCRFS1", "UQCRC2", "COX4I1", "COX5B", "COX6A1",
        "COX6B1", "COX7A2", "COX7B", "ATP5F1B", "ATP5F1C", "ATP5PB",
        "ATP5MC1", "ATP5PD", "ATP5PF", "LHPP", "IDH3A", "ACO2",
    ],
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
