"""Biomarker discovery via rank aggregation and pathway enrichment."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# MM-relevant pathways for enrichment (expanded from MSigDB Hallmark sets)
HALLMARK_PATHWAYS = {
    "MYC_TARGETS": [
        "MYC", "NPM1", "NCL", "RPS3", "RPL3", "EIF4A1", "LDHA", "PKM", "ENO1",
        "NOP56", "NOP58", "BYSL", "MRTO4", "RRP12", "IMP4", "SRM", "UTP20",
        "HSPD1", "PHB", "PHB2", "CCT2", "CCT3", "CCT5", "CCT7", "CCT8",
        "PA2G4", "EIF2S1", "EIF2S2", "EIF3B", "EIF3D", "EIF4G1", "AIMP2",
        "DDX18", "DDX21", "EXOSC5", "FARSA", "GNL3", "IARS1", "IPO4",
        "MPHOSPH10", "MRPS18B", "NOLC1", "POLR1B", "POLR2E", "RAN",
        "RCL1", "RPL22", "RPL6", "RPS2", "RPS5", "RPS6", "SNRPD1",
    ],
    "IL6_JAK_STAT3": [
        "IL6", "IL6R", "JAK1", "JAK2", "STAT3", "SOCS3", "MCL1", "BCL2L1",
        "IL6ST", "TYK2", "STAT1", "IL10", "IL10RA", "IL10RB", "IL21",
        "IL21R", "OSMR", "LIFR", "LIF", "CNTF", "CNTFR", "CSF3R",
        "PIM1", "JUNB", "IRF1", "MYC", "CCND1", "VEGFA", "HIF1A",
        "IL2RA", "CXCL9", "CCL2", "IL4R", "IL13RA1", "TNFRSF1B",
        "CD44", "CD38", "FAS", "IFNGR1", "IFNGR2", "CSF2RB",
    ],
    "UNFOLDED_PROTEIN_RESPONSE": [
        "XBP1", "ATF4", "ATF6", "DDIT3", "HSPA5", "CALR", "PDIA4",
        "ERN1", "EIF2AK3", "EIF2S1", "DNAJB9", "DNAJC3", "HERPUD1",
        "SEC61A1", "SEC61B", "SSR1", "CANX", "PDIA3", "PDIA6",
        "ERO1A", "HYOU1", "MANF", "SDF2L1", "VIMP", "DERL1",
        "OS9", "SEL1L", "EDEM1", "UGGT1", "PPP1R15A", "TRIB3",
        "GADD45A", "BBC3", "SERP1", "STC2", "WIPI1", "CEBPB",
    ],
    "P53_PATHWAY": [
        "TP53", "CDKN1A", "MDM2", "BAX", "BBC3", "GADD45A", "FAS",
        "PMAIP1", "MDM4", "CDKN2A", "RB1", "ATM", "ATR", "CHEK1",
        "CHEK2", "TP53BP1", "TP53BP2", "PERP", "SESN1", "SESN2",
        "TIGAR", "SCO2", "DRAM1", "ZMAT3", "DDB2", "XPC", "POLK",
        "RRM2B", "FDXR", "GLS2", "STEAP3", "PIDD1", "LRDD",
        "APAF1", "CASP9", "CASP3", "PTEN", "TSC2", "RPRM",
    ],
    "NFKB_SIGNALING": [
        "NFKB1", "RELA", "TNF", "TNFAIP3", "NFKBIA", "CXCL8", "IL1B",
        "NFKB2", "RELB", "REL", "IKBKB", "IKBKG", "CHUK", "TRAF2",
        "TRAF3", "TRAF6", "RIPK1", "BIRC2", "BIRC3", "XIAP",
        "BCL2A1", "BCL2L1", "CFLAR", "CCL5", "CXCL1", "CXCL2",
        "IL6", "ICAM1", "VCAM1", "SELE", "MMP9", "PTGS2",
        "CSF2", "CSF3", "LTA", "LTB", "CD40", "CD80", "CD86",
    ],
    "PI3K_AKT_MTOR": [
        "PIK3CA", "AKT1", "MTOR", "PTEN", "TSC1", "TSC2", "RHEB",
        "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2", "AKT2", "AKT3",
        "PDK1", "RPTOR", "RICTOR", "MLST8", "RPS6KB1", "EIF4EBP1",
        "GSK3B", "FOXO1", "FOXO3", "BAD", "CASP9", "CDKN1B",
        "RPS6", "EIF4E", "EIF4G1", "ULK1", "TFEB", "PPARG",
        "INSR", "IGF1R", "ERBB2", "ERBB3", "EGFR", "VEGFA",
    ],
    "CELL_CYCLE": [
        "CDK4", "CDK6", "CCND1", "CCND2", "CCNE1", "RB1",
        "CDK2", "CDK1", "CCNA2", "CCNB1", "CCNB2", "CDC25A",
        "CDC25B", "CDC25C", "CDC20", "CDH1", "BUB1", "BUB1B",
        "MAD2L1", "AURKA", "AURKB", "PLK1", "E2F1", "E2F2",
        "E2F3", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B",
        "SKP2", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
        "ORC1", "CDC6", "CDT1", "GMNN", "PCNA", "TOP2A",
    ],
    "APOPTOSIS": [
        "BCL2", "MCL1", "BAX", "BAK1", "CASP3", "CASP9",
        "BCL2L1", "BCL2L11", "BID", "BIM", "BAD", "BBC3", "PMAIP1",
        "APAF1", "CYCS", "CASP7", "CASP8", "CASP10", "FADD",
        "FAS", "FASLG", "TNFRSF10A", "TNFRSF10B", "TRAIL",
        "DIABLO", "XIAP", "BIRC2", "BIRC3", "BIRC5",
        "CFLAR", "RIPK1", "RIPK3", "MLKL", "DFFA", "DFFB",
        "ENDOG", "AIFM1", "HTRA2", "PARP1", "ICAD",
    ],
}


class BiomarkerDiscovery:
    """Aggregate feature rankings and discover biomarkers."""

    def aggregate_rankings(
        self, rankings: Dict[str, pd.Series], top_k: int = 50,
    ) -> pd.DataFrame:
        """Borda count rank aggregation across models."""
        all_features = set()
        for r in rankings.values():
            all_features.update(r.index[:top_k * 2])

        borda_scores = pd.Series(0.0, index=list(all_features))

        for model_name, ranking in rankings.items():
            # Convert to ranks (1 = best)
            sorted_features = ranking.sort_values(ascending=False)
            for rank, feat in enumerate(sorted_features.index):
                if feat in borda_scores.index:
                    borda_scores[feat] += max(0, len(sorted_features) - rank)

        borda_scores = borda_scores.sort_values(ascending=False)

        # Build evidence table
        records = []
        for feat in borda_scores.head(top_k).index:
            record = {"feature": feat, "borda_score": borda_scores[feat]}
            for model_name, ranking in rankings.items():
                if feat in ranking.index:
                    rank = list(ranking.sort_values(ascending=False).index).index(feat) + 1
                    record[f"{model_name}_rank"] = rank
                else:
                    record[f"{model_name}_rank"] = None
            records.append(record)

        df = pd.DataFrame(records).set_index("feature")
        logger.info(f"Aggregated rankings: top {len(df)} biomarkers from {len(rankings)} models")
        return df

    def stability_selection(
        self, X: np.ndarray, T: np.ndarray, E: np.ndarray,
        n_bootstrap: int = 100, threshold: float = 0.6, seed: int = 42,
    ) -> List[str]:
        """Features selected in >threshold fraction of bootstrap samples.

        Uses L1-penalized Cox regression (CoxnetSurvivalAnalysis) to properly
        handle censored survival outcomes instead of Lasso on raw times.
        """
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        rng = np.random.RandomState(seed)
        n, p = X.shape
        selection_counts = np.zeros(p)

        for i in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            y_boot = np.array(
                [(bool(e), t) for e, t in zip(E[idx], T[idx])],
                dtype=[("event", bool), ("time", float)],
            )
            try:
                coxnet = CoxnetSurvivalAnalysis(
                    l1_ratio=1.0, alpha_min_ratio=0.01,
                    max_iter=1000, fit_baseline_model=False,
                )
                coxnet.fit(X[idx], y_boot)
                # Use coefficients from the sparsest model with reasonable fit
                coefs = coxnet.coef_[:, -1]
                selected = np.abs(coefs) > 0
                selection_counts += selected
            except Exception:
                continue

        fractions = selection_counts / max(n_bootstrap, 1)
        stable = np.where(fractions >= threshold)[0].tolist()
        logger.info(f"Stability selection: {len(stable)} features above {threshold} threshold")
        return stable

    def pathway_enrichment(
        self, gene_list: List[str], background: List[str],
        pathway_db: Optional[Dict[str, List[str]]] = None,
        pvalue_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Fisher's exact test for pathway over-representation."""
        if pathway_db is None:
            pathway_db = HALLMARK_PATHWAYS

        gene_set = set(gene_list)
        bg_set = set(background)
        n_bg = len(bg_set)

        results = []
        for pathway, genes in pathway_db.items():
            pathway_set = set(genes) & bg_set
            if not pathway_set:
                continue

            overlap = gene_set & pathway_set
            a = len(overlap)  # in list & in pathway
            b = len(gene_set - pathway_set)  # in list, not in pathway
            c = len(pathway_set - gene_set)  # not in list, in pathway
            d = n_bg - a - b - c  # neither

            _, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            fold_enrichment = (a / max(len(gene_set), 1)) / (len(pathway_set) / max(n_bg, 1))

            results.append({
                "pathway": pathway,
                "overlap_genes": ", ".join(sorted(overlap)),
                "n_overlap": a,
                "n_pathway": len(pathway_set),
                "fold_enrichment": round(fold_enrichment, 2),
                "pvalue": pvalue,
            })

        if not results:
            logger.warning("No pathway overlaps found (gene names may not match pathway DB)")
            return pd.DataFrame(columns=["n_overlap", "n_pathway", "fold_enrichment", "pvalue", "significant"])

        df = pd.DataFrame(results).sort_values("pvalue")
        df["significant"] = df["pvalue"] < pvalue_threshold
        logger.info(f"Pathway enrichment: {df['significant'].sum()}/{len(df)} significant")
        return df.set_index("pathway")
