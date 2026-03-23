"""Biomarker discovery via rank aggregation and pathway enrichment."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# MM-relevant pathways for enrichment
HALLMARK_PATHWAYS = {
    "MYC_TARGETS": ["MYC", "NPM1", "NCL", "RPS3", "RPL3", "EIF4A1", "LDHA"],
    "IL6_JAK_STAT3": ["IL6", "IL6R", "JAK1", "JAK2", "STAT3", "SOCS3", "MCL1"],
    "UNFOLDED_PROTEIN_RESPONSE": ["XBP1", "ATF4", "ATF6", "DDIT3", "HSPA5", "CALR"],
    "P53_PATHWAY": ["TP53", "CDKN1A", "MDM2", "BAX", "BBC3", "GADD45A"],
    "NFKB_SIGNALING": ["NFKB1", "RELA", "TNF", "TNFAIP3", "NFKBIA"],
    "PI3K_AKT_MTOR": ["PIK3CA", "AKT1", "MTOR", "PTEN", "TSC1", "TSC2"],
    "CELL_CYCLE": ["CDK4", "CDK6", "CCND1", "CCND2", "CCNE1", "RB1"],
    "APOPTOSIS": ["BCL2", "MCL1", "BAX", "BAK1", "CASP3", "CASP9"],
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
        """Features selected in >threshold fraction of bootstrap samples."""
        from sklearn.linear_model import LassoCV
        rng = np.random.RandomState(seed)
        n, p = X.shape
        selection_counts = np.zeros(p)

        for i in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            try:
                lasso = LassoCV(cv=3, random_state=seed + i, max_iter=1000)
                lasso.fit(X[idx], T[idx])
                selected = np.abs(lasso.coef_) > 0
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
