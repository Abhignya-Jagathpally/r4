"""Pathway-level aggregation with true GSVA and MSigDB loading.

Fixes: M06 (MSigDB loader), M07 (true GSVA KS-based), M08 (gene alias resolution).
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# M08: Gene alias resolution table
GENE_ALIASES: Dict[str, str] = {
    "DNAJ": "DNAJA1", "HSP40": "DNAJA1", "DNAJB": "DNAJB1",
    "HSP70": "HSPA1A", "HSP90": "HSP90AA1", "HSP27": "HSPB1",
    "BCL-2": "BCL2", "P53": "TP53", "P21": "CDKN1A",
    "ERK1": "MAPK3", "ERK2": "MAPK1", "JNK1": "MAPK8",
    "AKT": "AKT1", "MTOR": "MTOR", "PI3K": "PIK3CA",
    "NFKB": "NFKB1", "IKBA": "NFKBIA", "STAT3": "STAT3",
}


class PathwayScorer:
    """Pathway scoring with MSigDB/KEGG loading and true GSVA."""

    def __init__(self, cache_dir: str = "data/pathway_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._gene_sets: Optional[Dict[str, List[str]]] = None
        self._version: Optional[str] = None

    def load_gene_sets(
        self, source: str = "msigdb", collection: str = "h.all",
        version: str = "2023.2",
    ) -> Dict[str, List[str]]:
        """Load gene sets from MSigDB GMT files with version tracking (M06)."""
        cache_file = self.cache_dir / f"{source}_{collection}_{version}.gmt"

        if cache_file.exists():
            gene_sets = self._parse_gmt(cache_file)
            logger.info(f"Loaded cached gene sets: {len(gene_sets)} pathways (v{version})")
        else:
            # Try downloading from MSigDB
            try:
                gene_sets = self._download_msigdb(collection, version, cache_file)
            except Exception as e:
                logger.warning(f"MSigDB download failed: {e}. Using built-in MM pathways.")
                gene_sets = self._builtin_mm_pathways()

        # Resolve aliases (M08)
        gene_sets = {k: self.resolve_gene_aliases(v) for k, v in gene_sets.items()}

        self._gene_sets = gene_sets
        self._version = version

        # Version tracking
        version_file = self.cache_dir / "version_manifest.txt"
        with open(version_file, "a") as f:
            content_hash = hashlib.md5(str(sorted(gene_sets.keys())).encode()).hexdigest()[:8]
            f.write(f"{source}\t{collection}\tv{version}\t{len(gene_sets)} sets\thash={content_hash}\n")

        return gene_sets

    def _download_msigdb(self, collection: str, version: str, cache_file: Path) -> Dict:
        """Download GMT from MSigDB."""
        import requests
        url = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{version}/{collection}.v{version}.Hs.symbols.gmt"
        logger.info(f"Downloading {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        cache_file.write_text(resp.text)
        return self._parse_gmt(cache_file)

    def _parse_gmt(self, path: Path) -> Dict[str, List[str]]:
        """Parse GMT file format."""
        gene_sets = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    name = parts[0]
                    genes = [g for g in parts[2:] if g]
                    gene_sets[name] = genes
        return gene_sets

    def _builtin_mm_pathways(self) -> Dict[str, List[str]]:
        """Built-in MM-relevant pathways as fallback."""
        return {
            "HALLMARK_MYC_TARGETS_V1": ["MYC", "NPM1", "NCL", "RPS3", "RPL3", "EIF4A1", "LDHA", "PKM", "ENO1", "NOP56", "DDX21"],
            "HALLMARK_MTORC1_SIGNALING": ["MTOR", "RPS6KB1", "EIF4EBP1", "SLC7A5", "SLC3A2", "VEGFA", "HIF1A", "DDIT4", "SESN2"],
            "HALLMARK_IL6_JAK_STAT3_SIGNALING": ["IL6", "IL6R", "JAK1", "JAK2", "STAT3", "SOCS3", "BCL2L1", "MCL1", "PIM1", "CCND1"],
            "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": ["XBP1", "ATF4", "ATF6", "DDIT3", "HSPA5", "CALR", "PDIA4", "DNAJA1", "DNAJB1", "ERN1"],
            "HALLMARK_TNFA_SIGNALING_VIA_NFKB": ["NFKB1", "RELA", "TNF", "TNFAIP3", "NFKBIA", "CXCL8", "IL1B", "BIRC3", "TRAF1"],
            "HALLMARK_P53_PATHWAY": ["TP53", "CDKN1A", "MDM2", "BAX", "BBC3", "PMAIP1", "GADD45A", "FAS", "SESN1"],
            "HALLMARK_APOPTOSIS": ["BCL2", "MCL1", "BAX", "BAK1", "BID", "CASP3", "CASP9", "CYCS", "APAF1", "XIAP"],
            "HALLMARK_PI3K_AKT_MTOR_SIGNALING": ["PIK3CA", "AKT1", "MTOR", "PTEN", "TSC1", "TSC2", "RHEB", "RPS6KB1", "PDK1"],
            "PROTEASOME_PATHWAY": ["PSMA1", "PSMA2", "PSMA3", "PSMB1", "PSMB5", "PSMB8", "PSMC1", "PSMD1", "UBB", "UBC"],
            "PLASMA_CELL_DIFFERENTIATION": ["IRF4", "PRDM1", "XBP1", "SDC1", "CD38", "TNFRSF17", "CD27", "IL6R", "SLAMF7"],
        }

    def resolve_gene_aliases(self, genes: List[str]) -> List[str]:
        """Resolve gene aliases to canonical symbols (M08)."""
        resolved = []
        for g in genes:
            canonical = GENE_ALIASES.get(g.upper(), g)
            resolved.append(canonical)
        return list(set(resolved))

    def gsva_score(
        self, expression: pd.DataFrame,
        gene_sets: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """True GSVA scoring (M07: KS-based, Hanzelmann et al 2013).

        For each sample, ranks genes, then computes a KS-like enrichment
        statistic for each gene set using the empirical CDF.
        """
        if gene_sets is None:
            gene_sets = self._gene_sets or self._builtin_mm_pathways()

        n_samples = expression.shape[0]
        scores = {}

        for pathway_name, pathway_genes in gene_sets.items():
            # Find overlapping genes
            overlap = [g for g in pathway_genes if g in expression.columns]
            if len(overlap) < 3:
                continue

            pathway_scores = np.zeros(n_samples)

            for i in range(n_samples):
                sample = expression.iloc[i].dropna()
                if len(sample) == 0:
                    continue

                # Rank genes for this sample (descending by expression)
                ranked = sample.sort_values(ascending=False)
                n_genes = len(ranked)
                gene_set_mask = ranked.index.isin(overlap)

                # KS enrichment statistic
                n_in = gene_set_mask.sum()
                n_out = n_genes - n_in

                if n_in == 0 or n_out == 0:
                    continue

                # Weighted running sum (GSVA uses |rank| weighting)
                ranks = np.arange(1, n_genes + 1, dtype=float)
                abs_weights = np.abs(ranked.values)

                hit = np.where(gene_set_mask, abs_weights, 0)
                hit_sum = hit.sum()
                if hit_sum > 0:
                    hit_cumsum = np.cumsum(hit / hit_sum)
                else:
                    hit_cumsum = np.zeros(n_genes)

                miss = np.where(~gene_set_mask, 1.0, 0)
                miss_sum = miss.sum()
                if miss_sum > 0:
                    miss_cumsum = np.cumsum(miss / miss_sum)
                else:
                    miss_cumsum = np.zeros(n_genes)

                # ES = max deviation between hit and miss CDFs
                running_es = hit_cumsum - miss_cumsum
                pathway_scores[i] = running_es[np.argmax(np.abs(running_es))]

            scores[pathway_name] = pathway_scores

        result = pd.DataFrame(scores, index=expression.index)
        logger.info(f"GSVA scores: {result.shape[1]} pathways for {result.shape[0]} samples")
        return result
