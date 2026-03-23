"""MM pathway oracle with complete gene sets (M59, M64)."""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# M59: All 8 pathway gene sets with FULL gene lists
MM_PATHWAYS = {
    "NFKB_SIGNALING": {
        "genes": ["NFKB1", "NFKB2", "RELA", "RELB", "REL", "NFKBIA", "NFKBIB",
                   "NFKBIE", "IKBKA", "IKBKB", "IKBKG", "TNFAIP3", "TRAF2", "TRAF3",
                   "TRAF6", "NIK", "BCL3", "CYLD", "BIRC2", "BIRC3", "XIAP", "CFLAR",
                   "TNF", "TNFRSF1A", "CD40", "LTBR", "BAFF", "APRIL"],
        "evidence": "high", "role": "survival/proliferation",
    },
    "PI3K_AKT_MTOR": {
        "genes": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PIK3R1", "PIK3R2",
                   "AKT1", "AKT2", "AKT3", "MTOR", "RPTOR", "RICTOR", "PTEN",
                   "TSC1", "TSC2", "RHEB", "RPS6KB1", "EIF4EBP1", "PDK1", "SGK1",
                   "FOXO1", "FOXO3", "GSK3B", "BAD", "BCL2L11", "DEPTOR"],
        "evidence": "high", "role": "survival/growth",
    },
    "JAK_STAT": {
        "genes": ["JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT2", "STAT3",
                   "STAT4", "STAT5A", "STAT5B", "STAT6", "IL6", "IL6R", "IL6ST",
                   "IL10", "IL21", "SOCS1", "SOCS3", "PIAS1", "PIAS3", "PIM1",
                   "PIM2", "BCL2L1", "MCL1", "CCND1", "MYC", "IRF4"],
        "evidence": "high", "role": "IL-6 driven proliferation",
    },
    "RAS_MAPK": {
        "genes": ["KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "ARAF", "MAP2K1",
                   "MAP2K2", "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK14",
                   "SOS1", "GRB2", "SHC1", "DUSP1", "DUSP6", "SPRY2", "ELK1",
                   "FOS", "JUN", "MYC", "CDKN2A", "NF1", "PTPN11"],
        "evidence": "high", "role": "proliferation/drug_resistance",
    },
    "MYC_PROGRAM": {
        "genes": ["MYC", "MYCN", "MYCL", "MAX", "MXD1", "MNT", "NPM1",
                   "NCL", "DDX21", "NOP56", "NOP58", "FBL", "RPS3", "RPL3",
                   "RPS6", "EIF4A1", "EIF4E", "LDHA", "PKM", "ENO1", "HK2",
                   "SLC2A1", "TERT", "CDK4", "CCND1", "ODC1"],
        "evidence": "high", "role": "proliferation/metabolism",
    },
    "CELL_CYCLE": {
        "genes": ["CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCND1",
                   "CCND2", "CCND3", "CCNE1", "CCNE2", "RB1", "RBL1", "RBL2",
                   "E2F1", "E2F2", "E2F3", "CDKN1A", "CDKN1B", "CDKN2A",
                   "CDKN2B", "TP53", "MDM2", "AURKA", "AURKB", "PLK1"],
        "evidence": "medium", "role": "proliferation",
    },
    "APOPTOSIS": {
        "genes": ["BCL2", "BCL2L1", "MCL1", "BCL2L2", "BCL2A1", "BAX", "BAK1",
                   "BID", "BAD", "BIM", "PUMA", "NOXA", "CASP3", "CASP7",
                   "CASP8", "CASP9", "APAF1", "CYCS", "DIABLO", "XIAP",
                   "BIRC5", "CFLAR", "FAS", "FASLG", "TRAIL", "DR4", "DR5"],
        "evidence": "high", "role": "apoptosis_evasion",
    },
    "UNFOLDED_PROTEIN_RESPONSE": {
        "genes": ["XBP1", "IRE1", "ERN1", "ATF4", "ATF6", "DDIT3", "HSPA5",
                   "HSP90AA1", "HSP90AB1", "HSPA1A", "DNAJA1", "DNAJB1",
                   "CALR", "PDIA4", "PDIA6", "EIF2AK3", "EIF2S1", "GADD34",
                   "CHOP", "EDEM1", "SEL1L", "HRD1", "VCP", "UFD1", "DERL1"],
        "evidence": "high", "role": "proteostasis/PI_target",
    },
}


class MMPathwayOracle:
    """Oracle for MM pathway recovery scoring (M64: evidence + sensitivity/specificity)."""

    def __init__(self, pathways: Optional[Dict] = None):
        self.pathways = pathways or MM_PATHWAYS

    def pathway_recovery_score(
        self, predicted_genes: List[str], pathway_name: str,
    ) -> Dict:
        """Score pathway recovery with sensitivity and specificity (M64)."""
        if pathway_name not in self.pathways:
            return {"error": f"Unknown pathway: {pathway_name}"}

        pathway = self.pathways[pathway_name]
        true_genes = set(pathway["genes"])
        pred_set = set(predicted_genes)

        tp = len(pred_set & true_genes)
        fp = len(pred_set - true_genes)
        fn = len(true_genes - pred_set)

        sensitivity = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-10)

        return {
            "pathway": pathway_name,
            "evidence_level": pathway["evidence"],
            "role": pathway["role"],
            "n_pathway_genes": len(true_genes),
            "n_recovered": tp,
            "sensitivity": float(sensitivity),
            "precision": float(precision),
            "f1": float(f1),
            "recovered_genes": sorted(pred_set & true_genes),
            "missed_genes": sorted(true_genes - pred_set),
        }

    def full_recovery_report(self, predicted_genes: List[str]) -> List[Dict]:
        """Score recovery across all pathways."""
        return [
            self.pathway_recovery_score(predicted_genes, pw)
            for pw in self.pathways
        ]
