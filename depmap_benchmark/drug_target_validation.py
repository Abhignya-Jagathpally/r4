"""Drug target validation (M57: full drug DB, M58: real validate_prediction)."""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# M57: Complete MM drug database
MM_DRUG_DATABASE = {
    "bortezomib": {"targets": ["PSMB5", "PSMB1"], "mechanism": "proteasome_inhibitor", "class": "PI"},
    "carfilzomib": {"targets": ["PSMB5"], "mechanism": "proteasome_inhibitor", "class": "PI"},
    "ixazomib": {"targets": ["PSMB5"], "mechanism": "proteasome_inhibitor", "class": "PI"},
    "lenalidomide": {"targets": ["CRBN", "IKZF1", "IKZF3"], "mechanism": "immunomodulatory", "class": "IMiD"},
    "pomalidomide": {"targets": ["CRBN", "IKZF1", "IKZF3"], "mechanism": "immunomodulatory", "class": "IMiD"},
    "thalidomide": {"targets": ["CRBN"], "mechanism": "immunomodulatory", "class": "IMiD"},
    "daratumumab": {"targets": ["CD38"], "mechanism": "monoclonal_antibody", "class": "mAb"},
    "elotuzumab": {"targets": ["SLAMF7"], "mechanism": "monoclonal_antibody", "class": "mAb"},
    "panobinostat": {"targets": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"], "mechanism": "hdac_inhibitor", "class": "HDACi"},
    "dexamethasone": {"targets": ["NR3C1"], "mechanism": "glucocorticoid", "class": "steroid"},
    "melphalan": {"targets": ["DNA"], "mechanism": "alkylating_agent", "class": "alkylator"},
    "venetoclax": {"targets": ["BCL2"], "mechanism": "bcl2_inhibitor", "class": "BH3_mimetic"},
    "selinexor": {"targets": ["XPO1"], "mechanism": "xpo1_inhibitor", "class": "SINE"},
}


class DrugTargetValidator:
    """Validate drug sensitivity predictions against known targets."""

    def __init__(self, drug_db: Optional[Dict] = None):
        self.drug_db = drug_db or MM_DRUG_DATABASE

    def validate_prediction(
        self, predictions: Dict[str, float], drug_name: str,
        gene_importance: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Validate prediction against known targets (M58: real validation)."""
        drug_info = self.drug_db.get(drug_name.lower())
        if drug_info is None:
            return {"validated": False, "reason": f"Drug '{drug_name}' not in database"}

        known_targets = set(drug_info["targets"])
        result = {
            "drug": drug_name,
            "mechanism": drug_info["mechanism"],
            "drug_class": drug_info["class"],
            "known_targets": list(known_targets),
            "validated": True,
        }

        if gene_importance:
            # Check if known targets are among top features
            sorted_genes = sorted(gene_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top_50 = {g for g, _ in sorted_genes[:50]}
            top_100 = {g for g, _ in sorted_genes[:100]}

            target_in_top50 = known_targets & top_50
            target_in_top100 = known_targets & top_100

            result["targets_in_top50"] = list(target_in_top50)
            result["targets_in_top100"] = list(target_in_top100)
            result["target_recovery_rate"] = len(target_in_top100) / max(len(known_targets), 1)

            # Confidence scoring
            if target_in_top50:
                result["confidence"] = "high"
            elif target_in_top100:
                result["confidence"] = "medium"
            else:
                result["confidence"] = "low"

        return result

    def get_all_targets(self) -> Dict[str, List[str]]:
        """Get all drug targets from database."""
        return {drug: info["targets"] for drug, info in self.drug_db.items()}
