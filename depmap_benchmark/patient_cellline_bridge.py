"""Patient-cell line bridge (M56: real cosine similarity, M65: top-k matches)."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class PatientCellLineBridge:
    """Match patients to cell lines via molecular similarity."""

    def match(
        self, patient_profile: pd.DataFrame, cellline_profiles: pd.DataFrame, top_k: int = 5,
    ) -> List[Dict]:
        """Return top-k matches with cosine similarity confidence (M56, M65)."""
        # Align features
        common_features = patient_profile.columns.intersection(cellline_profiles.columns)
        if len(common_features) == 0:
            logger.warning("No common features for patient-cellline matching")
            return []

        P = patient_profile[common_features].fillna(0).values
        C = cellline_profiles[common_features].fillna(0).values

        # M56: ACTUAL cosine similarity computation
        sim_matrix = cosine_similarity(P, C)

        results = []
        for i, patient_id in enumerate(patient_profile.index):
            sims = sim_matrix[i]
            top_indices = np.argsort(sims)[-top_k:][::-1]

            matches = []
            for idx in top_indices:
                matches.append({
                    "cellline": cellline_profiles.index[idx],
                    "cosine_similarity": float(sims[idx]),
                    "rank": len(matches) + 1,
                })

            results.append({
                "patient_id": patient_id,
                "top_k_matches": matches,
                "best_match": matches[0]["cellline"] if matches else None,
                "best_similarity": matches[0]["cosine_similarity"] if matches else 0.0,
            })

        logger.info(f"Matched {len(results)} patients to top-{top_k} cell lines")
        return results
