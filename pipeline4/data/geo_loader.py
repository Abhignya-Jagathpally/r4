"""GEO clinical data loader for Multiple Myeloma datasets."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GEOClinicalLoader:
    """Download and parse clinical metadata from GEO."""

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_clinical(self, accession: str) -> pd.DataFrame:
        """Fetch clinical metadata from GEO series matrix."""
        try:
            import GEOparse
            gse = GEOparse.get_GEO(geo=accession, destdir=str(self.cache_dir), silent=True)
            clinical = self._parse_clinical(gse)
            logger.info(f"Fetched clinical data for {accession}: {len(clinical)} patients")
            return clinical
        except Exception as e:
            logger.warning(f"GEO fetch failed for {accession}: {e}. Using synthetic data.")
            return pd.DataFrame()

    def _parse_clinical(self, gse) -> pd.DataFrame:
        """Parse clinical columns from GEO phenotype data."""
        pheno = gse.phenotype_data
        records = []

        for sample_id, row in pheno.iterrows():
            record = {"patient_id": sample_id}
            # Parse characteristics_ch1 fields
            for col in pheno.columns:
                val = str(row[col])
                if ":" in val:
                    key, value = val.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    record[key] = value.strip()
                elif col.startswith("characteristics_ch"):
                    continue
                else:
                    record[col] = val
            records.append(record)

        df = pd.DataFrame(records)
        df = self._standardize_columns(df)
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map common GEO column variants to standard names."""
        col_map = {
            "overall_survival": "survival_time",
            "os_months": "survival_time",
            "os_time": "survival_time",
            "efs_months": "survival_time",
            "pfs_months": "survival_time",
            "os_event": "event",
            "os_status": "event",
            "vital_status": "event",
            "iss_stage": "iss_stage",
            "iss": "iss_stage",
            "r_iss": "iss_stage",
            "cytogenetic_risk": "cytogenetic_risk",
            "cyto_risk": "cytogenetic_risk",
            "treatment": "treatment",
            "therapy": "treatment",
            "age": "age",
            "age_at_diagnosis": "age",
            "gender": "sex",
            "sex": "sex",
        }
        rename = {}
        for col in df.columns:
            normalized = col.lower().strip().replace(" ", "_")
            if normalized in col_map:
                rename[col] = col_map[normalized]
        df = df.rename(columns=rename)

        # Convert event to binary
        if "event" in df.columns:
            df["event"] = df["event"].map(
                lambda x: 1 if str(x).lower() in ("1", "dead", "deceased", "yes", "true") else 0
            )

        # Convert numeric columns
        for col in ["survival_time", "age"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def generate_synthetic_clinical(
        self, n_patients: int = 500, seed: int = 42
    ) -> pd.DataFrame:
        """Generate realistic synthetic MM clinical data for demo/testing."""
        rng = np.random.RandomState(seed)

        iss_stages = rng.choice([1, 2, 3], size=n_patients, p=[0.3, 0.4, 0.3])
        cyto_risk = rng.choice(
            ["standard", "high", "unknown"], size=n_patients, p=[0.5, 0.3, 0.2]
        )
        treatments = rng.choice(
            ["VRd", "Rd", "VCd", "KRd", "DRd"], size=n_patients, p=[0.3, 0.25, 0.2, 0.15, 0.1]
        )
        ages = rng.normal(65, 10, n_patients).clip(35, 90).astype(int)
        sex = rng.choice(["M", "F"], size=n_patients, p=[0.55, 0.45])

        # Survival times from Weibull with ISS-dependent scale
        scale = np.where(iss_stages == 1, 60, np.where(iss_stages == 2, 40, 25))
        # High-risk cytogenetics reduce survival
        scale = np.where(cyto_risk == "high", scale * 0.6, scale)
        survival_time = rng.weibull(1.5, n_patients) * scale
        survival_time = np.clip(survival_time, 1, 120).round(1)

        # Censoring: ~35% censored
        max_followup = rng.uniform(12, 72, n_patients)
        event = (survival_time <= max_followup).astype(int)
        survival_time = np.minimum(survival_time, max_followup).round(1)

        # Treatment response (correlated with ISS and cytogenetics)
        response_prob = 0.6 - (iss_stages - 1) * 0.1
        response_prob = np.where(cyto_risk == "high", response_prob - 0.15, response_prob)
        treatment_response = rng.binomial(1, response_prob.clip(0.1, 0.9))

        df = pd.DataFrame({
            "patient_id": [f"PT_{i:04d}" for i in range(n_patients)],
            "survival_time": survival_time,
            "event": event,
            "iss_stage": iss_stages,
            "cytogenetic_risk": cyto_risk,
            "treatment": treatments,
            "age": ages,
            "sex": sex,
            "treatment_response": treatment_response,
            "beta2_microglobulin": rng.lognormal(1.0, 0.5, n_patients).round(2),
            "albumin": rng.normal(3.5, 0.6, n_patients).clip(1.5, 5.5).round(2),
            "ldh": rng.lognormal(5.5, 0.3, n_patients).round(0),
        })
        df = df.set_index("patient_id")

        logger.info(
            f"Generated synthetic clinical data: {n_patients} patients, "
            f"event rate={event.mean():.2f}, median OS={np.median(survival_time):.1f}mo"
        )
        return df
