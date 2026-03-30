"""Clinical feature builder."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ClinicalFeatureBuilder:
    """Build clinical features for modeling."""

    def build(
        self, clinical_df: pd.DataFrame, config: Optional[dict] = None,
        train_idx: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Orchestrate all clinical feature engineering.

        When train_idx is provided, normalization statistics are computed
        from training rows only to prevent test-set leakage.
        """
        parts = []

        parts.append(self.compute_iss_features(clinical_df))
        parts.append(self.compute_cytogenetic_features(clinical_df))
        parts.append(self.compute_treatment_features(clinical_df))
        parts.append(self.compute_demographic_features(clinical_df, train_idx=train_idx))
        parts.append(self.compute_lab_features(clinical_df, train_idx=train_idx))

        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame(index=clinical_df.index)

        result = pd.concat(parts, axis=1).fillna(0)
        logger.info(f"Built {result.shape[1]} clinical features")
        return result

    def compute_iss_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ISS stage features."""
        if "iss_stage" not in df.columns:
            return pd.DataFrame(index=df.index)
        iss = pd.get_dummies(df["iss_stage"], prefix="iss", dtype=float)
        # Ordinal encoding
        iss["iss_ordinal"] = pd.to_numeric(df["iss_stage"], errors="coerce").fillna(2)
        return iss

    def compute_cytogenetic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cytogenetic risk features."""
        if "cytogenetic_risk" not in df.columns:
            return pd.DataFrame(index=df.index)
        return pd.get_dummies(df["cytogenetic_risk"], prefix="cyto", dtype=float)

    def compute_treatment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Treatment regimen features."""
        if "treatment" not in df.columns:
            return pd.DataFrame(index=df.index)
        return pd.get_dummies(df["treatment"], prefix="tx", dtype=float)

    def compute_demographic_features(
        self, df: pd.DataFrame,
        train_idx: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Age and sex features. Statistics fit on train only."""
        fit_df = df.iloc[train_idx] if train_idx is not None else df
        parts = []
        if "age" in df.columns:
            train_median = fit_df["age"].astype(float).median()
            train_mean = fit_df["age"].astype(float).mean()
            train_std = fit_df["age"].astype(float).std()
            age = df["age"].astype(float).fillna(train_median)
            age_scaled = (age - train_mean) / (train_std + 1e-8)
            parts.append(age_scaled.to_frame("age_scaled"))
            parts.append((age > 65).astype(float).to_frame("age_over_65"))
        if "sex" in df.columns:
            sex = df["sex"].map({"M": 1, "F": 0, "Male": 1, "Female": 0}).fillna(0)
            parts.append(sex.to_frame("sex_male"))
        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)

    def compute_lab_features(
        self, df: pd.DataFrame,
        train_idx: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Laboratory value features. Statistics fit on train only."""
        fit_df = df.iloc[train_idx] if train_idx is not None else df
        labs = ["beta2_microglobulin", "albumin", "ldh"]
        parts = []
        for lab in labs:
            if lab in df.columns:
                train_vals = fit_df[lab].astype(float)
                train_median = train_vals.median()
                train_mean = train_vals.mean()
                train_std = train_vals.std()
                vals = df[lab].astype(float).fillna(train_median)
                scaled = (vals - train_mean) / (train_std + 1e-8)
                parts.append(scaled.to_frame(f"{lab}_scaled"))
        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)
