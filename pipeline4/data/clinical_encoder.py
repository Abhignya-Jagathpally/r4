"""Clinical feature encoding and preprocessing."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ClinicalEncoder:
    """Encode clinical variables for model input."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._scaler = StandardScaler()
        self._lab_stats: Dict[str, Tuple[float, float]] = {}
        self._fitted = False

    def encode(
        self, clinical_df: pd.DataFrame,
        train_idx: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Encode all clinical features into numeric matrix.

        When train_idx is provided, scalers and statistics are fit on
        training rows only to prevent test-set leakage.
        """
        fit_df = clinical_df.iloc[train_idx] if train_idx is not None else clinical_df
        encoded_parts = []

        # ISS stage (ordinal — one-hot is stateless, no leakage)
        if "iss_stage" in clinical_df.columns:
            iss = pd.get_dummies(clinical_df["iss_stage"], prefix="iss", dtype=float)
            encoded_parts.append(iss)

        # Cytogenetic risk (one-hot)
        if "cytogenetic_risk" in clinical_df.columns:
            cyto = pd.get_dummies(clinical_df["cytogenetic_risk"], prefix="cyto", dtype=float)
            encoded_parts.append(cyto)

        # Treatment (one-hot)
        if "treatment" in clinical_df.columns:
            treat = pd.get_dummies(clinical_df["treatment"], prefix="tx", dtype=float)
            encoded_parts.append(treat)

        # Age (standardized — fit on train only)
        if "age" in clinical_df.columns:
            age = clinical_df[["age"]].copy().astype(float)
            train_median = fit_df["age"].median()
            age = age.fillna(train_median)
            if not self._fitted:
                self._scaler.fit(age.iloc[train_idx] if train_idx is not None else age)
                self._fitted = True
            age_scaled = pd.DataFrame(
                self._scaler.transform(age), index=clinical_df.index, columns=["age_scaled"]
            )
            encoded_parts.append(age_scaled)

        # Sex (binary — stateless mapping, no leakage)
        if "sex" in clinical_df.columns:
            sex = (clinical_df["sex"].map({"M": 1, "F": 0, "Male": 1, "Female": 0})
                   .fillna(0).to_frame("sex_male"))
            encoded_parts.append(sex)

        # Lab values — fit mean/std on training data only
        for lab in ["beta2_microglobulin", "albumin", "ldh"]:
            if lab in clinical_df.columns:
                train_vals = fit_df[lab].astype(float)
                lab_median = train_vals.median()
                lab_mean = train_vals.mean()
                lab_std = train_vals.std()
                self._lab_stats[lab] = (lab_mean, lab_std)

                vals = clinical_df[[lab]].astype(float).fillna(lab_median)
                vals_scaled = (vals - lab_mean) / (lab_std + 1e-8)
                vals_scaled.columns = [f"{lab}_scaled"]
                encoded_parts.append(vals_scaled)

        if not encoded_parts:
            logger.warning("No clinical features to encode")
            return pd.DataFrame(index=clinical_df.index)

        result = pd.concat(encoded_parts, axis=1).fillna(0)
        logger.info(f"Encoded {result.shape[1]} clinical features for {result.shape[0]} patients")
        return result

    def get_survival_data(
        self, clinical_df: pd.DataFrame,
        time_col: str = "survival_time", event_col: str = "event",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract survival time and event arrays."""
        T = clinical_df[time_col].values.astype(float)
        E = clinical_df[event_col].values.astype(int)
        logger.info(f"Survival data: {len(T)} patients, event rate={E.mean():.2f}")
        return T, E

    def get_treatment_response(
        self, clinical_df: pd.DataFrame, col: str = "treatment_response",
        train_idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract binary treatment response labels.

        When inferring from survival, median is computed from train set only
        to prevent label leakage.
        """
        if col not in clinical_df.columns:
            logger.warning(f"Column '{col}' not found, inferring from survival")
            if train_idx is not None:
                median_os = clinical_df["survival_time"].iloc[train_idx].median()
            else:
                median_os = clinical_df["survival_time"].median()
            return (clinical_df["survival_time"] > median_os).astype(int).values
        return clinical_df[col].values.astype(int)
