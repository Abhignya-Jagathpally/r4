"""Graph ML runner (M40: drug-aware split, M47: frozen preprocessing, M50: SMILES hash)."""

import hashlib
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def drug_aware_split(
    data: pd.DataFrame, drug_col: str = "drug_id", test_size: float = 0.2, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split ensuring same drug NOT in both train and test (M40)."""
    rng = np.random.RandomState(seed)
    drugs = data[drug_col].unique()
    rng.shuffle(drugs)
    n_test = max(1, int(len(drugs) * test_size))
    test_drugs = set(drugs[:n_test])
    train_drugs = set(drugs[n_test:])

    train_idx = data.index[~data[drug_col].isin(test_drugs)].values
    test_idx = data.index[data[drug_col].isin(test_drugs)].values

    assert len(set(data.loc[train_idx, drug_col]) & test_drugs) == 0, "Drug leakage detected!"
    logger.info(f"Drug-aware split: {len(train_drugs)} train drugs, {len(test_drugs)} test drugs")
    return train_idx, test_idx


def hash_smiles(smiles_list: List[str]) -> str:
    """Hash SMILES for reproducibility verification (M50)."""
    combined = "|".join(sorted(smiles_list))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def verify_frozen_preprocessing(data: pd.DataFrame, expected_hash: str) -> None:
    """Assert preprocessing is frozen (M47)."""
    current_hash = hashlib.sha256(pd.util.hash_pandas_object(data).values.tobytes()).hexdigest()[:16]
    assert current_hash == expected_hash, (
        f"Preprocessing changed! Expected {expected_hash}, got {current_hash}. "
        "Frozen preprocessing contract violated."
    )


def run(config: Dict) -> Dict:
    """Run graph ML pipeline."""
    logger.info("Graph ML pipeline starting")
    results = {"status": "completed"}
    # Placeholder for full pipeline orchestration
    return results
