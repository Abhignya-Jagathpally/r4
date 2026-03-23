"""Stage 3: Cohort construction — stratified splitting."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

logger = logging.getLogger(__name__)


def run_cohort(config: Any, context: Dict) -> Dict:
    """Build patient-level stratified train/val/test splits."""
    features = pd.read_parquet(context["features_path"])
    clinical = pd.read_parquet(context["clinical_path"])

    splits_dir = Path(config.base.data_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Stratification variable
    strat_col = config.cohort.stratify_by
    if strat_col in clinical.columns:
        strat = clinical[strat_col].values.astype(int)
    else:
        logger.warning(f"Stratify column '{strat_col}' not found, using event")
        strat = clinical["event"].values.astype(int)

    patient_ids = features.index.values
    n = len(patient_ids)

    # Test split
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=config.cohort.test_size,
        random_state=config.cohort.random_state,
    )
    trainval_idx, test_idx = next(sss_test.split(np.zeros(n), strat))

    # Val split from trainval
    strat_tv = strat[trainval_idx]
    relative_val = config.cohort.val_size / (1 - config.cohort.test_size)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_val,
        random_state=config.cohort.random_state,
    )
    train_idx_rel, val_idx_rel = next(sss_val.split(np.zeros(len(trainval_idx)), strat_tv))
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    # CV folds on training set
    cv_folds = []
    skf = StratifiedKFold(
        n_splits=config.cohort.cv_folds, shuffle=True,
        random_state=config.cohort.random_state,
    )
    for fold_train, fold_val in skf.split(np.zeros(len(train_idx)), strat[train_idx]):
        cv_folds.append({
            "train": train_idx[fold_train].tolist(),
            "val": train_idx[fold_val].tolist(),
        })

    # Verify no leakage
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    assert train_set.isdisjoint(val_set), "Train/val overlap!"
    assert train_set.isdisjoint(test_set), "Train/test overlap!"
    assert val_set.isdisjoint(test_set), "Val/test overlap!"

    split_info = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
        "cv_folds": cv_folds,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_event_rate": float(strat[train_idx].mean()),
        "val_event_rate": float(strat[val_idx].mean()),
        "test_event_rate": float(strat[test_idx].mean()),
    }

    from pipeline4.utils.io import write_json
    write_json(split_info, str(splits_dir / "split_indices.json"))

    context["split_info"] = split_info
    context["splits_path"] = str(splits_dir / "split_indices.json")

    logger.info(
        f"Cohort splits: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)}, {config.cohort.cv_folds} CV folds"
    )
    return context
