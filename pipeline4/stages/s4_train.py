"""Stage 4: Model training — dispatch to all enabled models."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_train(config: Any, context: Dict) -> Dict:
    """Train each enabled model, save checkpoints, log metrics."""
    from pipeline4.models import MODEL_REGISTRY
    from pipeline4.checkpoint import CheckpointManager
    from pipeline4.data.clinical_encoder import ClinicalEncoder

    features = pd.read_parquet(context["features_path"])
    clinical = pd.read_parquet(context["clinical_path"])
    split_info = context["split_info"]

    ckpt_mgr = CheckpointManager(config.base.checkpoints_dir)
    encoder = ClinicalEncoder()
    T, E = encoder.get_survival_data(clinical)

    # Split data
    train_idx = np.array(split_info["train"])
    val_idx = np.array(split_info["val"])

    X_train = features.iloc[train_idx].values
    X_val = features.iloc[val_idx].values
    T_train, E_train = T[train_idx], E[train_idx]
    T_val, E_val = T[val_idx], E[val_idx]

    run_id = context["run_id"]
    model_results = {}
    trained_models = {}

    for model_name in config.train.enabled_models:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        logger.info(f"Training {model_name}...")
        params = config.train.model_params.get(model_name, {})

        try:
            model_cls = MODEL_REGISTRY[model_name]

            if model_name == "attention_fusion":
                # Needs modality splits
                model = model_cls(device=config.base.device, **params)
                metrics = model.fit(
                    X_train, T_train, E_train,
                    modality_splits=context.get("modality_splits", {}),
                    val_X=X_val, val_T=T_val, val_E=E_val,
                    n_epochs=config.train.n_epochs,
                    batch_size=config.train.batch_size,
                    patience=config.train.early_stopping_patience,
                    lr=config.train.learning_rate,
                )
            elif model_name == "response_classifier":
                y_train = encoder.get_treatment_response(clinical)[train_idx]
                y_val = encoder.get_treatment_response(clinical)[val_idx]
                model = model_cls(**params)
                metrics = model.fit(X_train, y_train, X_val, y_val)
            elif model_name == "deepsurv":
                model = model_cls(
                    input_dim=X_train.shape[1], device=config.base.device, **params,
                )
                metrics = model.fit(
                    X_train, T_train, E_train,
                    val_X=X_val, val_T=T_val, val_E=E_val,
                    n_epochs=config.train.n_epochs,
                    batch_size=config.train.batch_size,
                    patience=config.train.early_stopping_patience,
                )
            elif model_name == "cox_ph":
                model = model_cls(**params)
                X_train_df = pd.DataFrame(X_train, columns=features.columns)
                metrics = model.fit(X_train_df, T_train, E_train)
            else:
                model = model_cls(**params)
                metrics = model.fit(X_train, T_train, E_train)

            config_hash = ckpt_mgr.compute_config_hash(params)
            ckpt_path = ckpt_mgr.save(
                model.model if hasattr(model, "model") and model.model is not None else model,
                model_name, run_id, epoch=0, metrics=metrics, config_hash=config_hash,
            )

            model_results[model_name] = {"metrics": metrics, "checkpoint": ckpt_path}
            trained_models[model_name] = model
            logger.info(f"  {model_name}: {metrics}")

        except Exception as e:
            logger.error(f"  {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            model_results[model_name] = {"metrics": {}, "error": str(e)}

    context["model_results"] = model_results
    context["trained_models"] = trained_models

    logger.info(f"Training complete: {len(trained_models)}/{len(config.train.enabled_models)} models")
    return context
