"""Stage 8: Hyperparameter autotuning via Optuna."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_autotune(config: Any, context: Dict) -> Dict:
    """Run Optuna hyperparameter search for target model."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed, skipping autotune")
        return context

    from pipeline4.models import MODEL_REGISTRY
    from pipeline4.data.clinical_encoder import ClinicalEncoder

    features = pd.read_parquet(context["features_path"])
    clinical = pd.read_parquet(context["clinical_path"])
    split_info = context["split_info"]

    encoder = ClinicalEncoder()
    T, E = encoder.get_survival_data(clinical)

    train_idx = np.array(split_info["train"])
    val_idx = np.array(split_info["val"])
    X_train, X_val = features.iloc[train_idx].values, features.iloc[val_idx].values
    T_train, E_train = T[train_idx], E[train_idx]
    T_val, E_val = T[val_idx], E[val_idx]

    target_model = config.autotune.target_model
    search_spaces = config.autotune.search_spaces.get(target_model, {})

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for param_name, spec in search_spaces.items():
            ptype = spec.get("type", "float")
            if ptype == "float":
                params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"])
            elif ptype == "loguniform":
                params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"], log=True)
            elif ptype == "int":
                params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"])
            elif ptype == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, [str(c) for c in spec["choices"]])
                # Convert back if needed
                if param_name == "hidden_dims":
                    import ast
                    params[param_name] = ast.literal_eval(params[param_name])
                elif param_name == "batch_size":
                    params[param_name] = int(params[param_name])

        try:
            model_cls = MODEL_REGISTRY[target_model]

            if target_model == "deepsurv":
                model = model_cls(input_dim=X_train.shape[1], device=config.base.device, **params)
                metrics = model.fit(
                    X_train, T_train, E_train,
                    val_X=X_val, val_T=T_val, val_E=E_val,
                    n_epochs=50, batch_size=params.get("batch_size", 64), patience=10,
                )
                return metrics.get("val_c_index", 0.5)
            elif target_model == "rsf":
                model = model_cls(**params)
                model.fit(X_train, T_train, E_train)
                risk = model.predict(X_val)
                from lifelines.utils import concordance_index as ci_fn
                return ci_fn(T_val, -risk, E_val)
            else:
                model = model_cls(**params)
                model.fit(X_train, T_train, E_train)
                risk = model.predict(X_val) if hasattr(model, "predict") else np.zeros(len(X_val))
                from lifelines.utils import concordance_index as ci_fn
                return ci_fn(T_val, -risk, E_val)
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.5

    # Create study with pruner
    pruner = optuna.pruners.HyperbandPruner(
        **config.autotune.pruner_params,
    ) if config.autotune.pruner == "hyperband" else optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        objective,
        n_trials=config.autotune.n_trials,
        timeout=config.autotune.timeout_seconds,
    )

    # Save best config
    best = study.best_trial
    from pipeline4.utils.io import write_json
    write_json(
        {"best_params": best.params, "best_value": best.value, "n_trials": len(study.trials)},
        "configs/best_trial.json",
    )

    context["autotune_best"] = best.params
    context["autotune_best_value"] = best.value

    logger.info(
        f"Autotune complete: best {config.autotune.target_metric}={best.value:.4f} "
        f"in {len(study.trials)} trials"
    )
    return context
