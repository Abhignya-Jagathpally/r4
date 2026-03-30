"""Stage 5: Evaluation — metrics, bootstrap CIs, fairness audit."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_evaluate(config: Any, context: Dict) -> Dict:
    """Evaluate all trained models on test set."""
    from pipeline4.evaluation.survival_metrics import concordance_index, time_dependent_auc
    from pipeline4.evaluation.classification_metrics import full_classification_report
    from pipeline4.evaluation.bootstrap import bootstrap_ci
    from pipeline4.evaluation.fairness import FairnessAuditor
    from pipeline4.data.clinical_encoder import ClinicalEncoder

    features = pd.read_parquet(context["features_path"])
    clinical = pd.read_parquet(context["clinical_path"])
    split_info = context["split_info"]
    trained_models = context.get("trained_models", {})

    encoder = ClinicalEncoder()
    T, E = encoder.get_survival_data(clinical)

    test_idx = np.array(split_info["test"])
    train_idx = np.array(split_info["train"])
    X_test = features.iloc[test_idx].values
    T_test, E_test = T[test_idx], E[test_idx]
    T_train, E_train = T[train_idx], E[train_idx]

    results_dir = Path(config.base.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for model_name, model in trained_models.items():
        logger.info(f"Evaluating {model_name}...")
        metrics = {}

        try:
            if model_name == "response_classifier":
                y_test = encoder.get_treatment_response(clinical, train_idx=train_idx)[test_idx]
                y_prob = model.predict_proba(X_test)[:, 1]
                clf_report = full_classification_report(y_test, y_prob)
                metrics.update(clf_report)
            else:
                # Survival model
                if model_name == "cox_ph":
                    X_test_df = pd.DataFrame(X_test, columns=features.columns)
                    risk_scores = model.predict(X_test_df)
                else:
                    risk_scores = model.predict(X_test)

                # C-index
                ci = concordance_index(T_test, E_test, risk_scores)
                metrics["c_index"] = ci

                # Bootstrap CI for C-index
                ci_boot = bootstrap_ci(
                    concordance_index, T_test, E_test, risk_scores,
                    n_iterations=min(config.evaluate.bootstrap_n, 200),
                    ci_level=config.evaluate.ci_level,
                )
                metrics["c_index_ci"] = ci_boot

                # Time-dependent AUC
                try:
                    td_auc = time_dependent_auc(
                        T_train, E_train, T_test, E_test, risk_scores,
                        config.evaluate.time_horizons,
                    )
                    metrics["td_auc"] = td_auc
                except Exception as e:
                    logger.warning(f"td-AUC failed for {model_name}: {e}")

            all_metrics[model_name] = metrics
            logger.info(f"  {model_name}: {_summarize(metrics)}")

        except Exception as e:
            logger.error(f"  {model_name} evaluation failed: {e}")
            all_metrics[model_name] = {"error": str(e)}

    # Fairness audit
    fairness_results = {}
    auditor = FairnessAuditor(config.evaluate.fairness_groups)
    for group_col in config.evaluate.fairness_groups:
        if group_col in clinical.columns:
            groups = clinical.iloc[test_idx][group_col]
            for model_name, model in trained_models.items():
                if model_name == "response_classifier":
                    continue
                try:
                    if model_name == "cox_ph":
                        risk = model.predict(pd.DataFrame(X_test, columns=features.columns))
                    else:
                        risk = model.predict(X_test)
                    fair = auditor.survival_fairness(T_test, E_test, risk, groups)
                    fairness_results[f"{model_name}_{group_col}"] = fair
                except Exception as e:
                    logger.warning(f"Fairness audit failed: {e}")

    # Cross-validate C-index with legacy baselines module
    try:
        from baselines import SurvivalAnalyzer
        legacy_analyzer = SurvivalAnalyzer()
        for model_name, model in trained_models.items():
            if model_name == "response_classifier" or "c_index" not in all_metrics.get(model_name, {}):
                continue
            if model_name == "cox_ph":
                risk = model.predict(pd.DataFrame(X_test, columns=features.columns))
            else:
                risk = model.predict(X_test)
            legacy_ci = legacy_analyzer.compute_c_index(T_test, E_test, risk)
            pipeline_ci = all_metrics[model_name]["c_index"]
            if abs(legacy_ci - pipeline_ci) > 0.01:
                logger.warning(
                    f"{model_name} C-index mismatch: pipeline={pipeline_ci:.4f} vs "
                    f"legacy={legacy_ci:.4f} — check sign conventions"
                )
            all_metrics[model_name]["legacy_c_index"] = legacy_ci
    except Exception as e:
        logger.debug(f"Legacy baselines cross-validation skipped: {e}")

    # Save results
    from pipeline4.utils.io import write_json
    write_json(all_metrics, str(results_dir / "metrics.json"))
    write_json(fairness_results, str(results_dir / "fairness.json"))

    # Build comparison table
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {"model": model_name}
        if "c_index" in metrics:
            row["c_index"] = metrics["c_index"]
        if "c_index_ci" in metrics:
            row["ci_lower"] = metrics["c_index_ci"]["ci_lower"]
            row["ci_upper"] = metrics["c_index_ci"]["ci_upper"]
        if "auroc" in metrics:
            row["auroc"] = metrics["auroc"]
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(results_dir / "model_comparison.csv", index=False)

    context["evaluation_metrics"] = all_metrics
    context["fairness_results"] = fairness_results

    logger.info(f"Evaluation complete for {len(all_metrics)} models")
    return context


def _summarize(metrics: Dict) -> str:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif isinstance(v, dict) and "point_estimate" in v:
            parts.append(f"{k}={v['point_estimate']:.4f}")
    return ", ".join(parts[:5])
