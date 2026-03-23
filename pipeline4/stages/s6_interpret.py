"""Stage 6: Interpretation — SHAP, attention weights, biomarker discovery."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_interpret(config: Any, context: Dict) -> Dict:
    """Run model interpretation and biomarker discovery."""
    from pipeline4.interpretation.shap_explain import SHAPExplainer
    from pipeline4.interpretation.biomarker_discovery import BiomarkerDiscovery

    features = pd.read_parquet(context["features_path"])
    split_info = context["split_info"]
    trained_models = context.get("trained_models", {})

    test_idx = np.array(split_info["test"])
    X_test = features.iloc[test_idx]

    interp_dir = Path(config.base.results_dir) / "interpretation"
    interp_dir.mkdir(parents=True, exist_ok=True)

    all_rankings = {}

    for model_name, model in trained_models.items():
        logger.info(f"Interpreting {model_name}...")

        try:
            # Feature importance from model
            if hasattr(model, "feature_importance"):
                try:
                    imp = model.feature_importance(feature_names=list(features.columns))
                    imp.to_csv(interp_dir / f"{model_name}_importance.csv")
                    all_rankings[model_name] = imp
                except Exception as e:
                    logger.warning(f"Feature importance failed for {model_name}: {e}")

            elif hasattr(model, "get_coefficients"):
                coefs = model.get_coefficients()
                coefs.to_csv(interp_dir / f"{model_name}_coefficients.csv")
                all_rankings[model_name] = coefs["coef"].abs().sort_values(ascending=False)

            # SHAP for tree models
            if model_name in ("rsf", "response_classifier"):
                try:
                    explainer = SHAPExplainer(model, "tree")
                    top = explainer.top_features(
                        X_test.values[:config.interpret.shap_max_samples],
                        feature_names=list(features.columns),
                        n=config.interpret.top_k_genes,
                    )
                    top.to_csv(interp_dir / f"{model_name}_shap_top.csv")
                    explainer.summary_plot(
                        X_test.values[:config.interpret.shap_max_samples],
                        feature_names=list(features.columns),
                        output_path=str(interp_dir / f"{model_name}_shap_summary.png"),
                    )
                except Exception as e:
                    logger.warning(f"SHAP failed for {model_name}: {e}")

            # Attention weights for fusion model
            if model_name == "attention_fusion" and config.interpret.attention_extraction:
                try:
                    from pipeline4.interpretation.attention_weights import AttentionAnalyzer
                    analyzer = AttentionAnalyzer(model)
                    weights = analyzer.extract_weights(X_test.values)
                    importance = analyzer.modality_importance(weights)
                    importance.to_csv(interp_dir / "attention_modality_importance.csv")
                    analyzer.plot_modality_importance(
                        importance, str(interp_dir / "modality_importance.png"),
                    )
                except Exception as e:
                    logger.warning(f"Attention analysis failed: {e}")

        except Exception as e:
            logger.error(f"Interpretation failed for {model_name}: {e}")

    # Biomarker discovery
    if all_rankings:
        discovery = BiomarkerDiscovery()
        consensus = discovery.aggregate_rankings(all_rankings, top_k=config.interpret.top_k_genes)
        consensus.to_csv(interp_dir / "consensus_biomarkers.csv")

        if config.interpret.run_pathway_enrichment:
            top_genes = consensus.head(config.interpret.top_k_genes).index.tolist()
            enrichment = discovery.pathway_enrichment(
                top_genes, list(features.columns),
            )
            enrichment.to_csv(interp_dir / "pathway_enrichment.csv")

        context["consensus_biomarkers"] = consensus

    context["interpretation_dir"] = str(interp_dir)
    logger.info("Interpretation complete")
    return context
