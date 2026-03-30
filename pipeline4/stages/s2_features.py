"""Stage 2: Feature engineering — transcriptomic, clinical, genomic features.

All fitted transforms (PCA, StandardScaler, variance thresholds) are computed
on training data only to prevent test-set leakage. The cohort split must
run before this stage.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_features(config: Any, context: Dict) -> Dict:
    """Build multi-modal feature matrix from expression + clinical data."""
    from pipeline4.features.transcriptomic import TranscriptomicFeatures
    from pipeline4.features.clinical_features import ClinicalFeatureBuilder
    from pipeline4.features.genomic import GenomicFeatures
    from pipeline4.features.multimodal import MultiModalFeatureBuilder

    expression = pd.read_parquet(context["expression_path"])
    clinical = pd.read_parquet(context["clinical_path"])

    # Get training indices for leakage-free fitting
    split_info = context["split_info"]
    train_idx = np.array(split_info["train"])

    features_dir = Path(config.base.data_dir) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    tx = TranscriptomicFeatures()

    # Variance filter + top genes (thresholds from train only)
    expr_filtered = tx.variance_filter(expression, config.features.variance_threshold, train_idx=train_idx)
    expr_filtered = tx.top_variable_genes(expr_filtered, config.features.n_top_genes, train_idx=train_idx)

    # Pathway scoring (z-scores fit on train only)
    # Try loading real MSigDB gene sets via legacy data_pipeline module
    pathway_scores = pd.DataFrame(index=expression.index)
    if config.features.pathway_scoring:
        gene_sets = None
        try:
            from data_pipeline import PathwayScorer as LegacyPathwayScorer
            legacy_scorer = LegacyPathwayScorer()
            loaded = legacy_scorer.load_gene_sets(collection="h.all")
            if loaded:
                gene_sets = loaded
                logger.info(f"Using MSigDB gene sets from legacy PathwayScorer ({len(loaded)} pathways)")
        except Exception as e:
            logger.debug(f"Legacy PathwayScorer unavailable, using built-in gene sets: {e}")
        pathway_scores = tx.pathway_scoring(expression, gene_sets=gene_sets, train_idx=train_idx)

    # PCA embedding (fit on train only)
    pca_emb, pca_model = tx.pca_embedding(expr_filtered, config.features.n_pca_components, train_idx=train_idx)
    pca_df = pd.DataFrame(
        pca_emb, index=expression.index,
        columns=[f"PC{i+1}" for i in range(pca_emb.shape[1])],
    )

    # Expression features = PCA + pathway scores
    expr_features = pd.concat([pca_df, pathway_scores], axis=1)

    # Clinical features (statistics fit on train only)
    clin_builder = ClinicalFeatureBuilder()
    clin_features = clin_builder.build(clinical, train_idx=train_idx)

    # Genomic features (synthetic for demo)
    genomic = GenomicFeatures()
    if config.ingest.demo_mode:
        gen_features = genomic.generate_synthetic_mutations(expression.index, config.base.seed)
        gen_features = genomic.encode_mutations(gen_features)
    else:
        gen_features = pd.DataFrame(index=expression.index)

    # Combine
    mm = MultiModalFeatureBuilder()
    combined = mm.combine(expr_features, clin_features, gen_features)
    modality_splits = mm.get_modality_splits(expr_features, clin_features, gen_features)
    combined = mm.normalize_modalities(combined, modality_splits, train_idx=train_idx)

    # Save
    combined.to_parquet(features_dir / "combined_features.parquet")
    expr_features.to_parquet(features_dir / "expression_features.parquet")
    clin_features.to_parquet(features_dir / "clinical_features.parquet")

    from pipeline4.utils.io import write_json
    write_json(modality_splits, str(features_dir / "modality_splits.json"))

    context["features_path"] = str(features_dir / "combined_features.parquet")
    context["modality_splits"] = modality_splits
    context["n_features"] = combined.shape[1]

    logger.info(f"Features complete: {combined.shape[1]} features across {len(modality_splits)} modalities")
    return context
