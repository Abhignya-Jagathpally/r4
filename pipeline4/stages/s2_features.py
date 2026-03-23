"""Stage 2: Feature engineering — transcriptomic, clinical, genomic features."""

import logging
from pathlib import Path
from typing import Any, Dict

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

    features_dir = Path(config.base.data_dir) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    tx = TranscriptomicFeatures()

    # Variance filter + top genes
    expr_filtered = tx.variance_filter(expression, config.features.variance_threshold)
    expr_filtered = tx.top_variable_genes(expr_filtered, config.features.n_top_genes)

    # Pathway scoring
    pathway_scores = pd.DataFrame(index=expression.index)
    if config.features.pathway_scoring:
        pathway_scores = tx.pathway_scoring(expression)

    # PCA embedding
    pca_emb, pca_model = tx.pca_embedding(expr_filtered, config.features.n_pca_components)
    pca_df = pd.DataFrame(
        pca_emb, index=expression.index,
        columns=[f"PC{i+1}" for i in range(pca_emb.shape[1])],
    )

    # Expression features = PCA + pathway scores
    expr_features = pd.concat([pca_df, pathway_scores], axis=1)

    # Clinical features
    clin_builder = ClinicalFeatureBuilder()
    clin_features = clin_builder.build(clinical)

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
    combined = mm.normalize_modalities(combined, modality_splits)

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
