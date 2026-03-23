#!/usr/bin/env nextflow
// M74: Correct tuple cardinality, M81: PYTHONHASHSEED

nextflow.enable.dsl=2

params.config_dir = './configs'
params.seed = 42
params.outdir = './results'

// M74: Fixed tuple cardinality between processes
process INGEST {
    publishDir "${params.outdir}", mode: 'copy'

    output:
    path "data/merged/expression.parquet", emit: expression
    path "data/merged/clinical.parquet", emit: clinical

    script:
    // M81: PYTHONHASHSEED in all process scripts
    """
    export PYTHONHASHSEED=${params.seed}
    python main.py --stages ingest --seed ${params.seed}
    """
}

process FEATURES {
    input:
    path expression
    path clinical

    output:
    path "data/features/combined_features.parquet", emit: features

    script:
    """
    export PYTHONHASHSEED=${params.seed}
    python main.py --stages features --seed ${params.seed}
    """
}

process TRAIN_EVALUATE {
    input:
    path features

    output:
    path "results/metrics.json", emit: metrics

    script:
    """
    export PYTHONHASHSEED=${params.seed}
    python main.py --stages cohort train evaluate --seed ${params.seed}
    """
}

process REPORT {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path metrics

    output:
    path "results/report.html"

    script:
    """
    export PYTHONHASHSEED=${params.seed}
    python main.py --stages interpret report --seed ${params.seed}
    """
}

workflow {
    INGEST()
    FEATURES(INGEST.out.expression, INGEST.out.clinical)
    TRAIN_EVALUATE(FEATURES.out.features)
    REPORT(TRAIN_EVALUATE.out.metrics)
}
