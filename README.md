# Multiple Myeloma Proteomics Pipeline

Complete end-to-end machine learning pipeline for biomarker ranking, pathway dysregulation analysis, and drug sensitivity prediction in Multiple Myeloma using proteomics data.

## Project Overview

This pipeline integrates:
- **PXD019126**: High-resolution mass spectrometry proteomics dataset from MM patient samples
- **DepMap**: Genome-wide CRISPR dependency screens for MM cell lines

The goal is to:
1. Identify and rank MM biomarkers
2. Detect dysregulated pathways
3. Prioritize drug targets
4. Predict drug sensitivity from proteomics

## Architecture Overview

```
Data Acquisition
    ↓
[PXD019126] [DepMap]
    ↓
Data Preprocessing (LOCKED)
    ├─ Protein ID Mapping (UniProt)
    ├─ Missing Data Imputation (kNN)
    ├─ Normalization (log2 + ComBat)
    └─ Pathway Aggregation (Hallmark)
    ↓
Feature Engineering (TUNABLE)
    ├─ Pathway scoring methods
    ├─ Feature selection
    └─ Dimensionality reduction
    ↓
Model Training (TUNABLE)
    ├─ Classical Models
    │  ├─ Lasso
    │  ├─ Random Forest
    │  └─ XGBoost
    ├─ Graph Neural Networks
    │  ├─ GAT (Graph Attention)
    │  └─ GCN (Graph Convolutional)
    └─ Advanced Models
    ↓
Benchmark Evaluation
    ├─ Train/Val/Test splits (patient-level)
    ├─ Cross-validation
    └─ Metrics (AUC, F1, precision, recall)
    ↓
Results & Reporting
    ├─ Model comparison
    ├─ Pathway importance
    └─ Drug target predictions
```

## Data Pipeline

### Stages

1. **Download Raw Data** (`download_raw`)
   - Fetch PXD019126 from ProteomXchange
   - Download DepMap expression data
   - Aggregate metadata

2. **Protein ID Mapping** (`uniprot_map`)
   - Map all protein identifiers to UniProt
   - Generate mapping report

3. **Missing Data Handling** (`handle_missingness`)
   - Classify missing mechanisms (MCAR, MAR, MNAR)
   - kNN imputation (k=5)
   - 50% missingness threshold per feature

4. **Normalization** (`normalize`)
   - Log2 transformation
   - ComBat batch correction (fitted on training data only)
   - Save scalers for test set application

5. **Pathway Aggregation** (`aggregate_pathways`)
   - Map proteins to Hallmark pathways
   - Compute pathway scores (mean aggregation)
   - ~50 pathway features

6. **Quality Control** (`quality_report`)
   - Distribution plots
   - Correlation heatmaps
   - Missingness patterns
   - HTML report generation

## Model Layers

### Baseline Models

Classical machine learning baselines for comparison:

- **Lasso**: Linear model with L1 regularization
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with optimized regularization

Configuration: See `configs/base_config.yaml`

### Graph Neural Networks

Protein interaction-aware models:

- **GAT** (Graph Attention Networks)
  - Multi-head self-attention over protein interactions
  - 4 heads, 3 layers, 64 hidden dimensions

- **GCN** (Graph Convolutional Networks)
  - Spectral convolutions over interaction graph
  - 3 layers, 64 hidden dimensions

Graph construction: STRING protein-protein interaction network (medium confidence cutoff)

### Agentic Tuning

Autonomous hyperparameter optimization implementing Karpathy's autoresearch pattern:

- **Single Primary Metric**: ROC-AUC maximization
- **Fixed Budget**: 3600 seconds or 50 experiments
- **Constrained Search Space**:
  - **LOCKED**: All preprocessing (hash-verified)
  - **TUNABLE**: Model hyperparameters, feature engineering, training settings
- **Full Logging**: Every experiment tracked in MLflow

## Key Design Principles

1. **Classical First**: Baselines established before complex models
2. **Patient-Level Splits**: No data leakage via stratified splits on patient ID
3. **Frozen Preprocessing**: No changes to preprocessing once data is locked
4. **One Metric**: Single ROC-AUC objective for agentic tuning
5. **Fixed Budget**: Time-bound search (1 hour default)
6. **Full Reproducibility**: Deterministic execution, config hashing, environment logging

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo>
cd r4

# Create conda environment
conda create -y -n mm_proteomics python=3.10
conda activate mm_proteomics

# Install dependencies
pip install -r requirements.txt
pip install -r orchestration/requirements.txt
```

### Run Full Pipeline

#### Option 1: Snakemake

```bash
# Execute workflow
snakemake -j 4 --use-conda

# Run specific rule
snakemake -j 4 normalize

# Generate DAG
snakemake --dag | dot -Tpng > dag.png

# Dry run
snakemake -j 4 -n
```

#### Option 2: Nextflow

```bash
# Local execution
nextflow run orchestration/nextflow/main.nf -profile local

# SLURM cluster
nextflow run orchestration/nextflow/main.nf -profile slurm

# Docker containers
nextflow run orchestration/nextflow/main.nf -profile docker

# Generate report
nextflow run orchestration/nextflow/main.nf -with-report

# DAG visualization
nextflow run orchestration/nextflow/main.nf -with-dag dag.html
```

#### Option 3: DVC

```bash
# Initialize DVC
bash orchestration/dvc_setup.sh local .dvc_cache

# Execute pipeline
dvc repro

# Show pipeline status
dvc status

# Push data to remote
dvc push

# Pull data from remote
dvc pull
```

### Run Agentic Tuning

```python
from orchestration.agentic_tuning import AgenticTuner, SearchSpace
from orchestration.mlflow_config import MLflowConfig

# Define training function
def train_fn(config):
    model = create_model(config)
    return evaluate(model, val_data)

# Define search space
search_space = {
    'lr': SearchSpace('lr', 'float', low=1e-5, high=1e-2, log_scale=True),
    'hidden_dim': SearchSpace('hidden_dim', 'int', low=32, high=256),
}

# Create and run tuner
tuner = AgenticTuner(
    experiment_name='mm_proteomics_tuning',
    train_fn=train_fn,
    search_space=search_space,
    preprocessing_path='data_pipeline',
    budget_seconds=3600,
    budget_experiments=50,
)

best_config, best_metric = tuner.tune()
tuner.save_results('results/tuning_results.json')
```

### Reproducibility Setup

```python
from orchestration.reproducibility import setup_reproducibility

# Setup reproducible experiment
config = setup_reproducibility(
    experiment_id='mm_experiment_001',
    seed=42,
    deterministic=True,
    log_dir='logs',
)
```

## Configuration

### Base Config (`configs/base_config.yaml`)

Master configuration file with sections for:
- Data paths and sources
- Preprocessing parameters (FROZEN)
- Pathway configuration
- Model hyperparameters
- Orchestration settings
- Agentic tuning constraints

### Search Spaces (`configs/search_spaces.yaml`)

Hyperparameter distributions for tuning:
- Parameter ranges (float, int, categorical)
- Log-scale options
- Frozen vs. tunable parameters

Frozen parameters prevent data leakage:
```yaml
preprocessing:
  missingness_strategy:
    frozen: true  # Cannot change
  normalization_method:
    frozen: true  # Cannot change
```

## Experiment Tracking

### MLflow

Centralized experiment tracking with:
- Automatic parameter logging
- Metric tracking per iteration
- Model registry integration
- Artifact storage

```python
from orchestration.mlflow_config import get_mlflow_config

config = get_mlflow_config()
config.set_experiment('mm_proteomics')
config.enable_auto_logging()
```

### DVC

Data versioning and pipeline reproducibility:

```bash
# Add data to DVC
dvc add data/raw/

# Create pipeline
dvc stage add -n preprocess \
  -d data/raw/expression.h5ad \
  -o data/standardized/expression.parquet \
  python data_pipeline/preprocess.py
```

## Directory Structure

```
r4/
├── data/
│   ├── raw/                      # Original PXD019126, DepMap
│   ├── standardized/             # Mapped, imputed, normalized
│   └── analysis_ready/           # Pathway aggregated, features engineered
├── results/
│   ├── qc/                       # Quality control reports
│   ├── baselines/                # Baseline model results
│   ├── graph_ml/                 # Graph neural network results
│   ├── benchmark/                # Benchmark comparisons
│   └── final_report.html         # Complete results report
├── configs/
│   ├── base_config.yaml          # Master configuration
│   └── search_spaces.yaml        # Hyperparameter distributions
├── orchestration/
│   ├── Snakefile                 # Snakemake workflow
│   ├── nextflow/
│   │   ├── main.nf               # Nextflow pipeline
│   │   ├── modules.nf            # Process definitions
│   │   └── nextflow.config        # Nextflow config
│   ├── mlflow_config.py          # MLflow tracking setup
│   ├── dvc_setup.sh              # DVC initialization
│   ├── agentic_tuning.py         # Autonomous hyperparameter tuning
│   ├── parallel_compute.py       # Ray, Dask, GPU allocation
│   ├── reproducibility.py        # Reproducibility utilities
│   ├── Dockerfile                # Container image
│   ├── Apptainer.def             # HPC container
│   ├── requirements.txt           # Orchestration dependencies
│   └── tests/
│       └── test_orchestration.py  # Unit tests
├── data_pipeline/                # Data preprocessing (modules)
├── baselines/                    # Classical ML models
├── graph_ml/                     # Graph neural networks
├── depmap_benchmark/             # Benchmark evaluation
├── docs/                         # Documentation
├── notebooks/                    # Jupyter notebooks
├── literature/                   # Research papers
├── .dvc/                         # DVC configuration
├── .git/                         # Git repository
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC lock file
├── mlflow.db                     # MLflow tracking database
├── pyproject.toml                # Python project metadata
├── requirements.txt              # All dependencies
└── README.md                     # This file
```

## Computing Resources

### Local Machine
```bash
snakemake -j 4 --use-conda
```

### HPC (SLURM)
```bash
nextflow run orchestration/nextflow/main.nf -profile slurm
# Configurable in nextflow.config
```

### Docker
```bash
docker build -t proteomics-pipeline:latest -f orchestration/Dockerfile .
docker run --gpus all -v /data:/data proteomics-pipeline:latest snakemake -j 4
```

### Apptainer/Singularity (HPC)
```bash
apptainer build proteomics-pipeline.sif orchestration/Apptainer.def
apptainer run --nv proteomics-pipeline.sif snakemake -j 4
```

## Performance Benchmarks

Expected runtime on typical hardware:

| Stage | Time (hrs) | CPUs | Memory (GB) | Notes |
|-------|-----------|------|-------------|-------|
| Data Download | 1-2 | 2 | 4 | Depends on network |
| Preprocessing | 1-2 | 4 | 16 | Data quality dependent |
| Baselines | 2-4 | 8 | 16 | XGBoost dominates |
| Graph ML | 4-8 | 8 | 32 | GPU recommended |
| Benchmark | 1-2 | 4 | 8 | Evaluation only |
| Agentic Tuning | 1 | 4 | 8 | Fixed time budget |
| **Total** | **10-20** | - | - | Parallel execution possible |

## Citation

If you use this pipeline, please cite:

```bibtex
@misc{mm_proteomics_pipeline,
  title={Multiple Myeloma Proteomics Pipeline},
  author={Researcher 6},
  year={2024},
  howpublished={GitHub},
  url={https://github.com/...}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact the development team.

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in config
- Decrease number of workers
- Use `--keep-going` with Snakemake for partial completion

### GPU Not Detected
- Check CUDA installation: `nvidia-smi`
- Set `CUDA_VISIBLE_DEVICES`: `export CUDA_VISIBLE_DEVICES=0`
- Install PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### DVC Remote Issues
- Configure remote: `dvc remote add -d storage /path/to/storage`
- Verify remote: `dvc remote list`
- Push data: `dvc push`

### MLflow Connection Issues
- Check tracking URI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- Verify database permissions: `ls -la mlflow.db`

## Contributing

1. Create a feature branch: `git checkout -b feature/new-model`
2. Make changes and test: `pytest orchestration/tests/`
3. Commit with descriptive message
4. Push and create pull request

## Acknowledgments

- ProteomXchange for PXD019126 dataset
- DepMap for CRISPR dependency screens
- PyTorch, scikit-learn, XGBoost communities
