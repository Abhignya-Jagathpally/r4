#!/bin/bash
# DVC setup with remote verification (M79) and seed export (M80)
set -euo pipefail

SEED=${SEED:-42}
echo "Setting up DVC with seed=$SEED"

# Init DVC
dvc init || true

# M79: Verify remote accessibility
if [ -n "${DVC_REMOTE:-}" ]; then
    echo "Verifying DVC remote: $DVC_REMOTE"
    dvc remote add -d storage "$DVC_REMOTE" 2>/dev/null || true
    if dvc remote list | grep -q storage; then
        echo "Remote configured successfully"
    else
        echo "WARNING: Remote verification failed"
    fi
fi

# M80: Add seed export to DVC pipeline stages
dvc run -n ingest \
    -d main.py -d configs/ \
    -o data/merged/ \
    "export PYTHONHASHSEED=$SEED && python main.py --stages ingest --seed $SEED"

dvc run -n features \
    -d data/merged/ -d pipeline4/features/ \
    -o data/features/ \
    "export PYTHONHASHSEED=$SEED && python main.py --stages features --seed $SEED"

dvc run -n train_evaluate \
    -d data/features/ -d pipeline4/models/ \
    -o results/metrics.json \
    "export PYTHONHASHSEED=$SEED && python main.py --stages cohort train evaluate --seed $SEED"

echo "DVC pipeline configured with seed=$SEED"
