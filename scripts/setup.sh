#!/usr/bin/env bash
# Setup script for R4-MM-Clinical pipeline
# Usage: bash scripts/setup.sh [--demo]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== R4-MM-Clinical Pipeline Setup ==="

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists"
fi

source venv/bin/activate

# 2. Install dependencies
echo "[2/4] Installing dependencies..."
pip install --upgrade pip -q
pip install -e ".[dev]" -q
echo "  Installed $(pip list --format=freeze | wc -l) packages"

# 3. Create data directories
echo "[3/4] Creating data directories..."
mkdir -p data/{raw,merged,features,splits}
mkdir -p results/{figures,interpretation,tables}
mkdir -p checkpoints logs

# 4. Download or generate data
if [[ "${1:-}" == "--demo" ]]; then
    echo "[4/4] Generating synthetic demo data..."
    python scripts/download_data.py --demo
else
    echo "[4/4] Downloading data from GEO..."
    python scripts/download_data.py
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the environment:  source venv/bin/activate"
echo "Run the full pipeline:     python main.py --stages all"
echo "Run in demo mode:          python main.py --stages all  (set demo_mode=true in configs/ingest.json)"
echo "Run tests:                 pytest"
