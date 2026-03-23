"""I/O utility functions for data persistence."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def read_h5ad(path: str) -> "anndata.AnnData":
    """Read an h5ad file."""
    import anndata
    adata = anndata.read_h5ad(path)
    logger.info(f"Loaded h5ad: {adata.n_obs} obs x {adata.n_vars} vars from {path}")
    return adata


def read_parquet(path: str) -> pd.DataFrame:
    """Read a parquet file."""
    df = pd.read_parquet(path)
    logger.info(f"Loaded parquet: {df.shape} from {path}")
    return df


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to parquet with atomic write."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, engine="pyarrow")
    tmp.rename(out)
    logger.info(f"Wrote parquet: {df.shape} to {path}")


def write_json(data: Any, path: str, indent: int = 2) -> None:
    """Write data to JSON with atomic write."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=indent, cls=_NumpyEncoder)
    tmp.rename(out)
    logger.info(f"Wrote JSON to {path}")


def read_json(path: str) -> Any:
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)
