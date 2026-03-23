"""Test checkpoint manager."""

import numpy as np
from pipeline4.checkpoint import CheckpointManager


def test_save_load_sklearn(tmp_path):
    from sklearn.linear_model import LinearRegression
    mgr = CheckpointManager(str(tmp_path), max_keep=3)
    model = LinearRegression()
    model.fit(np.random.randn(10, 3), np.random.randn(10))

    path = mgr.save(model, "lr", "run_001", epoch=0,
                     metrics={"c_index": 0.65}, config_hash="abc123")
    loaded = mgr.load(path)
    assert loaded["metrics"]["c_index"] == 0.65


def test_manifest(tmp_path):
    from sklearn.linear_model import LinearRegression
    mgr = CheckpointManager(str(tmp_path), max_keep=3)
    model = LinearRegression()
    model.fit(np.random.randn(10, 3), np.random.randn(10))

    for epoch in range(5):
        mgr.save(model, "lr", "run_002", epoch=epoch,
                 metrics={"c_index": 0.5 + epoch * 0.05}, config_hash="def456")

    runs = mgr.list_runs("lr")
    assert len(runs) == 1
    assert runs[0]["best_epoch"] == 4


def test_pruning(tmp_path):
    from sklearn.linear_model import LinearRegression
    mgr = CheckpointManager(str(tmp_path), max_keep=2)
    model = LinearRegression()
    model.fit(np.random.randn(10, 3), np.random.randn(10))

    for epoch in range(5):
        mgr.save(model, "lr", "run_003", epoch=epoch,
                 metrics={"c_index": 0.5 + epoch * 0.05}, config_hash="ghi789")

    manifest = mgr._load_manifest("lr", "run_003")
    assert len(manifest["checkpoints"]) <= 2
