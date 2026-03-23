"""Test Pydantic configuration loading."""

from pipeline4.config import PipelineConfig


def test_default_config():
    config = PipelineConfig()
    assert config.base.seed == 42
    assert config.train.n_epochs == 100
    assert len(config.train.enabled_models) > 0


def test_from_dir(tmp_config_dir):
    config = PipelineConfig.from_dir(tmp_config_dir)
    assert config.base.device == "cpu"
    assert config.ingest.demo_mode is True
    assert config.train.n_epochs == 5


def test_config_round_trip(tmp_path):
    config = PipelineConfig()
    config.save(str(tmp_path / "out"))
    loaded = PipelineConfig.from_dir(str(tmp_path / "out"))
    assert loaded.base.seed == config.base.seed
    assert loaded.train.enabled_models == config.train.enabled_models
