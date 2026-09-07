"""設定の読み込みと検証。"""

import pytest

from src import config
from src.config import CLOUD, LOCAL, ConfigError, cloud_requested_but_unavailable, settings


def reload(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config.settings.cache_clear()
    return settings()


def test_defaults_are_local_mode():
    current = settings()
    assert current.mode == LOCAL
    assert current.is_cloud is False


def test_default_rag_parameters():
    current = settings()
    assert (current.chunk_size, current.chunk_overlap, current.top_k) == (600, 100, 4)
    assert current.score_threshold == 0.15


def test_cloud_mode_requires_a_key(monkeypatch):
    assert reload(monkeypatch, APP_MODE="cloud").mode == LOCAL


def test_cloud_mode_activates_with_a_key(monkeypatch):
    assert reload(monkeypatch, APP_MODE="cloud", OPENAI_API_KEY="sk-x").mode == CLOUD


def test_cloud_requested_but_unavailable_is_detected(monkeypatch):
    monkeypatch.setenv("APP_MODE", "cloud")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cloud_requested_but_unavailable() is True


def test_unknown_mode_is_rejected(monkeypatch):
    with pytest.raises(ConfigError, match="APP_MODE"):
        reload(monkeypatch, APP_MODE="staging")


def test_overrides_are_read_from_the_environment(monkeypatch):
    current = reload(monkeypatch, CHUNK_SIZE="900", CHUNK_OVERLAP="50", TOP_K="7", TEMPERATURE="0.9")
    assert (current.chunk_size, current.chunk_overlap, current.top_k) == (900, 50, 7)
    assert current.temperature == 0.9


@pytest.mark.parametrize(
    "env",
    [
        {"CHUNK_SIZE": "abc"},
        {"CHUNK_SIZE": "10"},
        {"CHUNK_SIZE": "99999"},
        {"TOP_K": "0"},
        {"TOP_K": "999"},
        {"TEMPERATURE": "3.5"},
        {"TEMPERATURE": "not-a-number"},
        {"TIMEOUT_SECONDS": "0"},
        {"MAX_RETRIES": "-1"},
        {"SCORE_THRESHOLD": "2"},
        {"MAX_UPLOAD_MB": "0"},
    ],
)
def test_invalid_values_raise_a_readable_error(monkeypatch, env):
    with pytest.raises(ConfigError):
        reload(monkeypatch, **env)


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
        reload(monkeypatch, CHUNK_SIZE="200", CHUNK_OVERLAP="200")


def test_blank_value_falls_back_to_the_default(monkeypatch):
    assert reload(monkeypatch, TOP_K="").top_k == 4


def test_require_cloud_raises_in_local_mode():
    with pytest.raises(ConfigError, match="クラウドモード"):
        settings().require_cloud()


def test_require_cloud_passes_in_cloud_mode(cloud_mode):
    cloud_mode.require_cloud()
