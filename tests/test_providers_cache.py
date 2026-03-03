import json
import time

import pytest

import modelcost.providers.cache as cache_module
from modelcost.providers.cache import load_cache, save_cache


@pytest.fixture(autouse=True)
def patch_cache_file(tmp_path, monkeypatch):
    """Redirect CACHE_FILE to a temp directory for every test."""
    tmp_cache = tmp_path / ".modelcost_cache.json"
    monkeypatch.setattr(cache_module, "CACHE_FILE", tmp_cache)
    return tmp_cache


class TestLoadCache:
    def test_returns_empty_dict_when_file_does_not_exist(self):
        result = load_cache("openrouter")
        assert result == {}

    def test_returns_models_when_within_ttl(self, tmp_path):
        models = {"openai/gpt-4o": {"prompt": 0.000003, "completion": 0.000015}}
        data = {"openrouter": {"_ts": time.time(), "models": models}}
        cache_module.CACHE_FILE.write_text(json.dumps(data))

        result = load_cache("openrouter")
        assert result == models

    def test_returns_empty_dict_when_ttl_expired(self, tmp_path):
        models = {"openai/gpt-4o": {"prompt": 0.000003, "completion": 0.000015}}
        expired_ts = time.time() - cache_module.CACHE_TTL - 1
        data = {"openrouter": {"_ts": expired_ts, "models": models}}
        cache_module.CACHE_FILE.write_text(json.dumps(data))

        result = load_cache("openrouter")
        assert result == {}

    def test_returns_empty_dict_for_missing_namespace(self):
        data = {"other_namespace": {"_ts": time.time(), "models": {"m": {}}}}
        cache_module.CACHE_FILE.write_text(json.dumps(data))

        result = load_cache("openrouter")
        assert result == {}

    def test_returns_empty_dict_when_ts_is_missing(self):
        # Entry has no _ts → defaults to 0 → always expired
        data = {"openrouter": {"models": {"m": {}}}}
        cache_module.CACHE_FILE.write_text(json.dumps(data))

        result = load_cache("openrouter")
        assert result == {}

    def test_different_namespaces_are_independent(self):
        litellm_models = {"gpt-4o": {"prompt": 0.000003, "completion": 0.000015}}
        openrouter_models = {
            "openai/gpt-4o": {"prompt": 0.000003, "completion": 0.000015}
        }
        now = time.time()
        data = {
            "litellm": {"_ts": now, "models": litellm_models},
            "openrouter": {"_ts": now, "models": openrouter_models},
        }
        cache_module.CACHE_FILE.write_text(json.dumps(data))

        assert load_cache("litellm") == litellm_models
        assert load_cache("openrouter") == openrouter_models


class TestSaveCache:
    def test_creates_file_if_not_exists(self):
        models = {"gpt-4o": {"prompt": 0.000003, "completion": 0.000015}}
        save_cache("litellm", models)

        assert cache_module.CACHE_FILE.exists()

    def test_written_data_has_correct_structure(self):
        models = {"gpt-4o": {"prompt": 0.000003, "completion": 0.000015}}
        before = time.time()
        save_cache("litellm", models)
        after = time.time()

        data = json.loads(cache_module.CACHE_FILE.read_text())
        assert "litellm" in data
        entry = data["litellm"]
        assert entry["models"] == models
        assert before <= entry["_ts"] <= after

    def test_preserves_existing_namespaces(self):
        existing = {"openrouter": {"_ts": time.time(), "models": {"m": {}}}}
        cache_module.CACHE_FILE.write_text(json.dumps(existing))

        new_models = {"gpt-4o": {"prompt": 0.000003, "completion": 0.000015}}
        save_cache("litellm", new_models)

        data = json.loads(cache_module.CACHE_FILE.read_text())
        assert "openrouter" in data
        assert "litellm" in data

    def test_overwrites_existing_namespace(self):
        old_models = {"old-model": {}}
        data = {"litellm": {"_ts": time.time() - 1000, "models": old_models}}
        cache_module.CACHE_FILE.write_text(json.dumps(data))

        new_models = {"new-model": {"prompt": 0.000001, "completion": 0.000002}}
        save_cache("litellm", new_models)

        data = json.loads(cache_module.CACHE_FILE.read_text())
        assert data["litellm"]["models"] == new_models

    def test_round_trip_load_after_save(self):
        models = {"gpt-4o": {"prompt": 0.000003, "completion": 0.000015}}
        save_cache("litellm", models)
        result = load_cache("litellm")
        assert result == models
