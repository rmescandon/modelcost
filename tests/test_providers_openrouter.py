from unittest.mock import MagicMock, patch

import httpx
import pytest

from modelcost.providers.openrouter import fetch_openrouter_prices, find_model


def _make_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


FAKE_API_RESPONSE = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {
            "id": "anthropic/claude-3-5-sonnet",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
    ]
}


class TestFetchOpenrouterPrices:
    def test_uses_cache_when_available(self):
        cached = {"openai/gpt-4o": {"prompt": 0.000003, "completion": 0.000015}}
        with (
            patch(
                "modelcost.providers.openrouter.load_cache", return_value=cached
            ) as mock_cache,
            patch("modelcost.providers.openrouter.httpx.get") as mock_get,
        ):
            result = fetch_openrouter_prices(use_cache=True)

        mock_cache.assert_called_once_with("openrouter")
        mock_get.assert_not_called()
        assert result == cached

    def test_calls_api_when_cache_empty(self):
        with (
            patch("modelcost.providers.openrouter.load_cache", return_value={}),
            patch(
                "modelcost.providers.openrouter.httpx.get",
                return_value=_make_response(FAKE_API_RESPONSE),
            ),
            patch("modelcost.providers.openrouter.save_cache") as mock_save,
        ):
            result = fetch_openrouter_prices(use_cache=True)

        assert "openai/gpt-4o" in result
        assert "anthropic/claude-3-5-sonnet" in result
        mock_save.assert_called_once()

    def test_skips_cache_when_use_cache_false(self):
        with (
            patch("modelcost.providers.openrouter.load_cache") as mock_cache,
            patch(
                "modelcost.providers.openrouter.httpx.get",
                return_value=_make_response(FAKE_API_RESPONSE),
            ),
            patch("modelcost.providers.openrouter.save_cache"),
        ):
            fetch_openrouter_prices(use_cache=False)

        mock_cache.assert_not_called()

    def test_parses_pricing_correctly(self):
        with (
            patch("modelcost.providers.openrouter.load_cache", return_value={}),
            patch(
                "modelcost.providers.openrouter.httpx.get",
                return_value=_make_response(FAKE_API_RESPONSE),
            ),
            patch("modelcost.providers.openrouter.save_cache"),
        ):
            result = fetch_openrouter_prices(use_cache=False)

        assert result["openai/gpt-4o"] == {"prompt": 0.000003, "completion": 0.000015}

    def test_filters_models_without_pricing(self):
        response = {
            "data": [
                {
                    "id": "model-with-pricing",
                    "pricing": {"prompt": "0.001", "completion": "0.002"},
                },
                {"id": "model-without-pricing"},
            ]
        }
        with (
            patch("modelcost.providers.openrouter.load_cache", return_value={}),
            patch(
                "modelcost.providers.openrouter.httpx.get",
                return_value=_make_response(response),
            ),
            patch("modelcost.providers.openrouter.save_cache"),
        ):
            result = fetch_openrouter_prices(use_cache=False)

        assert "model-with-pricing" in result
        assert "model-without-pricing" not in result

    def test_saves_cache_after_fetch(self):
        with (
            patch("modelcost.providers.openrouter.load_cache", return_value={}),
            patch(
                "modelcost.providers.openrouter.httpx.get",
                return_value=_make_response(FAKE_API_RESPONSE),
            ),
            patch("modelcost.providers.openrouter.save_cache") as mock_save,
        ):
            result = fetch_openrouter_prices(use_cache=True)

        mock_save.assert_called_once_with("openrouter", result)

    def test_raises_on_http_error(self):
        with (
            patch("modelcost.providers.openrouter.load_cache", return_value={}),
            patch(
                "modelcost.providers.openrouter.httpx.get",
                return_value=_make_response({}, status_code=429),
            ),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                fetch_openrouter_prices(use_cache=False)


class TestFindModel:
    def _prices(self):
        return {
            "openai/gpt-4o": {"prompt": 0.000003, "completion": 0.000015},
            "anthropic/claude-3-5-sonnet": {"prompt": 0.000003, "completion": 0.000015},
            "meta-llama/llama-3.1-8b-instruct": {
                "prompt": 0.0000001,
                "completion": 0.0000002,
            },
        }

    def test_exact_match_returns_correct_dict(self):
        prices = self._prices()
        result = find_model("openai/gpt-4o", prices)
        assert result == prices["openai/gpt-4o"]

    def test_suffix_match_single_result(self):
        prices = self._prices()
        result = find_model("gpt-4o", prices)
        assert result == prices["openai/gpt-4o"]

    def test_suffix_match_multiple_results_returns_none(self):
        prices = {
            "openai/gpt-4o": {"prompt": 0.000003, "completion": 0.000015},
            "azure/gpt-4o": {"prompt": 0.000004, "completion": 0.000016},
        }
        result = find_model("gpt-4o", prices)
        assert result is None

    def test_no_match_returns_none(self):
        prices = self._prices()
        result = find_model("nonexistent-model", prices)
        assert result is None

    def test_empty_prices_returns_none(self):
        result = find_model("gpt-4o", {})
        assert result is None

    def test_exact_match_takes_priority_over_suffix(self):
        """If exact key exists, it is returned directly without suffix scan."""
        prices = {
            "gpt-4o": {"prompt": 0.000001, "completion": 0.000002},
            "openai/gpt-4o": {"prompt": 0.000003, "completion": 0.000015},
        }
        result = find_model("gpt-4o", prices)
        assert result == {"prompt": 0.000001, "completion": 0.000002}
