from unittest.mock import MagicMock, patch

import httpx
import pytest

from modelcost.providers.litellm import fetch_litellm_prices


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


class TestFetchLitellmPrices:
    def test_parses_response_correctly(self):
        raw = {
            "gpt-4o": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
            "gpt-3.5-turbo": {
                "input_cost_per_token": 0.0000005,
                "output_cost_per_token": 0.0000015,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert "gpt-4o" in prices
        assert prices["gpt-4o"] == {"prompt": 0.000003, "completion": 0.000015}
        assert "gpt-3.5-turbo" in prices
        assert prices["gpt-3.5-turbo"] == {"prompt": 0.0000005, "completion": 0.0000015}

    def test_filters_models_missing_input_cost(self):
        raw = {
            "model-with-costs": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
            "model-missing-input": {"output_cost_per_token": 0.000015},
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert "model-with-costs" in prices
        assert "model-missing-input" not in prices

    def test_filters_models_missing_output_cost(self):
        raw = {
            "model-with-costs": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
            "model-missing-output": {"input_cost_per_token": 0.000003},
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert "model-with-costs" in prices
        assert "model-missing-output" not in prices

    def test_filters_models_missing_both_costs(self):
        raw = {
            "model-with-costs": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
            "model-no-costs": {"context_window": 128000},
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert "model-with-costs" in prices
        assert "model-no-costs" not in prices

    def test_empty_response_returns_empty_dict(self):
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response({})
        ):
            prices = fetch_litellm_prices()

        assert prices == {}

    def test_raises_on_http_error(self):
        with patch(
            "modelcost.providers.litellm.httpx.get",
            return_value=_make_response({}, status_code=500),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                fetch_litellm_prices()

    def test_calls_correct_url(self):
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response({})
        ) as mock_get:
            fetch_litellm_prices()

        call_url = mock_get.call_args[0][0]
        assert (
            "litellm" in call_url.lower()
            or "berriAI" in call_url
            or "github" in call_url
        )

    def test_extracts_cache_read_price(self):
        raw = {
            "gpt-4.1-mini": {
                "input_cost_per_token": 4e-07,
                "output_cost_per_token": 1.6e-06,
                "cache_read_input_token_cost": 1e-07,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert prices["gpt-4.1-mini"]["cache_read"] == 1e-07

    def test_extracts_cache_creation_price(self):
        raw = {
            "model-a": {
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
                "cache_creation_input_token_cost": 1.5e-06,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert prices["model-a"]["cache_creation"] == 1.5e-06

    def test_extracts_reasoning_price(self):
        raw = {
            "o3": {
                "input_cost_per_token": 1e-05,
                "output_cost_per_token": 4e-05,
                "output_cost_per_reasoning_token": 4e-05,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert prices["o3"]["reasoning"] == 4e-05

    def test_omits_extra_fields_when_absent(self):
        raw = {
            "gpt-4o": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert "cache_read" not in prices["gpt-4o"]
        assert "cache_creation" not in prices["gpt-4o"]
        assert "reasoning" not in prices["gpt-4o"]

    def test_extracts_all_extra_fields_together(self):
        raw = {
            "gemini-flash": {
                "input_cost_per_token": 2.5e-07,
                "output_cost_per_token": 1.5e-06,
                "cache_read_input_token_cost": 2.5e-08,
                "cache_creation_input_token_cost": 5e-07,
                "output_cost_per_reasoning_token": 1.5e-06,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        entry = prices["gemini-flash"]
        assert entry["prompt"] == 2.5e-07
        assert entry["completion"] == 1.5e-06
        assert entry["cache_read"] == 2.5e-08
        assert entry["cache_creation"] == 5e-07
        assert entry["reasoning"] == 1.5e-06

    def test_cache_hit_field_used_as_fallback_for_cache_read(self):
        """Models with input_cost_per_token_cache_hit but no cache_read_input_token_cost."""
        raw = {
            "deepseek/deepseek-r1": {
                "input_cost_per_token": 5.5e-07,
                "output_cost_per_token": 2.19e-06,
                "input_cost_per_token_cache_hit": 1.4e-07,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert prices["deepseek/deepseek-r1"]["cache_read"] == 1.4e-07

    def test_cache_read_takes_priority_over_cache_hit(self):
        """When both fields exist, cache_read_input_token_cost wins."""
        raw = {
            "deepseek/deepseek-chat": {
                "input_cost_per_token": 2.8e-07,
                "output_cost_per_token": 1.1e-06,
                "cache_read_input_token_cost": 2.8e-08,
                "input_cost_per_token_cache_hit": 2.8e-08,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert prices["deepseek/deepseek-chat"]["cache_read"] == 2.8e-08

    def test_no_cache_read_when_neither_field_present(self):
        """No cache_read key when neither cache_read nor cache_hit is present."""
        raw = {
            "gpt-4o": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert "cache_read" not in prices["gpt-4o"]

    def test_cohere_embedding_zero_output_cost(self):
        """Embedding models have output_cost=0; must be included with correct values."""
        raw = {
            "azure_ai/Cohere-embed-v3-multilingual": {
                "input_cost_per_token": 1e-07,
                "output_cost_per_token": 0.0,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        assert "azure_ai/Cohere-embed-v3-multilingual" in prices
        entry = prices["azure_ai/Cohere-embed-v3-multilingual"]
        assert entry["prompt"] == 1e-07
        assert entry["completion"] == 0.0
        assert "cache_read" not in entry
        assert "reasoning" not in entry

    def test_xai_grok_cache_read_only(self):
        """xAI/Grok models have cache_read but no cache_hit or reasoning."""
        raw = {
            "xai/grok-3": {
                "input_cost_per_token": 3e-06,
                "output_cost_per_token": 1.5e-05,
                "cache_read_input_token_cost": 7.5e-07,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        entry = prices["xai/grok-3"]
        assert entry["prompt"] == 3e-06
        assert entry["completion"] == 1.5e-05
        assert entry["cache_read"] == 7.5e-07
        assert "cache_creation" not in entry
        assert "reasoning" not in entry

    def test_deepseek_cache_hit_and_reasoning(self):
        """DeepSeek R1 on replicate: reasoning field + cache_hit fallback."""
        raw = {
            "deepseek/deepseek-r1": {
                "input_cost_per_token": 5.5e-07,
                "output_cost_per_token": 2.19e-06,
                "input_cost_per_token_cache_hit": 1.4e-07,
            },
            "replicate/deepseek-ai/deepseek-r1": {
                "input_cost_per_token": 3.75e-06,
                "output_cost_per_token": 1e-05,
                "output_cost_per_reasoning_token": 1e-05,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        ds = prices["deepseek/deepseek-r1"]
        assert ds["cache_read"] == 1.4e-07
        assert "reasoning" not in ds

        rep = prices["replicate/deepseek-ai/deepseek-r1"]
        assert rep["reasoning"] == 1e-05
        assert "cache_read" not in rep

    def test_rerank_model_zero_costs_included(self):
        """Rerank models with input=0 and output=0 pass the filter but yield $0."""
        raw = {
            "azure_ai/cohere-rerank-v3-english": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "input_cost_per_query": 0.002,
            },
        }
        with patch(
            "modelcost.providers.litellm.httpx.get", return_value=_make_response(raw)
        ):
            prices = fetch_litellm_prices()

        entry = prices["azure_ai/cohere-rerank-v3-english"]
        assert entry["prompt"] == 0.0
        assert entry["completion"] == 0.0
