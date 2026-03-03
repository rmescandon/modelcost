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
