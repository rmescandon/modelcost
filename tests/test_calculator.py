from unittest.mock import MagicMock, patch

import pytest

from modelcost.calculator import (
    VALID_SOURCES,
    _compute,
    _tokencost_source,
    calculate_cost,
    list_models,
)
from modelcost.models import CostResult, SourceCost

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_PRICES = {
    "gpt-4o": {"prompt": 0.000003, "completion": 0.000015},
    "gpt-3.5-turbo": {"prompt": 0.0000005, "completion": 0.0000015},
}


# ---------------------------------------------------------------------------
# calculate_cost
# ---------------------------------------------------------------------------


class TestCalculateCost:
    def test_valid_single_source_returns_cost_result(self):
        with (
            patch(
                "modelcost.calculator.fetch_litellm_prices", return_value=FAKE_PRICES
            ),
            patch("modelcost.calculator.find_model", return_value=None),
        ):
            result = calculate_cost("gpt-4o", 100, 50, source="litellm")

        assert isinstance(result, CostResult)
        assert result.model == "gpt-4o"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.single_source is True
        assert len(result.sources) == 1
        assert result.sources[0].source == "litellm"

    def test_cost_calculation_is_correct(self):
        # prompt=3e-6, completion=15e-6 → 100*3e-6 + 50*15e-6 = 0.0003 + 0.00075 = 0.00105
        with (
            patch(
                "modelcost.calculator.fetch_litellm_prices", return_value=FAKE_PRICES
            ),
            patch("modelcost.calculator.find_model", return_value=None),
        ):
            result = calculate_cost("gpt-4o", 100, 50, source="litellm")

        s = result.sources[0]
        assert s.available is True
        assert s.total_cost_usd == pytest.approx(0.00105)
        assert s.price_per_million_input == pytest.approx(3.0)
        assert s.price_per_million_output == pytest.approx(15.0)

    def test_invalid_source_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid source"):
            calculate_cost("gpt-4o", 100, 50, source="unknown")

    def test_all_source_returns_three_sources_in_order(self):
        litellm_src = SourceCost("litellm", 0.001, 1.0, 2.0)
        openrouter_src = SourceCost("openrouter", 0.002, 1.5, 3.0)
        tokencost_src = SourceCost("tokencost", 0.0015, 1.2, 2.5)

        with (
            patch("modelcost.calculator._compute") as mock_compute,
            patch("modelcost.calculator._tokencost_source", return_value=tokencost_src),
        ):
            # _compute is called for litellm and openrouter via ThreadPoolExecutor
            def compute_side_effect(name, fn, model, inp, out):
                if name == "litellm":
                    return litellm_src
                return openrouter_src

            mock_compute.side_effect = compute_side_effect

            result = calculate_cost("gpt-4o", 100, 50, source="all")

        assert result.single_source is False
        names = [s.source for s in result.sources]
        assert names == ["litellm", "openrouter", "tokencost"]

    def test_model_not_found_returns_unavailable_source(self):
        with (
            patch("modelcost.calculator.fetch_litellm_prices", return_value={}),
            patch("modelcost.calculator.find_model", return_value=None),
        ):
            result = calculate_cost("nonexistent-model", 100, 50, source="litellm")

        s = result.sources[0]
        assert s.available is False
        assert "not found" in s.error.lower()


# ---------------------------------------------------------------------------
# _compute
# ---------------------------------------------------------------------------


class TestCompute:
    def test_returns_correct_source_cost(self):
        fetch_fn = MagicMock(return_value=FAKE_PRICES)
        with patch("modelcost.calculator.find_model", return_value=None):
            result = _compute("litellm", fetch_fn, "gpt-4o", 200, 100)

        assert result.source == "litellm"
        assert result.available is True
        # 200 * 3e-6 + 100 * 15e-6 = 0.0006 + 0.0015 = 0.0021
        assert result.total_cost_usd == pytest.approx(0.0021)
        assert result.price_per_million_input == pytest.approx(3.0)
        assert result.price_per_million_output == pytest.approx(15.0)

    def test_model_not_found_returns_error_source_cost(self):
        fetch_fn = MagicMock(return_value={})
        with patch("modelcost.calculator.find_model", return_value=None):
            result = _compute("litellm", fetch_fn, "unknown-model", 100, 50)

        assert result.available is False
        assert "not found" in result.error.lower()
        assert result.source == "litellm"

    def test_fetch_exception_returns_error_source_cost(self):
        fetch_fn = MagicMock(side_effect=RuntimeError("network error"))
        result = _compute("litellm", fetch_fn, "gpt-4o", 100, 50)

        assert result.available is False
        assert "network error" in result.error
        assert result.source == "litellm"

    def test_openrouter_uses_find_model(self):
        prices = {"openai/gpt-4o": {"prompt": 0.000003, "completion": 0.000015}}
        fetch_fn = MagicMock(return_value=prices)
        found = {"prompt": 0.000003, "completion": 0.000015}

        with patch("modelcost.calculator.find_model", return_value=found):
            result = _compute("openrouter", fetch_fn, "gpt-4o", 100, 50)

        assert result.available is True

    def test_litellm_uses_dict_get_not_find_model(self):
        """litellm source must use prices.get(model), not find_model."""
        fetch_fn = MagicMock(return_value=FAKE_PRICES)
        with patch("modelcost.calculator.find_model") as mock_find:
            _compute("litellm", fetch_fn, "gpt-4o", 10, 5)
            mock_find.assert_not_called()


# ---------------------------------------------------------------------------
# _tokencost_source
# ---------------------------------------------------------------------------


class TestTokencostSource:
    def test_returns_correct_source_cost(self):
        fake_entry = {
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
        }

        with (
            patch(
                "modelcost.calculator.TOKEN_COSTS",
                {"gpt-4o": fake_entry},
                create=True,
            ),
            patch(
                "modelcost.calculator.calculate_cost_by_tokens",
                side_effect=lambda tokens, model, kind: (
                    tokens * (0.000003 if kind == "input" else 0.000015)
                ),
                create=True,
            ),
        ):
            # Patch the imports inside the function
            with patch.dict(
                "sys.modules",
                {
                    "tokencost.costs": MagicMock(
                        calculate_cost_by_tokens=lambda tokens, model, kind: (
                            tokens * (0.000003 if kind == "input" else 0.000015)
                        )
                    ),
                    "tokencost.constants": MagicMock(
                        TOKEN_COSTS={"gpt-4o": fake_entry}
                    ),
                },
            ):
                result = _tokencost_source("gpt-4o", 100, 50)

        assert result.source == "tokencost"
        assert result.available is True
        assert result.total_cost_usd == pytest.approx(100 * 0.000003 + 50 * 0.000015)
        assert result.price_per_million_input == pytest.approx(3.0)
        assert result.price_per_million_output == pytest.approx(15.0)

    def test_key_error_returns_error_source_cost(self):
        with patch.dict(
            "sys.modules",
            {
                "tokencost.costs": MagicMock(
                    calculate_cost_by_tokens=MagicMock(side_effect=KeyError("gpt-4o"))
                ),
                "tokencost.constants": MagicMock(TOKEN_COSTS={}),
            },
        ):
            result = _tokencost_source("gpt-4o", 100, 50)

        assert result.available is False
        assert result.source == "tokencost"
        assert "Model not found" in result.error

    def test_generic_exception_returns_error_source_cost(self):
        with patch.dict(
            "sys.modules",
            {
                "tokencost.costs": MagicMock(
                    calculate_cost_by_tokens=MagicMock(
                        side_effect=Exception("unexpected failure")
                    )
                ),
                "tokencost.constants": MagicMock(TOKEN_COSTS={}),
            },
        ):
            result = _tokencost_source("gpt-4o", 100, 50)

        assert result.available is False
        assert "unexpected failure" in result.error


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_litellm_returns_sorted_list(self):
        with patch(
            "modelcost.calculator.fetch_litellm_prices",
            return_value={"z-model": {}, "a-model": {}, "m-model": {}},
        ):
            models = list_models("litellm")

        assert models == ["a-model", "m-model", "z-model"]

    def test_openrouter_returns_sorted_list(self):
        with patch(
            "modelcost.calculator.fetch_openrouter_prices",
            return_value={"z-model": {}, "a-model": {}},
        ):
            models = list_models("openrouter")

        assert models == ["a-model", "z-model"]

    def test_tokencost_returns_sorted_list(self):
        fake_costs = {"z-model": {}, "a-model": {}, "b-model": {}}
        with patch.dict(
            "sys.modules",
            {"tokencost.constants": MagicMock(TOKEN_COSTS=fake_costs)},
        ):
            models = list_models("tokencost")

        assert models == sorted(fake_costs.keys())

    def test_invalid_source_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid source"):
            list_models("unknown")

    def test_all_is_not_valid_for_list_models(self):
        """'all' is valid for calculate_cost but not for list_models."""
        with pytest.raises(ValueError):
            list_models("all")

    def test_valid_sources_constant(self):
        assert set(VALID_SOURCES) == {"litellm", "openrouter", "tokencost", "all"}
