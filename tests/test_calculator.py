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

    @pytest.mark.parametrize(
        ("input_tokens", "output_tokens"),
        [(-1, 50), (100, -1), (-1, -1)],
    )
    def test_negative_tokens_raise_value_error(self, input_tokens, output_tokens):
        with pytest.raises(ValueError, match=r"must be >= 0"):
            calculate_cost("gpt-4o", input_tokens, output_tokens, source="litellm")

    @pytest.mark.parametrize(
        ("kwarg", "value"),
        [
            ("cached_input_tokens", -1),
            ("cache_creation_input_tokens", -1),
            ("reasoning_tokens", -1),
        ],
    )
    def test_negative_extended_tokens_raise_value_error(self, kwarg, value):
        with pytest.raises(ValueError, match=r"must be >= 0"):
            calculate_cost("gpt-4o", 100, 50, source="litellm", **{kwarg: value})

    def test_all_source_returns_three_sources_in_order(self):
        litellm_src = SourceCost("litellm", 0.001, 1.0, 2.0)
        openrouter_src = SourceCost("openrouter", 0.002, 1.5, 3.0)
        tokencost_src = SourceCost("tokencost", 0.0015, 1.2, 2.5)

        with (
            patch("modelcost.calculator._compute") as mock_compute,
            patch("modelcost.calculator._tokencost_source", return_value=tokencost_src),
        ):
            # _compute is called for litellm and openrouter via ThreadPoolExecutor
            def compute_side_effect(name, fn, model, inp, out, **kwargs):
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


# ---------------------------------------------------------------------------
# Extended pricing: cache + reasoning tokens
# ---------------------------------------------------------------------------

FAKE_PRICES_WITH_EXTRAS = {
    "gpt-4.1-mini": {
        "prompt": 4e-07,
        "completion": 1.6e-06,
        "cache_read": 1e-07,
        "cache_creation": 4.8e-07,
        "reasoning": 1.6e-06,
    },
}

FAKE_PRICES_NO_EXTRAS = {
    "gpt-4o": {"prompt": 0.000003, "completion": 0.000015},
}


class TestComputeExtendedPricing:
    """Tests for cached input, cache creation, and reasoning tokens in _compute."""

    def test_cost_with_all_token_types(self):
        """Full cost formula: input + cache_read + cache_creation + text_output + reasoning."""
        fetch_fn = MagicMock(return_value=FAKE_PRICES_WITH_EXTRAS)
        result = _compute(
            "litellm",
            fetch_fn,
            "gpt-4.1-mini",
            1000,  # input_tokens
            500,  # output_tokens
            cached_input_tokens=200,
            cache_creation_input_tokens=100,
            reasoning_tokens=150,
        )

        # cost = 1000*4e-7 + 200*1e-7 + 100*4.8e-7 + 350*1.6e-6 + 150*1.6e-6
        #      = 0.0004 + 0.00002 + 0.000048 + 0.00056 + 0.00024
        #      = 0.001268
        expected = (
            1000 * 4e-7
            + 200 * 1e-7
            + 100 * 4.8e-7
            + (500 - 150) * 1.6e-6
            + 150 * 1.6e-6
        )
        assert result.total_cost_usd == pytest.approx(expected)
        assert result.price_per_million_cache_read == pytest.approx(0.1)
        assert result.price_per_million_cache_creation == pytest.approx(0.48)
        assert result.price_per_million_reasoning == pytest.approx(1.6)

    def test_cache_read_fallback_to_prompt_rate(self):
        """When cache_read price is absent, fall back to prompt rate."""
        fetch_fn = MagicMock(return_value=FAKE_PRICES_NO_EXTRAS)
        result = _compute(
            "litellm",
            fetch_fn,
            "gpt-4o",
            1000,
            500,
            cached_input_tokens=200,
        )

        # cache_read falls back to prompt: 200 * 0.000003
        expected = 1000 * 0.000003 + 200 * 0.000003 + 500 * 0.000015
        assert result.total_cost_usd == pytest.approx(expected)
        assert result.price_per_million_cache_read is None

    def test_reasoning_fallback_to_completion_rate(self):
        """When reasoning price is absent, fall back to completion rate."""
        fetch_fn = MagicMock(return_value=FAKE_PRICES_NO_EXTRAS)
        result = _compute(
            "litellm",
            fetch_fn,
            "gpt-4o",
            1000,
            500,
            reasoning_tokens=200,
        )

        # reasoning falls back to completion: 200 * 0.000015
        # text output: (500-200) * 0.000015
        expected = 1000 * 0.000003 + 300 * 0.000015 + 200 * 0.000015
        assert result.total_cost_usd == pytest.approx(expected)
        assert result.price_per_million_reasoning is None

    def test_reasoning_tokens_clamped_to_output_tokens(self):
        """reasoning_tokens > output_tokens should be clamped."""
        fetch_fn = MagicMock(return_value=FAKE_PRICES_WITH_EXTRAS)
        result = _compute(
            "litellm",
            fetch_fn,
            "gpt-4.1-mini",
            1000,
            100,  # output_tokens
            reasoning_tokens=500,  # more than output
        )

        # effective_reasoning = min(500, 100) = 100, text_output = 0
        expected = 1000 * 4e-7 + 0 * 1.6e-6 + 100 * 1.6e-6
        assert result.total_cost_usd == pytest.approx(expected)

    def test_backward_compat_no_new_params(self):
        """Calling _compute without new params produces same result as before."""
        fetch_fn = MagicMock(return_value=FAKE_PRICES_NO_EXTRAS)
        result = _compute("litellm", fetch_fn, "gpt-4o", 200, 100)

        expected = 200 * 0.000003 + 100 * 0.000015
        assert result.total_cost_usd == pytest.approx(expected)
        assert result.price_per_million_input == pytest.approx(3.0)
        assert result.price_per_million_output == pytest.approx(15.0)
        assert result.price_per_million_cache_read is None
        assert result.price_per_million_cache_creation is None
        assert result.price_per_million_reasoning is None

    def test_extended_fields_populated_only_when_present(self):
        """Price-per-million fields for cache/reasoning only set when source has them."""
        fetch_fn = MagicMock(return_value=FAKE_PRICES_WITH_EXTRAS)
        result = _compute("litellm", fetch_fn, "gpt-4.1-mini", 100, 50)

        assert result.price_per_million_cache_read is not None
        assert result.price_per_million_cache_creation is not None
        assert result.price_per_million_reasoning is not None


class TestCalculateCostExtended:
    """Tests for calculate_cost with the new token parameters."""

    def test_new_params_threaded_through(self):
        """New parameters reach _compute via calculate_cost."""
        with patch(
            "modelcost.calculator.fetch_litellm_prices",
            return_value=FAKE_PRICES_WITH_EXTRAS,
        ):
            result = calculate_cost(
                "gpt-4.1-mini",
                1000,
                500,
                cached_input_tokens=200,
                cache_creation_input_tokens=100,
                reasoning_tokens=150,
                source="litellm",
            )

        s = result.sources[0]
        expected = (
            1000 * 4e-7
            + 200 * 1e-7
            + 100 * 4.8e-7
            + (500 - 150) * 1.6e-6
            + 150 * 1.6e-6
        )
        assert s.total_cost_usd == pytest.approx(expected)

    def test_backward_compat_calculate_cost(self):
        """Existing call without new params returns identical result."""
        with patch(
            "modelcost.calculator.fetch_litellm_prices",
            return_value=FAKE_PRICES_NO_EXTRAS,
        ):
            result = calculate_cost("gpt-4o", 100, 50, source="litellm")

        s = result.sources[0]
        expected = 100 * 0.000003 + 50 * 0.000015
        assert s.total_cost_usd == pytest.approx(expected)

    def test_token_counts_stored_in_cost_result(self):
        """CostResult must carry the token counts for output."""
        with patch(
            "modelcost.calculator.fetch_litellm_prices",
            return_value=FAKE_PRICES_WITH_EXTRAS,
        ):
            result = calculate_cost(
                "gpt-4.1-mini",
                1000,
                500,
                cached_input_tokens=200,
                cache_creation_input_tokens=100,
                reasoning_tokens=150,
                source="litellm",
            )

        assert result.cached_input_tokens == 200
        assert result.cache_creation_input_tokens == 100
        assert result.reasoning_tokens == 150

    def test_token_counts_default_zero_in_cost_result(self):
        """Without new params, CostResult token counts are 0."""
        with patch(
            "modelcost.calculator.fetch_litellm_prices",
            return_value=FAKE_PRICES_NO_EXTRAS,
        ):
            result = calculate_cost("gpt-4o", 100, 50, source="litellm")

        assert result.cached_input_tokens == 0
        assert result.cache_creation_input_tokens == 0
        assert result.reasoning_tokens == 0


# ---------------------------------------------------------------------------
# Real-world model pricing patterns
# ---------------------------------------------------------------------------

COHERE_EMBED_PRICES = {
    "azure_ai/Cohere-embed-v3-multilingual": {
        "prompt": 1e-07,
        "completion": 0.0,
    },
}

XAI_GROK_PRICES = {
    "xai/grok-3": {
        "prompt": 3e-06,
        "completion": 1.5e-05,
        "cache_read": 7.5e-07,
    },
}

DEEPSEEK_R1_PRICES = {
    "deepseek/deepseek-r1": {
        "prompt": 5.5e-07,
        "completion": 2.19e-06,
        "cache_read": 1.4e-07,
    },
}

DEEPSEEK_R1_REPLICATE_PRICES = {
    "replicate/deepseek-ai/deepseek-r1": {
        "prompt": 3.75e-06,
        "completion": 1e-05,
        "reasoning": 1e-05,
    },
}


class TestRealWorldPricingPatterns:
    """Cost calculations with realistic model pricing structures."""

    def test_cohere_embedding_zero_output_cost(self):
        """Embedding model: output_cost=0, only input tokens matter."""
        fetch_fn = MagicMock(return_value=COHERE_EMBED_PRICES)
        result = _compute(
            "litellm",
            fetch_fn,
            "azure_ai/Cohere-embed-v3-multilingual",
            512,
            0,
        )

        expected = 512 * 1e-07 + 0 * 0.0
        assert result.total_cost_usd == pytest.approx(expected)
        assert result.price_per_million_input == pytest.approx(0.1)
        assert result.price_per_million_output == pytest.approx(0.0)

    def test_cohere_embedding_nonzero_output_still_zero_cost(self):
        """Edge case: embedding called with output_tokens > 0 but rate is 0."""
        fetch_fn = MagicMock(return_value=COHERE_EMBED_PRICES)
        result = _compute(
            "litellm",
            fetch_fn,
            "azure_ai/Cohere-embed-v3-multilingual",
            512,
            128,
        )

        expected = 512 * 1e-07 + 128 * 0.0
        assert result.total_cost_usd == pytest.approx(expected)

    def test_xai_grok_with_cached_tokens(self):
        """xAI/Grok: cache_read=25% of input, no reasoning pricing."""
        fetch_fn = MagicMock(return_value=XAI_GROK_PRICES)
        result = _compute(
            "litellm",
            fetch_fn,
            "xai/grok-3",
            5000,
            2000,
            cached_input_tokens=3000,
        )

        # input: 5000 * 3e-6, cache: 3000 * 7.5e-7, output: 2000 * 1.5e-5
        expected = 5000 * 3e-06 + 3000 * 7.5e-07 + 2000 * 1.5e-05
        assert result.total_cost_usd == pytest.approx(expected)
        assert result.price_per_million_cache_read == pytest.approx(0.75)
        assert result.price_per_million_reasoning is None

    def test_xai_grok_reasoning_falls_back_to_completion(self):
        """xAI/Grok has no reasoning price; should fall back to completion rate."""
        fetch_fn = MagicMock(return_value=XAI_GROK_PRICES)
        result = _compute(
            "litellm",
            fetch_fn,
            "xai/grok-3",
            1000,
            500,
            reasoning_tokens=200,
        )

        # reasoning falls back to completion: both at 1.5e-5
        expected = 1000 * 3e-06 + 300 * 1.5e-05 + 200 * 1.5e-05
        assert result.total_cost_usd == pytest.approx(expected)

    def test_deepseek_r1_cache_hit_pricing(self):
        """DeepSeek R1: cache_read from cache_hit fallback (25% of input)."""
        fetch_fn = MagicMock(return_value=DEEPSEEK_R1_PRICES)
        result = _compute(
            "litellm",
            fetch_fn,
            "deepseek/deepseek-r1",
            2000,
            1000,
            cached_input_tokens=8000,
        )

        # input: 2000 * 5.5e-7, cache: 8000 * 1.4e-7, output: 1000 * 2.19e-6
        expected = 2000 * 5.5e-07 + 8000 * 1.4e-07 + 1000 * 2.19e-06
        assert result.total_cost_usd == pytest.approx(expected)
        assert result.price_per_million_cache_read == pytest.approx(0.14)

    def test_deepseek_r1_replicate_with_reasoning(self):
        """DeepSeek R1 on replicate: reasoning=output rate, no cache pricing."""
        fetch_fn = MagicMock(return_value=DEEPSEEK_R1_REPLICATE_PRICES)
        result = _compute(
            "litellm",
            fetch_fn,
            "replicate/deepseek-ai/deepseek-r1",
            5000,
            10000,
            reasoning_tokens=7000,
        )

        # input: 5000*3.75e-6, text: 3000*1e-5, reasoning: 7000*1e-5
        expected = 5000 * 3.75e-06 + 3000 * 1e-05 + 7000 * 1e-05
        assert result.total_cost_usd == pytest.approx(expected)
        assert result.price_per_million_reasoning == pytest.approx(10.0)
        assert result.price_per_million_cache_read is None

    def test_deepseek_r1_cache_and_reasoning_combined(self):
        """Full scenario: cached input + reasoning on a model that has both."""
        # Hypothetical model with all pricing fields
        prices = {
            "full-model": {
                "prompt": 5.5e-07,
                "completion": 2.19e-06,
                "cache_read": 1.4e-07,
                "reasoning": 3e-06,
            },
        }
        fetch_fn = MagicMock(return_value=prices)
        result = _compute(
            "litellm",
            fetch_fn,
            "full-model",
            2000,  # input
            5000,  # output
            cached_input_tokens=8000,
            reasoning_tokens=3000,
        )

        # input: 2000*5.5e-7, cache: 8000*1.4e-7, text_out: 2000*2.19e-6, reasoning: 3000*3e-6
        expected = (
            2000 * 5.5e-07 + 8000 * 1.4e-07 + (5000 - 3000) * 2.19e-06 + 3000 * 3e-06
        )
        assert result.total_cost_usd == pytest.approx(expected)
