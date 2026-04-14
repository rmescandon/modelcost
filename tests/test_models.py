from modelcost.models import CostResult, SourceCost


class TestSourceCost:
    def test_available_true_when_cost_is_float(self):
        s = SourceCost(
            source="litellm",
            total_cost_usd=0.005,
            price_per_million_input=3.0,
            price_per_million_output=15.0,
        )
        assert s.available is True

    def test_available_false_when_cost_is_none(self):
        s = SourceCost(
            source="litellm",
            total_cost_usd=None,
            price_per_million_input=None,
            price_per_million_output=None,
            error="Model not found",
        )
        assert s.available is False

    def test_available_true_when_cost_is_zero(self):
        s = SourceCost(
            source="litellm",
            total_cost_usd=0.0,
            price_per_million_input=0.0,
            price_per_million_output=0.0,
        )
        assert s.available is True

    def test_error_defaults_to_none(self):
        s = SourceCost(
            source="litellm",
            total_cost_usd=0.001,
            price_per_million_input=1.0,
            price_per_million_output=2.0,
        )
        assert s.error is None

    def test_new_price_fields_default_to_none(self):
        s = SourceCost(
            source="litellm",
            total_cost_usd=0.001,
            price_per_million_input=1.0,
            price_per_million_output=2.0,
        )
        assert s.price_per_million_cache_read is None
        assert s.price_per_million_cache_creation is None
        assert s.price_per_million_reasoning is None

    def test_new_price_fields_can_be_set(self):
        s = SourceCost(
            source="litellm",
            total_cost_usd=0.001,
            price_per_million_input=1.0,
            price_per_million_output=2.0,
            price_per_million_cache_read=0.25,
            price_per_million_cache_creation=1.5,
            price_per_million_reasoning=3.0,
        )
        assert s.price_per_million_cache_read == 0.25
        assert s.price_per_million_cache_creation == 1.5
        assert s.price_per_million_reasoning == 3.0


class TestCostResult:
    def _make_result(self, sources):
        return CostResult(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            sources=sources,
        )

    def test_available_sources_returns_only_available(self):
        s1 = SourceCost("litellm", 0.005, 3.0, 15.0)
        s2 = SourceCost("openrouter", None, None, None, error="not found")
        s3 = SourceCost("tokencost", 0.004, 2.5, 12.0)
        result = self._make_result([s1, s2, s3])
        assert result.available_sources == [s1, s3]

    def test_available_sources_empty_when_all_unavailable(self):
        s1 = SourceCost("litellm", None, None, None, error="err")
        s2 = SourceCost("openrouter", None, None, None, error="err")
        result = self._make_result([s1, s2])
        assert result.available_sources == []

    def test_available_sources_all_when_all_available(self):
        s1 = SourceCost("litellm", 0.001, 1.0, 2.0)
        s2 = SourceCost("openrouter", 0.002, 1.5, 3.0)
        result = self._make_result([s1, s2])
        assert result.available_sources == [s1, s2]

    def test_to_dict_structure(self):
        s = SourceCost("litellm", 0.005, 3.0, 15.0)
        result = CostResult(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            sources=[s],
        )
        d = result.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert len(d["costs"]) == 1
        cost = d["costs"][0]
        assert cost["source"] == "litellm"
        assert cost["total_cost_usd"] == 0.005
        assert cost["price_per_million_input"] == 3.0
        assert cost["price_per_million_output"] == 15.0
        assert cost["error"] is None

    def test_to_dict_includes_error_field(self):
        s = SourceCost("litellm", None, None, None, error="Model not found")
        result = CostResult(
            model="unknown", input_tokens=10, output_tokens=5, sources=[s]
        )
        d = result.to_dict()
        assert d["costs"][0]["error"] == "Model not found"
        assert d["costs"][0]["total_cost_usd"] is None

    def test_to_dict_multiple_sources(self):
        sources = [
            SourceCost("litellm", 0.001, 1.0, 2.0),
            SourceCost("openrouter", 0.002, 1.5, 3.0),
            SourceCost("tokencost", None, None, None, error="err"),
        ]
        result = CostResult(
            model="gpt-4o", input_tokens=10, output_tokens=5, sources=sources
        )
        d = result.to_dict()
        assert len(d["costs"]) == 3
        assert d["costs"][0]["source"] == "litellm"
        assert d["costs"][1]["source"] == "openrouter"
        assert d["costs"][2]["source"] == "tokencost"

    def test_single_source_default_true(self):
        result = CostResult(model="m", input_tokens=1, output_tokens=1, sources=[])
        assert result.single_source is True

    def test_single_source_false_when_set(self):
        result = CostResult(
            model="m", input_tokens=1, output_tokens=1, sources=[], single_source=False
        )
        assert result.single_source is False

    def test_to_dict_omits_new_fields_when_none(self):
        """New price fields must not appear in output when they are None."""
        s = SourceCost("litellm", 0.005, 3.0, 15.0)
        result = CostResult(
            model="gpt-4o", input_tokens=100, output_tokens=50, sources=[s]
        )
        d = result.to_dict()
        cost = d["costs"][0]
        assert "price_per_million_cache_read" not in cost
        assert "price_per_million_cache_creation" not in cost
        assert "price_per_million_reasoning" not in cost

    def test_to_dict_includes_new_fields_when_present(self):
        """New price fields must appear in output when they are set."""
        s = SourceCost(
            "litellm",
            0.005,
            3.0,
            15.0,
            price_per_million_cache_read=0.75,
            price_per_million_cache_creation=3.75,
            price_per_million_reasoning=15.0,
        )
        result = CostResult(
            model="gpt-4o", input_tokens=100, output_tokens=50, sources=[s]
        )
        d = result.to_dict()
        cost = d["costs"][0]
        assert cost["price_per_million_cache_read"] == 0.75
        assert cost["price_per_million_cache_creation"] == 3.75
        assert cost["price_per_million_reasoning"] == 15.0

    def test_to_dict_partial_new_fields(self):
        """Only non-None new fields should appear."""
        s = SourceCost(
            "litellm",
            0.005,
            3.0,
            15.0,
            price_per_million_cache_read=0.75,
        )
        result = CostResult(
            model="gpt-4o", input_tokens=100, output_tokens=50, sources=[s]
        )
        d = result.to_dict()
        cost = d["costs"][0]
        assert cost["price_per_million_cache_read"] == 0.75
        assert "price_per_million_cache_creation" not in cost
        assert "price_per_million_reasoning" not in cost

    def test_token_counts_default_to_zero(self):
        result = CostResult(model="m", input_tokens=1, output_tokens=1, sources=[])
        assert result.cached_input_tokens == 0
        assert result.cache_creation_input_tokens == 0
        assert result.reasoning_tokens == 0

    def test_to_dict_omits_token_counts_when_zero(self):
        s = SourceCost("litellm", 0.005, 3.0, 15.0)
        result = CostResult(
            model="gpt-4o", input_tokens=100, output_tokens=50, sources=[s]
        )
        d = result.to_dict()
        assert "cached_input_tokens" not in d
        assert "cache_creation_input_tokens" not in d
        assert "reasoning_tokens" not in d

    def test_to_dict_includes_token_counts_when_nonzero(self):
        s = SourceCost("litellm", 0.005, 3.0, 15.0)
        result = CostResult(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            sources=[s],
            cached_input_tokens=200,
            cache_creation_input_tokens=50,
            reasoning_tokens=30,
        )
        d = result.to_dict()
        assert d["cached_input_tokens"] == 200
        assert d["cache_creation_input_tokens"] == 50
        assert d["reasoning_tokens"] == 30

    def test_to_dict_partial_token_counts(self):
        """Only nonzero token counts appear."""
        s = SourceCost("litellm", 0.005, 3.0, 15.0)
        result = CostResult(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            sources=[s],
            cached_input_tokens=200,
        )
        d = result.to_dict()
        assert d["cached_input_tokens"] == 200
        assert "cache_creation_input_tokens" not in d
        assert "reasoning_tokens" not in d
