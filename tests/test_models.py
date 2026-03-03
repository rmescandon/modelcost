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
