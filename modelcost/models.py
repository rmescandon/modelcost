from dataclasses import dataclass


@dataclass
class SourceCost:
    source: str
    total_cost_usd: float | None
    price_per_million_input: float | None
    price_per_million_output: float | None
    error: str | None = None
    price_per_million_cache_read: float | None = None
    price_per_million_cache_creation: float | None = None
    price_per_million_reasoning: float | None = None

    @property
    def available(self) -> bool:
        return self.total_cost_usd is not None


@dataclass
class CostResult:
    model: str
    input_tokens: int
    output_tokens: int
    sources: list[SourceCost]
    single_source: bool = True  # False when source="all"
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def available_sources(self) -> list[SourceCost]:
        return [s for s in self.sources if s.available]

    def to_dict(self) -> dict:
        costs = []
        for s in self.sources:
            entry: dict = {
                "source": s.source,
                "total_cost_usd": s.total_cost_usd,
                "price_per_million_input": s.price_per_million_input,
                "price_per_million_output": s.price_per_million_output,
                "error": s.error,
            }
            if s.price_per_million_cache_read is not None:
                entry["price_per_million_cache_read"] = s.price_per_million_cache_read
            if s.price_per_million_cache_creation is not None:
                entry["price_per_million_cache_creation"] = (
                    s.price_per_million_cache_creation
                )
            if s.price_per_million_reasoning is not None:
                entry["price_per_million_reasoning"] = s.price_per_million_reasoning
            costs.append(entry)
        result: dict = {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "costs": costs,
        }
        if self.cached_input_tokens > 0:
            result["cached_input_tokens"] = self.cached_input_tokens
        if self.cache_creation_input_tokens > 0:
            result["cache_creation_input_tokens"] = self.cache_creation_input_tokens
        if self.reasoning_tokens > 0:
            result["reasoning_tokens"] = self.reasoning_tokens
        return result
