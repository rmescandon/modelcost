from dataclasses import dataclass, field


@dataclass
class SourceCost:
    source: str
    total_cost_usd: float | None
    price_per_million_input: float | None
    price_per_million_output: float | None
    error: str | None = None

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

    @property
    def available_sources(self) -> list[SourceCost]:
        return [s for s in self.sources if s.available]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "costs": [
                {
                    "source": s.source,
                    "total_cost_usd": s.total_cost_usd,
                    "price_per_million_input": s.price_per_million_input,
                    "price_per_million_output": s.price_per_million_output,
                    "error": s.error,
                }
                for s in self.sources
            ],
        }
