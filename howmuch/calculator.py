from concurrent.futures import ThreadPoolExecutor, as_completed
from .models import CostResult, SourceCost
from .providers.openrouter import fetch_openrouter_prices, find_model
from .providers.litellm import fetch_litellm_prices

VALID_SOURCES = ("litellm", "openrouter", "tokencost", "all")


def list_models(source: str = "litellm") -> list[str]:
    """Return the available models for the given source."""
    if source not in ("litellm", "openrouter", "tokencost"):
        raise ValueError(
            f"Invalid source '{source}'. Valid values: litellm, openrouter, tokencost"
        )

    if source == "litellm":
        return sorted(fetch_litellm_prices().keys())

    if source == "openrouter":
        return sorted(fetch_openrouter_prices().keys())

    if source == "tokencost":
        from tokencost.constants import TOKEN_COSTS

        return sorted(TOKEN_COSTS.keys())


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    source: str = "litellm",
) -> CostResult:
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source '{source}'. Valid values: {VALID_SOURCES}")

    active = ["litellm", "openrouter", "tokencost"] if source == "all" else [source]

    sources = _fetch_all(model, input_tokens, output_tokens, active)

    return CostResult(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        sources=sources,
        single_source=source != "all",  # output formatting flag
    )


def _fetch_all(
    model: str, input_tokens: int, output_tokens: int, active: list[str]
) -> list[SourceCost]:
    network_tasks = {
        name: fn
        for name, fn in {
            "litellm": fetch_litellm_prices,
            "openrouter": fetch_openrouter_prices,
        }.items()
        if name in active
    }

    results: dict[str, SourceCost] = {}

    if network_tasks:
        with ThreadPoolExecutor(max_workers=len(network_tasks)) as executor:
            futures = {
                executor.submit(
                    _compute, name, fn, model, input_tokens, output_tokens
                ): name
                for name, fn in network_tasks.items()
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    if "tokencost" in active:
        results["tokencost"] = _tokencost_source(model, input_tokens, output_tokens)

    # Preserve order: litellm -> openrouter -> tokencost
    order = ["litellm", "openrouter", "tokencost"]
    return [results[name] for name in order if name in results]


def _compute(
    source_name: str, fetch_fn, model: str, input_tokens: int, output_tokens: int
) -> SourceCost:
    try:
        prices = fetch_fn()
        pricing = (
            find_model(model, prices)
            if source_name == "openrouter"
            else prices.get(model)
        )
        if pricing is None:
            return SourceCost(
                source=source_name,
                total_cost_usd=None,
                price_per_million_input=None,
                price_per_million_output=None,
                error=f"Model '{model}' not found",
            )
        cost = pricing["prompt"] * input_tokens + pricing["completion"] * output_tokens
        return SourceCost(
            source=source_name,
            total_cost_usd=cost,
            price_per_million_input=pricing["prompt"] * 1_000_000,
            price_per_million_output=pricing["completion"] * 1_000_000,
        )
    except Exception as e:
        return SourceCost(
            source=source_name,
            total_cost_usd=None,
            price_per_million_input=None,
            price_per_million_output=None,
            error=str(e),
        )


def _tokencost_source(model: str, input_tokens: int, output_tokens: int) -> SourceCost:
    try:
        from tokencost.costs import calculate_cost_by_tokens
        from tokencost.constants import TOKEN_COSTS

        input_cost = calculate_cost_by_tokens(input_tokens, model, "input")
        output_cost = calculate_cost_by_tokens(output_tokens, model, "output")
        entry = TOKEN_COSTS[model.lower()]

        return SourceCost(
            source="tokencost",
            total_cost_usd=float(input_cost + output_cost),
            price_per_million_input=float(entry["input_cost_per_token"]) * 1_000_000,
            price_per_million_output=float(entry["output_cost_per_token"]) * 1_000_000,
        )
    except KeyError as e:
        return SourceCost(
            source="tokencost",
            total_cost_usd=None,
            price_per_million_input=None,
            price_per_million_output=None,
            error=f"Model not found: {e}",
        )
    except Exception as e:
        return SourceCost(
            source="tokencost",
            total_cost_usd=None,
            price_per_million_input=None,
            price_per_million_output=None,
            error=str(e),
        )
