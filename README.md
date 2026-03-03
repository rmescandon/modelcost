# llmcost

Calculate LLM API call costs from token usage using price catalogs from multiple sources.

Supported pricing sources:
- `litellm` (default)
- `openrouter`
- `tokencost`

## Install

```bash
python -m pip install llmcost
```

## CLI

The default command calculates cost, so you can omit the `cost` subcommand.

```bash
# Default (cost)
llmcost gpt-4o 1000 500

# Explicit cost (optional)
llmcost cost gpt-4o 1000 500

# All sources in one run
llmcost --source all gpt-4o 1000 500

# JSON output
llmcost --json gpt-4o 1000 500
```

List available models:

```bash
llmcost models
llmcost models --source openrouter
llmcost models --filter gpt
llmcost models --json
```

CLI help:

```bash
llmcost --help
llmcost models --help
```

## Library

```python
from llmcost.calculator import calculate_cost, list_models

result = calculate_cost("gpt-4o", 1000, 500)

for source in result.available_sources:
    print(f"{source.source}: ${source.total_cost_usd:.6f}")

litellm_cost = next(s for s in result.sources if s.source == "litellm")
print(litellm_cost.price_per_million_input, litellm_cost.price_per_million_output)

models = list_models("openrouter")
```

## Output details

`calculate_cost()` returns a `CostResult` with:
- `model`, `input_tokens`, `output_tokens`
- `sources`: list of `SourceCost` objects
- `available_sources`: only sources with prices found

Each `SourceCost` includes:
- `source`
- `total_cost_usd`
- `price_per_million_input`
- `price_per_million_output`
- `error` (when not available)

## Caching

`openrouter` responses are cached in `~/.llm_cost_cache.json` for 1 hour.

## Notes

- Prices are fetched at runtime from the upstream catalogs.
- If a model is missing in a source, that source is marked as unavailable.
- Network sources are fetched in parallel for the `all` option.
