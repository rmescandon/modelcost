# modelcost

Calculate LLM API call costs from token usage using price catalogs from multiple sources.

Supported pricing sources:
- `litellm` (default)
- `openrouter`
- `tokencost`

## Install

```bash
python -m pip install modelcost
```

## CLI

The default command calculates cost, so you can omit the `cost` subcommand.

```bash
# Default (cost)
modelcost gpt-4o 1000 500

# Explicit cost (optional)
modelcost cost gpt-4o 1000 500

# All sources in one run
modelcost --source all gpt-4o 1000 500

# JSON output
modelcost --json gpt-4o 1000 500
```

### Cached and reasoning tokens

Modern LLM APIs charge differently for cached input and reasoning output tokens.
Pass them as optional flags — they default to 0 so existing usage is unchanged.

```bash
# Cached input tokens (served from prompt cache)
modelcost gpt-4o 1000 500 --cached-input-tokens 200

# Cache creation tokens (first-time cache writes)
modelcost gpt-4o 1000 500 --cache-creation-input-tokens 100

# Reasoning tokens (subset of output_tokens, e.g. o1/R1 thinking)
modelcost deepseek/deepseek-r1 2000 5000 --reasoning-tokens 3000

# All together
modelcost gpt-4.1-mini 1000 500 \
  --cached-input-tokens 200 \
  --cache-creation-input-tokens 100 \
  --reasoning-tokens 150
```

### List models

```bash
modelcost models
modelcost models --source openrouter
modelcost models --filter gpt
modelcost models --json
```

### Help

```bash
modelcost --help
modelcost models --help
```

## Library

```python
from modelcost.calculator import calculate_cost, list_models

# Basic usage (backward compatible)
result = calculate_cost("gpt-4o", 1000, 500)

for source in result.available_sources:
    print(f"{source.source}: ${source.total_cost_usd:.6f}")

# With cached and reasoning tokens
result = calculate_cost(
    "gpt-4.1-mini",
    input_tokens=1000,
    output_tokens=500,
    cached_input_tokens=200,
    cache_creation_input_tokens=100,
    reasoning_tokens=150,
)

s = result.sources[0]
print(f"${s.total_cost_usd:.6f}")
print(f"  cache read:  ${s.price_per_million_cache_read}/M")
print(f"  reasoning:   ${s.price_per_million_reasoning}/M")

models = list_models("openrouter")
```

## Output details

`calculate_cost()` returns a `CostResult` with:
- `model`, `input_tokens`, `output_tokens`
- `cached_input_tokens`, `cache_creation_input_tokens`, `reasoning_tokens` (0 when not used)
- `sources`: list of `SourceCost` objects
- `available_sources`: only sources with prices found

Each `SourceCost` includes:
- `source`
- `total_cost_usd`
- `price_per_million_input`, `price_per_million_output`
- `price_per_million_cache_read`, `price_per_million_cache_creation`, `price_per_million_reasoning` (present only when the source has specific pricing for these)
- `error` (when not available)

## Caching

`openrouter` responses are cached in `~/.modelcost_cache.json` for 1 hour.

## Notes

- Prices are fetched at runtime from the upstream catalogs.
- If a model is missing in a source, that source is marked as unavailable.
- Network sources are fetched in parallel for the `all` option.
- `tokencost` does not expose cache/reasoning pricing — when used with `source="all"`, its cost may be higher than `litellm` for calls that include cached or reasoning tokens.
