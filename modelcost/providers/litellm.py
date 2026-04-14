import httpx

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)


def fetch_litellm_prices() -> dict:
    resp = httpx.get(LITELLM_URL, timeout=10)
    resp.raise_for_status()
    prices = {}
    for model, info in resp.json().items():
        if "input_cost_per_token" not in info or "output_cost_per_token" not in info:
            continue
        entry: dict = {
            "prompt": info["input_cost_per_token"],
            "completion": info["output_cost_per_token"],
        }
        cache_read = info.get(
            "cache_read_input_token_cost",
            info.get("input_cost_per_token_cache_hit"),
        )
        if cache_read is not None:
            entry["cache_read"] = cache_read
        cache_creation = info.get("cache_creation_input_token_cost")
        if cache_creation is not None:
            entry["cache_creation"] = cache_creation
        reasoning = info.get("output_cost_per_reasoning_token")
        if reasoning is not None:
            entry["reasoning"] = reasoning
        prices[model] = entry
    return prices
