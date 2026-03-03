import httpx

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

def fetch_litellm_prices() -> dict:
    resp = httpx.get(LITELLM_URL, timeout=10)
    resp.raise_for_status()
    return {
        model: {
            "prompt": info["input_cost_per_token"],
            "completion": info["output_cost_per_token"],
        }
        for model, info in resp.json().items()
        if "input_cost_per_token" in info and "output_cost_per_token" in info
    }
