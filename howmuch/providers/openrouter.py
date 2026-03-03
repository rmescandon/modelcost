import httpx
from .cache import load_cache, save_cache

OPENROUTER_API = "https://openrouter.ai/api/v1/models"

def fetch_openrouter_prices(use_cache: bool = True) -> dict:
    if use_cache:
        cached = load_cache("openrouter")
        if cached:
            return cached

    resp = httpx.get(OPENROUTER_API, timeout=10)
    resp.raise_for_status()

    models = {
        m["id"]: {
            "prompt": float(m["pricing"]["prompt"]),
            "completion": float(m["pricing"]["completion"]),
        }
        for m in resp.json().get("data", [])
        if "pricing" in m
    }
    save_cache("openrouter", models)
    return models

def find_model(model_id: str, prices: dict) -> dict | None:
    if model_id in prices:
        return prices[model_id]
    matches = [v for k, v in prices.items() if k.endswith(f"/{model_id}")]
    return matches[0] if len(matches) == 1 else None
