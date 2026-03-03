import json
import time
from pathlib import Path

CACHE_FILE = Path.home() / ".howmuch_cache.json"
CACHE_TTL = 3600


def load_cache(namespace: str) -> dict:
    if not CACHE_FILE.exists():
        return {}
    data = json.loads(CACHE_FILE.read_text())
    entry = data.get(namespace, {})
    if time.time() - entry.get("_ts", 0) > CACHE_TTL:
        return {}
    return entry.get("models", {})


def save_cache(namespace: str, models: dict) -> None:
    data = {}
    if CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text())
    data[namespace] = {"_ts": time.time(), "models": models}
    CACHE_FILE.write_text(json.dumps(data))
