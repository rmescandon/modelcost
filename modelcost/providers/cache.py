import json
import time
from pathlib import Path
from uuid import uuid4

CACHE_FILE = Path.home() / ".modelcost_cache.json"
CACHE_TTL = 3600


def load_cache(namespace: str) -> dict:
    if not CACHE_FILE.exists():
        return {}

    try:
        data = json.loads(CACHE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    entry = data.get(namespace, {})
    if time.time() - entry.get("_ts", 0) > CACHE_TTL:
        return {}
    return entry.get("models", {})


def save_cache(namespace: str, models: dict) -> None:
    data = {}
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}

    data[namespace] = {"_ts": time.time(), "models": models}
    tmp_path = CACHE_FILE.with_name(f"{CACHE_FILE.name}.{uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(data))
    tmp_path.replace(CACHE_FILE)
