"""Gemini API client: rate limiting, retries, JSON-schema output, quota rotation.

Google's current free tier allows ~20 generateContent requests per day, per
project, *per model*. To make a 50-contract run reproducible on the free tier,
the client rotates through a fallback chain of (api_key, model) combinations:
when one model's daily bucket is spent (HTTP 429 with a ``PerDay`` quotaId) or
a model isn't available to the project (404), it advances to the next combo
and keeps going. Per-minute 429s and transient network errors are retried
with exponential backoff instead.

Uses the plain REST API (no heavy SDK) so the project stays dependency-light.
"""

import json
import time

import requests

from . import config

_last_request_ts = 0.0
_combo_index = 0  # current position in the (key, model) fallback chain


def _combos() -> list[tuple[str, str]]:
    return [(k, m) for k in config.GEMINI_API_KEYS for m in config.GENERATION_MODELS]


def current_model() -> str:
    return _combos()[_combo_index][1]


def _throttle() -> None:
    """Client-side pacing to stay inside free-tier requests-per-minute."""
    global _last_request_ts
    min_interval = 60.0 / config.REQUESTS_PER_MINUTE
    wait = _last_request_ts + min_interval - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.time()


def _is_daily_quota(resp: requests.Response) -> bool:
    """True when a 429 is the per-day bucket (rotating helps, waiting doesn't)."""
    try:
        details = resp.json()["error"].get("details", [])
        for det in details:
            for violation in det.get("violations", []):
                if "PerDay" in violation.get("quotaId", ""):
                    return True
    except Exception:
        pass
    return False


def _advance_combo(reason: str) -> None:
    global _combo_index
    combos = _combos()
    _combo_index += 1
    if _combo_index >= len(combos):
        raise RuntimeError("retries exhausted: all (api_key, model) fallbacks spent "
                           f"(last: {reason})")
    key, model = combos[_combo_index]
    print(f"[llm] {reason} -> switching to model '{model}' "
          f"(fallback {_combo_index + 1}/{len(combos)})")


def _post_generate(payload: dict) -> dict:
    attempt = 0
    while True:
        key, model = _combos()[_combo_index]
        url = f"{config.API_BASE}/models/{model}:generateContent?key={key}"
        _throttle()
        try:
            resp = requests.post(url, json=payload, timeout=300)
        except requests.RequestException as exc:  # resets, timeouts, DNS blips
            attempt += 1
            if attempt >= config.MAX_RETRIES:
                raise RuntimeError(f"retries exhausted: network ({exc})") from exc
            delay = min(2 ** attempt * 5, 60)
            print(f"[llm] network error ({type(exc).__name__}), retrying in {delay}s")
            time.sleep(delay)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 404:  # model not served for this project
            _advance_combo(f"model '{model}' unavailable (404)")
            attempt = 0
            continue

        if resp.status_code == 429 and _is_daily_quota(resp):
            _advance_combo(f"daily quota spent for '{model}'")
            attempt = 0
            continue

        if resp.status_code in (429, 500, 502, 503):  # per-minute limit / transient
            attempt += 1
            if attempt >= config.MAX_RETRIES:
                _advance_combo(f"persistent HTTP {resp.status_code} on '{model}'")
                attempt = 0
                continue
            delay = min(2 ** attempt * 5, 60)
            print(f"[llm] HTTP {resp.status_code}, retrying in {delay}s "
                  f"(attempt {attempt}/{config.MAX_RETRIES})")
            time.sleep(delay)
            continue

        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")


def generate_json(prompt: str, schema: dict, temperature: float = 0.0) -> dict:
    """Call the model in JSON mode, constrained to the given response schema."""
    if not config.GEMINI_API_KEYS or not config.GEMINI_API_KEYS[0]:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env "
                           "and add your key (https://aistudio.google.com/apikey).")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    data = _post_generate(payload)
    try:
        parts = data["candidates"][0]["content"]["parts"]
        # Thinking-capable models may emit thought parts before the answer;
        # the JSON payload is in the non-thought text parts.
        text = "".join(p["text"] for p in parts
                       if "text" in p and not p.get("thought"))
        return json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {json.dumps(data)[:500]}") from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts (embeddings have a separate, roomier quota)."""
    key = config.GEMINI_API_KEYS[0]
    url = (f"{config.API_BASE}/models/{config.EMBEDDING_MODEL}:batchEmbedContents"
           f"?key={key}")
    payload = {
        "requests": [
            {
                "model": f"models/{config.EMBEDDING_MODEL}",
                "content": {"parts": [{"text": t[:8000]}]},
                "outputDimensionality": 768,
            }
            for t in texts
        ]
    }
    for attempt in range(config.MAX_RETRIES):
        _throttle()
        try:
            resp = requests.post(url, json=payload, timeout=120)
        except requests.RequestException:
            time.sleep(min(2 ** attempt * 5, 60))
            continue
        if resp.status_code == 200:
            return [e["values"] for e in resp.json()["embeddings"]]
        if resp.status_code in (429, 500, 502, 503):
            time.sleep(min(2 ** attempt * 5, 60))
            continue
        raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("Embedding API: retries exhausted")
