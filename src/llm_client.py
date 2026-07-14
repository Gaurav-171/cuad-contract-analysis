"""Thin Gemini API client with rate limiting, retries and JSON-schema output.

Uses the plain REST API (no heavy SDK) so the project stays dependency-light
and easy to reproduce. Swap ``GENERATION_MODEL`` in ``.env`` to try other
models without touching code.
"""

import json
import time

import requests

from . import config

_last_request_ts = 0.0


def _throttle() -> None:
    """Client-side pacing to stay inside free-tier requests-per-minute."""
    global _last_request_ts
    min_interval = 60.0 / config.REQUESTS_PER_MINUTE
    wait = _last_request_ts + min_interval - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.time()


def _post(url: str, payload: dict) -> dict:
    """POST with exponential backoff on rate limits and transient errors."""
    for attempt in range(config.MAX_RETRIES):
        _throttle()
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503):
            delay = min(2 ** attempt * 5, 60)
            print(f"[llm] HTTP {resp.status_code}, retrying in {delay}s "
                  f"(attempt {attempt + 1}/{config.MAX_RETRIES})")
            time.sleep(delay)
            continue
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")
    raise RuntimeError("Gemini API: retries exhausted")


def generate_json(prompt: str, schema: dict, temperature: float = 0.0) -> dict:
    """Call the model in JSON mode, constrained to the given response schema."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env "
                           "and add your key (https://aistudio.google.com/apikey).")
    url = (f"{config.API_BASE}/models/{config.GENERATION_MODEL}:generateContent"
           f"?key={config.GEMINI_API_KEY}")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    data = _post(url, payload)
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {json.dumps(data)[:500]}") from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the Gemini embedding model."""
    url = (f"{config.API_BASE}/models/{config.EMBEDDING_MODEL}:batchEmbedContents"
           f"?key={config.GEMINI_API_KEY}")
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
    data = _post(url, payload)
    return [e["values"] for e in data["embeddings"]]
